import os
import shutil
import time
import logging
import docker
import random
import string
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List
import subprocess
import pandas as pd

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store model state and metrics
model_names = set()  # Using a set to ensure unique model names
triton_container = None

# Metrics dictionary
metrics = {}


class DeployModelsRequest(BaseModel):
    models: List[str] = Field(..., example=["mnist_model_onnx", "mnist_model_openvino"])

class InferenceRequest(BaseModel):
    model_name: str
    correct: bool = None
class MetricsRequest(BaseModel):
    model_name: str

@app.on_event("startup")
async def startup_event():
    global docker_client
    docker_client = docker.from_env()

@app.post("/deploy/")
async def deploy_models(request_body: DeployModelsRequest):
    models = request_body.models
    global triton_container, model_names

    # Check if a triton_server container is already running
    triton_containers = docker_client.containers.list(filters={"name": "triton_server"})
    if triton_containers:
        logger.info("A triton_server container is already running. It will be replaced with a new one.")
        for container in triton_containers:
            container.remove(force=True)

    required_models = {"mnist_model_onnx", "mnist_model_openvino", "mnist_model_pytorch", "bert_model_onnx"}
    if not all(model in required_models for model in models):
        return JSONResponse(status_code=422, content={"detail": "Invalid model names. Use the exact model names."})

    docker_run_command = [
        'tritonserver', '--model-repository=/models',
        '--model-control-mode=explicit'
    ]

    for model in models:
        docker_run_command.extend(['--load-model', model])

    try:
        # Debugging: print current working directory and model repository path
        current_working_directory = os.getcwd()
        parent_directory = os.path.basename(os.path.dirname(current_working_directory))
        network_name = f"{parent_directory}_my_network"
        model_repo_path = os.path.join(current_working_directory, 'model_repository')
        logger.info(f"Current working directory: {current_working_directory}")
        logger.info(f"Model repository path: {model_repo_path}")
        logger.info(f"Network name: {network_name}")
        
        # Run the triton_server container with the specified models and connect it to the same network
        triton_container = docker_client.containers.run(
            'nvcr.io/nvidia/tritonserver:24.05-py3',
            docker_run_command,
            detach=True,
            name="triton_server",
            network=network_name,
            ports={'8000/tcp': 8000, '8001/tcp': 8001, '8002/tcp': 8002},
            volumes={model_repo_path: {'bind': '/models', 'mode': 'rw'}},
            shm_size="256m",
            environment={"TRITON_SERVER_CPU_ONLY": "1"}
        )

        model_names = set(models)
        for model in models:
            metrics[model] = {
                "accuracy": 0,
                "latency": 0,
                "throughput": 0,
                "request_count": 0,
                "correct_count": 0,
                "total_latency": 0,
                "data_shift": 0,
                "mean_shift": 0,
                "std_shift": 0
            }

        return {"message": "Models deployed successfully"}
    except Exception as e:
        logger.error(f"Failed to deploy models: {e}")
        return JSONResponse(status_code=500, content={"detail": "Failed to deploy models"})

@app.get("/logs/")
async def get_triton_logs():
    global triton_container
    if not triton_container:
        return {"message": "No Triton container running"}

    try:
        logs = triton_container.logs().decode('utf-8')
        return {"logs": logs}
    except Exception as e:
        logger.error(f"Error fetching logs: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": "Failed to fetch logs"})

@app.post("/inference/")
async def inference(model_name: str = Form(...), file: UploadFile = File(...)):
    if model_name not in metrics:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        # Create the directory for storing uploaded files if it doesn't exist
        input_dir = "inference_inputs"
        permanent_dir = os.path.join("permanent_storage", model_name)
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(permanent_dir, exist_ok=True)

        # Generate a random 10-character filename
        random_filename = ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + os.path.splitext(file.filename)[1]
        permanent_file_path = os.path.join(permanent_dir, random_filename)

        # Save the uploaded file permanently
        with open(permanent_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Also save the uploaded file for inference processing
        file_path = os.path.join(input_dir, random_filename)
        with open(file_path, "wb") as buffer:
            file.file.seek(0)
            shutil.copyfileobj(file.file, buffer)

        # Map model names to their corresponding scripts
        model_to_script = {
            "mnist_model_onnx": "client_mnist_onnx.py",
            "mnist_model_openvino": "client_mnist_openvino.py",
            "mnist_model_pytorch": "client_mnist_pytorch.py",
            "bert_model_onnx": "client_bert_onnx.py",
        }
        script_name = model_to_script.get(model_name)
        if not script_name:
            raise HTTPException(status_code=400, detail="Invalid model name")

        # Run the model-specific inference script
        start_time = time.time()
        script_path = f"clientfiles/{script_name}"
        subprocess.run(["python3", script_path, file_path])
        end_time = time.time()
        latency = end_time - start_time

        metrics[model_name]["request_count"] += 1
        metrics[model_name]["total_latency"] += latency
        metrics[model_name]["latency"] = metrics[model_name]["total_latency"] / metrics[model_name]["request_count"]
        metrics[model_name]["throughput"] = metrics[model_name]["request_count"] / (metrics[model_name]["latency"] or 1)

        return {"message": "Inference request processed"}

    finally:
        # Remove the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/results/")
async def get_results():
    results_file = "inference_results/results.txt"
    if os.path.exists(results_file):
        with open(results_file, "r") as file:
            result = file.read()
        # Clear the file after reading
        open(results_file, "w").close()
        return {"result": result}
    else:
        raise HTTPException(status_code=404, detail="Results not found")

@app.post("/feedback/")
async def submit_feedback(request_body: InferenceRequest):
    model_name = request_body.model_name
    correct = request_body.correct

    if model_name not in metrics:
        raise HTTPException(status_code=404, detail="Model not found")

    if correct is not None:
        if correct:
            metrics[model_name]["correct_count"] += 1
        metrics[model_name]["accuracy"] = metrics[model_name]["correct_count"] / metrics[model_name]["request_count"]

    return {"message": "Feedback submitted"}

@app.get("/metrics/")
async def get_metrics():
    return metrics

# Endpoint to calculate data shift metrics
@app.post("/calculate_shift_metrics/")
async def get_data_shift_metrics(request_body: MetricsRequest):
    model_name = request_body.model_name
    try:
        # Load training data
        training_dir = os.path.join("training_images", model_name)
        if not os.path.exists(training_dir):
            raise HTTPException(status_code=404, detail=f"Training data not found for model: {model_name}")

        # Load newly uploaded data
        storage_dir = os.path.join("permanent_storage", model_name)
        if not os.path.exists(storage_dir):
            raise HTTPException(status_code=404, detail=f"No current data found for model: {model_name}")

        # Assuming both training_dir and storage_dir contain image files
   
        # Step 1: Get all image file paths
        training_images = [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.endswith('.jpg') or f.endswith('.png')]
        storage_images = [os.path.join(storage_dir, f) for f in os.listdir(storage_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
        # Step 2: Calculate statistics on images
        def calculate_stats(images):
            means = []
            stds = []
            for img_path in images:
                img = cv2.imread(img_path)
                # Calculate mean and standard deviation for each channel
                mean, std = cv2.meanStdDev(img)
                means.append(mean.flatten())
                stds.append(std.flatten())
            return np.mean(means, axis=0), np.mean(stds, axis=0)
    
        # Step 3: Compute statistics for both sets of images
        training_mean, training_std = calculate_stats(training_images)
        storage_mean, storage_std = calculate_stats(storage_images)
    
        # Step 4: Calculate overall data shift
        overall_mean_shift = np.linalg.norm(training_mean - storage_mean)
        overall_std_shift = np.linalg.norm(training_std - storage_std)
    
        overall_data_shift = np.sqrt(overall_mean_shift**2 + overall_std_shift**2) / 100
    
        metrics[model_name]["data_shift"] = overall_data_shift
        metrics[model_name]["mean_shift"] = overall_mean_shift
        metrics[model_name]["std_shift"] = overall_std_shift

        return {"data_shift": overall_data_shift, "mean_shift": overall_mean_shift, "std_shift": overall_std_shift, "model_name": model_name, "status": "success", "message": "Data shift metrics calculated successfully"}

    except Exception as e:
        logger.error(f"Error calculating data shift metrics: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": "Failed to calculate data shift metrics"})
