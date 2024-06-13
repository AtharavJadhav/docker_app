import os
import numpy as np
import cv2
from scipy.stats import entropy
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List

app = FastAPI()

# Load the ONNX model (Example: mnist_model.onnx)
session = ort.InferenceSession("mnist_model.onnx")

# Calculate the histogram of an image
def calculate_histogram(image, bins=256):
    histogram = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram

# Compute histograms for all images in a directory
def compute_reference_histograms(image_dir):
    histograms = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            if image is not None:
                histogram = calculate_histogram(image)
                histograms.append(histogram)
    return histograms

# Path to the directory containing training images
training_image_dir = "training_images"

# Compute reference histograms
reference_histograms = compute_reference_histograms(training_image_dir)

# Compute the average reference histogram
average_reference_histogram = np.mean(reference_histograms, axis=0)

# In-memory store for metrics
metrics_store = {
    "mnist_model_onnx": {
        "accuracy": 0.98,
        "latency": 0.0025,
        "throughput": 100,
        "request_count": 1000,
        "kl_divergence": 0.1,
        "average_reference_histogram": average_reference_histogram.tolist(),
        "average_inference_histogram": [],
        "retraining_needed": "Not Needed"
    }
}

# Threshold for KL divergence
KL_THRESHOLD = 0.5

# Calculate KL divergence between two histograms
def calculate_kl_divergence(p, q):
    p = p / np.sum(p)
    q = q / np.sum(q)
    return entropy(p, q)

# Calculate KL divergence and average inference histogram
def get_histogram_kl_divergence(new_images, reference_histograms):
    kl_divergences = []
    inference_histograms = []
    for new_image in new_images:
        new_histogram = calculate_histogram(new_image)
        inference_histograms.append(new_histogram)
        kl_divergence = np.mean([calculate_kl_divergence(new_histogram, ref_hist) for ref_hist in reference_histograms])
        kl_divergences.append(kl_divergence)
    average_inference_histogram = np.mean(inference_histograms, axis=0)
    return kl_divergences, average_inference_histogram

@app.post("/compute_kl")
async def compute_kl(images: List[UploadFile] = File(...)):
    new_images = []
    for image in images:
        image_data = await image.read()
        image_np = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        new_images.append(img)
    
    kl_divergences, average_inference_histogram = get_histogram_kl_divergence(new_images, reference_histograms)
    
    # Update metrics_store with the latest KL divergence value and average inference histogram
    if kl_divergences:
        kl_value = kl_divergences[0]
        metrics_store["mnist_model_onnx"]["kl_divergence"] = kl_value
        metrics_store["mnist_model_onnx"]["average_inference_histogram"] = average_inference_histogram.tolist()
        if kl_value > KL_THRESHOLD:
            metrics_store["mnist_model_onnx"]["retraining_needed"] = "Retraining Needed"
        else:
            metrics_store["mnist_model_onnx"]["retraining_needed"] = "Not Needed"
    
    return JSONResponse(content={
        "kl_divergences": kl_divergences,
        "average_inference_histogram": average_inference_histogram.tolist(),
        "retraining_needed": metrics_store["mnist_model_onnx"]["retraining_needed"]
    })

@app.get("/metrics")
async def metrics():
    # Retrieve metrics from the in-memory store
    return JSONResponse(content=metrics_store)

if _name_ == '_main_':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)