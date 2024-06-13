# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Install the OpenGL library needed by OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Set the working directory in the container
WORKDIR /home/atharav/react/docker_app/app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /home/atharav/react/docker_app/app

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
