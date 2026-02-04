#!/bin/bash

# Build the Docker image
echo "Building the docker image..."
docker build -t shelly-training -f dockerfile_model_training .

echo "Running the docker container..."
#docker run --name shelly-training-container -d --gpus all -v "$(pwd)":/app shelly-training
docker run --name shelly-training-container -d --gpus all -v "$(pwd)":/app shelly-training > training_logs_llama.txt 2>&1

echo "Container is running..."
