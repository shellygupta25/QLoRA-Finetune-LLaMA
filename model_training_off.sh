#!/bin/bash

echo "Stopping the docker container..."
docker stop shelly-training-container

echo "Removing the docker container..."
docker rm shelly-training-container

echo "Removing the docker image..."
docker rmi shelly-training

echo "Container is removed..."
