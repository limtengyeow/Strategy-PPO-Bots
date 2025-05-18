#!/bin/bash

# Default config file
CONFIG_FILE="${1:-config.json}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
  echo "❌ Config file '$CONFIG_FILE' not found!"
  exit 1
fi

echo "✅ Using config file: $CONFIG_FILE"

# Build the Docker image
echo "🔧 Building Docker image 'ppo-trainer'..."
docker build -t ppo-trainer .

# Run the Docker container with config and model volume
docker run --rm \
  -v "$PWD/$CONFIG_FILE:/app/config.json" \
  -v "$PWD/models:/app/models" \
  -e CONFIG_PATH=/app/config.json \
  ppo-trainer
