#!/bin/bash
# Train the model using Docker with Python 3.11

echo "================================================"
echo "Training Ibani ByT5 Model in Docker"
echo "================================================"
echo ""

echo "Building Docker image..."
docker build -t ibani-byt5-trainer .

echo ""
echo "Starting training container..."
echo "This will take 30-60 minutes on GPU or 4-8 hours on CPU"
echo ""

docker run --rm \
  --name ibani-training \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/logs:/app/logs" \
  -v "$(pwd)/ibani_eng_training_data.json:/app/ibani_eng_training_data.json" \
  ibani-byt5-trainer \
  python train.py

echo ""
echo "================================================"
echo "Training complete! Model saved to models/"
echo "================================================"
echo ""
echo "Next step: Start the API server with:"
echo "  docker-compose up -d"
echo ""
