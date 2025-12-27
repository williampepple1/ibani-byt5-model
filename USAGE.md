# Ibani-English Translation System - Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Training the Model](#training-the-model)
3. [Using the API](#using-the-api)
4. [Command-Line Translation](#command-line-translation)
5. [Docker Deployment](#docker-deployment)
6. [Troubleshooting](#troubleshooting)

---

## Installation

### Option 1: Automatic Setup (Recommended)

**Windows:**
```bash
setup.bat
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup

1. **Create virtual environment:**
```bash
python -m venv venv
```

2. **Activate virtual environment:**

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## Training the Model

### Basic Training

```bash
python train.py
```

This will:
- Load your `ibani_eng_training_data.json` file
- Fine-tune the ByT5 model (default: `google/byt5-small`)
- Save the model to `models/ibani-byt5-finetuned/`
- Generate training metrics and logs

### Training Output

You'll see:
- Progress bars for data preprocessing
- Training progress with loss metrics
- Validation BLEU scores
- Final model saved location

**Expected Training Time:**
- **CPU**: 4-8 hours (depending on hardware)
- **GPU (CUDA)**: 30-60 minutes

### Customizing Training

Edit `train.py` to modify:

```python
@dataclass
class TrainingConfig:
    # Use a larger model for better quality
    model_name: str = "google/byt5-base"  # or "google/byt5-large"
    
    # Train for more epochs
    num_train_epochs: int = 20
    
    # Adjust batch size based on GPU memory
    per_device_train_batch_size: int = 16
    
    # Learning rate
    learning_rate: float = 3e-5
```

---

## Using the API

### 1. Start the Server

```bash
python app.py
```

Or with uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Access API Documentation

Open in browser:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3. API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Translation (English → Ibani)
```bash
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "source_lang": "en",
    "target_lang": "ibani",
    "max_length": 256,
    "num_beams": 4
  }'
```

#### Single Translation (Ibani → English)
```bash
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Mịị anịị diri bie anị fịnị ḅara",
    "source_lang": "ibani",
    "target_lang": "en",
    "max_length": 256,
    "num_beams": 4
  }'
```

#### Batch Translation
```bash
curl -X POST "http://localhost:8000/batch-translate" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello", "Good morning", "Thank you"],
    "source_lang": "en",
    "target_lang": "ibani",
    "max_length": 256,
    "num_beams": 4
  }'
```

### 4. Python Client Example

```python
import requests

API_URL = "http://localhost:8000"

# Single translation
response = requests.post(
    f"{API_URL}/translate",
    json={
        "text": "Hello, how are you?",
        "source_lang": "en",
        "target_lang": "ibani"
    }
)

result = response.json()
print(f"Translation: {result['translated_text']}")
print(f"Processing time: {result['processing_time']:.2f}s")
```

### 5. Test the API

```bash
python test_api.py
```

---

## Command-Line Translation

### Interactive Mode (Recommended)

```bash
python translate.py --interactive
```

This starts an interactive session where you can:
- Type text to translate
- Type `switch` to change direction (en→ibani or ibani→en)
- Type `quit` to exit

### Single Translation

```bash
python translate.py --text "Hello, how are you?" --source en --target ibani
```

### Batch Test

```bash
python translate.py --test
```

### Custom Model Path

```bash
python translate.py --model-path /path/to/your/model --interactive
```

---

## Docker Deployment

### Option 1: Docker Compose (Recommended)

1. **Build and start:**
```bash
docker-compose up -d
```

2. **View logs:**
```bash
docker-compose logs -f
```

3. **Stop:**
```bash
docker-compose down
```

### Option 2: Docker CLI

1. **Build image:**
```bash
docker build -t ibani-translator .
```

2. **Run container:**
```bash
docker run -d \
  --name ibani-translator \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  ibani-translator
```

3. **View logs:**
```bash
docker logs -f ibani-translator
```

4. **Stop container:**
```bash
docker stop ibani-translator
docker rm ibani-translator
```

### GPU Support (NVIDIA)

Ensure you have:
- NVIDIA Docker runtime installed
- CUDA-capable GPU

Then use `docker-compose.yml` which includes GPU configuration.

---

## Troubleshooting

### Model Not Found Error

**Problem:** `FileNotFoundError: Model not found at models/ibani-byt5-finetuned`

**Solution:**
```bash
python train.py
```
You need to train the model first before running the API or inference.

### Out of Memory (OOM) Error

**Problem:** CUDA out of memory during training

**Solutions:**
1. Reduce batch size in `train.py`:
```python
per_device_train_batch_size: int = 4  # or 2
```

2. Use a smaller model:
```python
model_name: str = "google/byt5-small"
```

3. Enable gradient checkpointing (add to `train.py`):
```python
model.gradient_checkpointing_enable()
```

### Slow Training on CPU

**Problem:** Training is very slow

**Solutions:**
1. Use a GPU if available
2. Reduce dataset size for testing:
```python
# In train.py, after loading data
english_texts = english_texts[:1000]
ibani_texts = ibani_texts[:1000]
```

3. Use fewer epochs:
```python
num_train_epochs: int = 3
```

### API Server Won't Start

**Problem:** Port 8000 already in use

**Solution:**
```bash
# Use a different port
uvicorn app:app --host 0.0.0.0 --port 8080
```

### Poor Translation Quality

**Solutions:**
1. Train for more epochs
2. Use a larger model (`byt5-base` or `byt5-large`)
3. Increase `num_beams` for better quality (slower):
```json
{
  "num_beams": 8
}
```

### Docker Build Fails

**Problem:** Docker build fails on Windows

**Solution:**
1. Ensure Docker Desktop is running
2. Check line endings (should be LF, not CRLF):
```bash
git config core.autocrlf false
```

---

## Performance Tips

### For Training:
- Use GPU if available (20-30x faster)
- Start with `byt5-small` for quick iterations
- Use `byt5-base` or `byt5-large` for production

### For Inference:
- Batch multiple translations together
- Use `num_beams=4` for good quality/speed balance
- Use `num_beams=1` for fastest inference
- Use `num_beams=8` for best quality (slower)

### For API:
- Use Docker for production deployment
- Enable GPU support for faster inference
- Use batch endpoint for multiple translations

---

## Next Steps

1. **Train the model** with your data
2. **Test locally** using `translate.py`
3. **Deploy the API** using Docker
4. **Integrate** into your application

For more information, see the [README.md](README.md) file.
