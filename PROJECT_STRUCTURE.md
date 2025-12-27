# Project Structure

```
ibani-byt5-model/
│
├── 📊 Data
│   └── ibani_eng_training_data.json    # Your training dataset (47,804 pairs)
│
├── 🧠 Training
│   └── train.py                        # ByT5 model training script
│
├── 🌐 API Server
│   ├── app.py                          # FastAPI application
│   └── test_api.py                     # API testing script
│
├── 💻 CLI Tools
│   └── translate.py                    # Command-line translation tool
│
├── 🐳 Docker
│   ├── Dockerfile                      # Container configuration
│   ├── docker-compose.yml              # Orchestration config
│   └── .dockerignore                   # Docker ignore rules
│
├── ⚙️ Configuration
│   ├── requirements.txt                # Python dependencies
│   ├── .env.example                    # Environment variables template
│   └── .gitignore                      # Git ignore rules
│
├── 🚀 Setup Scripts
│   ├── setup.bat                       # Windows setup script
│   └── setup.sh                        # Linux/Mac setup script
│
├── 📖 Documentation
│   ├── README.md                       # Main documentation
│   ├── USAGE.md                        # Detailed usage guide
│   ├── QUICKSTART.md                   # Quick reference
│   └── PROJECT_STRUCTURE.md            # This file
│
└── 📁 Generated (after training)
    ├── models/
    │   └── ibani-byt5-finetuned/       # Trained model files
    │       ├── config.json
    │       ├── pytorch_model.bin
    │       ├── tokenizer_config.json
    │       └── final_metrics.json
    │
    ├── logs/                            # Training logs
    │   └── tensorboard/
    │
    └── venv/                            # Python virtual environment
```

## Component Overview

### 1. Training Pipeline
```
ibani_eng_training_data.json
         ↓
    train.py (ByT5 fine-tuning)
         ↓
models/ibani-byt5-finetuned/
```

### 2. Inference Options

#### Option A: CLI
```
User Input → translate.py → ByT5 Model → Translation Output
```

#### Option B: API
```
HTTP Request → FastAPI (app.py) → ByT5 Model → JSON Response
```

#### Option C: Docker
```
HTTP Request → Docker Container → FastAPI → ByT5 Model → JSON Response
```

## Data Flow

### Training
1. Load `ibani_eng_training_data.json`
2. Split into train/validation (90/10)
3. Preprocess with ByT5 tokenizer
4. Fine-tune model
5. Evaluate with BLEU score
6. Save to `models/ibani-byt5-finetuned/`

### Inference (API)
1. Client sends POST request to `/translate`
2. FastAPI validates request
3. Model generates translation
4. Return JSON response with translation

### Inference (CLI)
1. User enters text
2. Script loads model
3. Generate translation
4. Display result

## Key Files Explained

### `train.py`
- Loads training data
- Configures ByT5 model
- Handles preprocessing
- Runs training loop
- Saves trained model

**Key Classes:**
- `TrainingConfig`: Training hyperparameters
- `train_model()`: Main training function
- `preprocess_function()`: Data preprocessing
- `compute_metrics()`: BLEU score calculation

### `app.py`
- FastAPI application
- Model loading and caching
- Translation endpoints
- Request/response validation

**Key Endpoints:**
- `GET /health`: Health check
- `POST /translate`: Single translation
- `POST /batch-translate`: Batch translation

### `translate.py`
- Standalone inference script
- Interactive mode
- Batch testing
- Command-line interface

**Key Functions:**
- `load_model()`: Load trained model
- `translate()`: Generate translation
- `interactive_mode()`: Interactive CLI

## Technology Stack

### Core ML
- **ByT5**: Byte-level T5 model
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face library

### API & Server
- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

### Deployment
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration

### Evaluation
- **SacreBLEU**: Translation quality metric
- **TensorBoard**: Training visualization

## Model Architecture

```
Input Text (English or Ibani)
         ↓
    ByT5 Tokenizer (Byte-level)
         ↓
    ByT5 Encoder (Transformer)
         ↓
    ByT5 Decoder (Transformer)
         ↓
    Output Text (Ibani or English)
```

## Why ByT5?

1. **Byte-level tokenization**: No vocabulary limitations
2. **Preserves special characters**: Handles á, ḅ, etc. perfectly
3. **Language-agnostic**: No language-specific preprocessing
4. **Low-resource friendly**: Works well with limited data
5. **No OOV tokens**: Every byte sequence is valid

## Deployment Options

### Development
```bash
python app.py
# Local server at http://localhost:8000
```

### Production (Docker)
```bash
docker-compose up -d
# Containerized server with auto-restart
```

### Cloud Deployment
- AWS ECS/Fargate
- Google Cloud Run
- Azure Container Instances
- Kubernetes

## Performance Characteristics

### Training
- **Dataset**: 47,804 translation pairs
- **Model**: ByT5-small (300M parameters)
- **Time**: 30-60 min (GPU) / 4-8 hours (CPU)
- **Memory**: ~4GB GPU / 8GB RAM

### Inference
- **Latency**: 100-500ms per translation (GPU)
- **Throughput**: 10-50 translations/second (GPU)
- **Memory**: ~2GB GPU / 4GB RAM

## Next Steps

1. **Train**: `python train.py`
2. **Test**: `python translate.py --test`
3. **Deploy**: `docker-compose up -d`
4. **Monitor**: Check logs and metrics
5. **Iterate**: Improve based on results

---

For more details, see:
- [README.md](README.md) - Overview and features
- [USAGE.md](USAGE.md) - Detailed usage instructions
- [QUICKSTART.md](QUICKSTART.md) - Quick reference
