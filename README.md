# Ibani-English Translation with ByT5

A production-ready translation system for Ibani and English languages using Google's ByT5 (Byte-level T5) model, served via FastAPI and containerized with Docker.

## 🌟 Features

- **ByT5 Model**: Byte-level tokenization preserves Ibani's unique characters (á, ḅ, etc.)
- **Bidirectional Translation**: English ↔ Ibani
- **FastAPI Backend**: High-performance REST API
- **Docker Support**: Easy deployment with containerization
- **Batch Processing**: Translate multiple texts efficiently
- **GPU Support**: Optimized for CUDA acceleration

## 📋 Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- CUDA-capable GPU (optional, for faster training/inference)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

This will:
- Load the `ibani_eng_training_data.json` dataset
- Fine-tune the ByT5 model
- Save the trained model to `models/ibani-byt5-finetuned/`
- Generate training metrics and logs

**Training Configuration:**
- Model: `google/byt5-small` (can be changed to `byt5-base` or `byt5-large`)
- Epochs: 10
- Batch Size: 8
- Learning Rate: 5e-5
- Evaluation: BLEU score

### 3. Run the API Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Or simply:

```bash
python app.py
```

The API will be available at `http://localhost:8000`

### 4. Access the API Documentation

Open your browser and navigate to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build the image
docker build -t ibani-translator .

# Run the container
docker run -p 8000:8000 -v $(pwd)/models:/app/models ibani-translator
```

### Using Docker Compose

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

### Single Translation
```bash
POST /translate
Content-Type: application/json

{
  "text": "Hello, how are you?",
  "source_lang": "en",
  "target_lang": "ibani",
  "max_length": 256,
  "num_beams": 4
}
```

### Batch Translation
```bash
POST /batch-translate
Content-Type: application/json

{
  "texts": [
    "Hello",
    "Good morning",
    "Thank you"
  ],
  "source_lang": "en",
  "target_lang": "ibani",
  "max_length": 256,
  "num_beams": 4
}
```

## 🔧 Configuration

### Training Configuration

Edit `train.py` to modify training parameters:

```python
@dataclass
class TrainingConfig:
    model_name: str = "google/byt5-small"  # or byt5-base, byt5-large
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    learning_rate: float = 5e-5
    max_source_length: int = 256
    max_target_length: int = 256
```

### Environment Variables

Create a `.env` file:

```bash
MODEL_PATH=models/ibani-byt5-finetuned
MAX_LENGTH=256
BATCH_SIZE=8
DEVICE=cuda
```

## 📊 Model Performance

The model is evaluated using BLEU score during training. Check the training logs and `models/ibani-byt5-finetuned/final_metrics.json` for detailed metrics.

## 🎯 Why ByT5?

ByT5 is ideal for Ibani translation because:

1. **Byte-level Tokenization**: No vocabulary limitations, handles any Unicode character
2. **Preserves Diacritics**: Tonal marks (á, ḅ, etc.) are naturally preserved
3. **Low-Resource Friendly**: Works well with limited training data
4. **No Custom Tokenizer**: No need to train language-specific tokenizers

## 📁 Project Structure

```
ibani-byt5-model/
├── app.py                          # FastAPI application
├── train.py                        # Training script
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── docker-compose.yml              # Docker Compose configuration
├── .env.example                    # Environment variables template
├── ibani_eng_training_data.json    # Training data
├── models/                         # Trained models (generated)
│   └── ibani-byt5-finetuned/
├── logs/                           # Training logs (generated)
└── README.md                       # This file
```

## 🔬 Advanced Usage

### Custom Generation Parameters

```python
# In your API request
{
  "text": "Your text here",
  "source_lang": "en",
  "target_lang": "ibani",
  "max_length": 512,        # Longer outputs
  "num_beams": 8,           # Better quality (slower)
  "temperature": 0.8        # More creative (if do_sample=True)
}
```

### Training on Different Model Sizes

```python
# In train.py, change model_name:
model_name: str = "google/byt5-base"   # Better quality, slower
# or
model_name: str = "google/byt5-large"  # Best quality, requires more GPU memory
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Google Research for the ByT5 model
- Hugging Face for the Transformers library
- The Ibani language community

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for the Ibani language community**
