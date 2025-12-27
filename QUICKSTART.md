# 🚀 Quick Start Guide - Ibani Translation System

## ⚡ 3-Step Quick Start

### Step 1: Install Dependencies
```bash
# Windows
setup.bat

# Linux/Mac
chmod +x setup.sh && ./setup.sh
```

### Step 2: Train the Model
```bash
python train.py
```
⏱️ **Time**: 30-60 min (GPU) or 4-8 hours (CPU)

### Step 3: Start Translating!

**Option A - Interactive CLI:**
```bash
python translate.py --interactive
```

**Option B - API Server:**
```bash
python app.py
# Then visit: http://localhost:8000/docs
```

**Option C - Docker:**
```bash
docker-compose up -d
```

---

## 📝 Common Commands

### Training
```bash
# Basic training
python train.py

# Check training progress
tensorboard --logdir models/ibani-byt5-finetuned/runs
```

### Translation (CLI)
```bash
# Interactive mode
python translate.py --interactive

# Single translation
python translate.py --text "Hello" --source en --target ibani

# Test mode
python translate.py --test
```

### API Server
```bash
# Start server
python app.py

# Test API
python test_api.py

# Health check
curl http://localhost:8000/health
```

### Docker
```bash
# Start
docker-compose up -d

# Logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 🔧 Configuration Quick Reference

### Model Sizes
- `google/byt5-small` - Fast, good for testing (300M params)
- `google/byt5-base` - Balanced (580M params)
- `google/byt5-large` - Best quality (1.2B params)

### Generation Parameters
- `num_beams=1` - Fastest
- `num_beams=4` - Balanced (default)
- `num_beams=8` - Best quality

### Batch Sizes (adjust for your GPU)
- 16GB GPU: batch_size=8-16
- 8GB GPU: batch_size=4-8
- CPU: batch_size=2-4

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Model not found | Run `python train.py` first |
| Out of memory | Reduce batch_size in train.py |
| Port 8000 in use | Use `--port 8080` |
| Slow training | Use GPU or reduce dataset size |
| Poor quality | Train longer or use larger model |

---

## 📚 File Reference

| File | Purpose |
|------|---------|
| `train.py` | Train the ByT5 model |
| `app.py` | FastAPI server |
| `translate.py` | CLI translation tool |
| `test_api.py` | Test the API endpoints |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Docker container config |
| `docker-compose.yml` | Docker orchestration |

---

## 🌐 API Endpoints

```bash
GET  /health          # Check server status
POST /translate       # Single translation
POST /batch-translate # Multiple translations
GET  /docs            # API documentation
```

---

## 💡 Pro Tips

1. **Start small**: Use `byt5-small` for initial testing
2. **GPU matters**: Training is 20-30x faster on GPU
3. **Batch translations**: Use `/batch-translate` for multiple texts
4. **Save checkpoints**: Training saves every 500 steps
5. **Monitor training**: Use TensorBoard to track progress

---

## 📞 Need Help?

- **Full Documentation**: See [README.md](README.md)
- **Detailed Guide**: See [USAGE.md](USAGE.md)
- **API Docs**: http://localhost:8000/docs (when server is running)

---

**Built with ❤️ using ByT5, FastAPI, and Docker**
