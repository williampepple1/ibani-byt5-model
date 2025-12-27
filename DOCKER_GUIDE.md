# 🐳 Complete Docker Guide for Ibani Translation

## Why Use Docker?

You're using Docker because:
- ✅ Your computer has Python 3.10, but the app needs Python 3.11
- ✅ Docker provides Python 3.11 in an isolated environment
- ✅ Consistent environment across different machines
- ✅ Easy deployment

---

## 📋 Prerequisites

1. **Docker Desktop installed and running**
   - Download from: https://www.docker.com/products/docker-desktop
   - Make sure it's running (check system tray)

2. **Your training data**
   - `ibani_eng_training_data.json` in the project folder ✅

---

## 🚀 Complete Workflow (2 Steps)

### **Step 1: Train the Model in Docker**

This trains the model using Python 3.11 inside a Docker container:

**Windows:**
```bash
docker-train.bat
```

**Linux/Mac:**
```bash
chmod +x docker-train.sh
./docker-train.sh
```

**Or manually:**
```bash
# Build the image
docker build -t ibani-byt5-trainer .

# Run training
docker run --rm \
  --name ibani-training \
  -v "%cd%\models:/app/models" \
  -v "%cd%\logs:/app/logs" \
  -v "%cd%\ibani_eng_training_data.json:/app/ibani_eng_training_data.json" \
  ibani-byt5-trainer \
  python train.py
```

**What happens:**
1. Builds Docker image with Python 3.11
2. Installs all dependencies
3. Runs training script
4. Saves model to `models/` folder on your computer
5. Training takes 30-60 min (GPU) or 4-8 hours (CPU)

**Monitor progress:**
```bash
# In another terminal, watch the logs
docker logs -f ibani-training
```

---

### **Step 2: Start the API Server**

Once training is complete, start the API server:

```bash
docker-compose up -d
```

**What happens:**
1. Uses the trained model from `models/` folder
2. Starts FastAPI server on port 8000
3. Runs in background (`-d` = detached mode)

**Access the API:**
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **API Root**: http://localhost:8000

---

## 📝 Common Docker Commands

### Training Commands

```bash
# Build training image
docker build -t ibani-byt5-trainer .

# Start training
docker-train.bat  # Windows
./docker-train.sh  # Linux/Mac

# Check training progress
docker logs -f ibani-training

# Stop training (if needed)
docker stop ibani-training
```

### API Server Commands

```bash
# Start server
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps

# Stop server
docker-compose down

# Restart server
docker-compose restart

# Rebuild and restart
docker-compose up -d --build
```

### Cleanup Commands

```bash
# Remove all containers
docker-compose down

# Remove images
docker rmi ibani-byt5-trainer
docker rmi ibani-byt5-model-ibani-translator

# Remove volumes (careful - deletes data!)
docker-compose down -v

# Clean up everything
docker system prune -a
```

---

## 🔍 Troubleshooting

### Issue: "Docker is not running"

**Solution:**
1. Open Docker Desktop
2. Wait for it to start (whale icon in system tray)
3. Try command again

### Issue: "Port 8000 already in use"

**Solution:**
```bash
# Find what's using port 8000
netstat -ano | findstr :8000  # Windows
lsof -i :8000  # Linux/Mac

# Stop the process or change port in docker-compose.yml:
ports:
  - "8080:8000"  # Use port 8080 instead
```

### Issue: "Out of memory during training"

**Solution:**
Edit `train.py` and reduce batch size:
```python
per_device_train_batch_size: int = 4  # or 2
```

Then rebuild and retrain:
```bash
docker build -t ibani-byt5-trainer .
docker-train.bat
```

### Issue: "Cannot find model"

**Solution:**
Make sure training completed successfully:
```bash
# Check if model folder exists
dir models\ibani-byt5-finetuned  # Windows
ls -la models/ibani-byt5-finetuned  # Linux/Mac

# If not, run training again
docker-train.bat
```

### Issue: "Training is very slow"

**Solutions:**
1. **Use GPU** (if you have NVIDIA GPU):
   - Install NVIDIA Docker runtime
   - Training will be 20-30x faster

2. **Reduce dataset for testing**:
   Edit `train.py` after line 115:
   ```python
   # Add this to use only 1000 samples for testing
   english_texts = english_texts[:1000]
   ibani_texts = ibani_texts[:1000]
   ```

3. **Use fewer epochs**:
   Edit `train.py`:
   ```python
   num_train_epochs: int = 3  # Instead of 10
   ```

---

## 📊 Monitoring

### Check Training Progress

```bash
# View live logs
docker logs -f ibani-training

# Check if container is running
docker ps

# Check container resource usage
docker stats ibani-training
```

### Check API Server

```bash
# Health check
curl http://localhost:8000/health

# View logs
docker-compose logs -f

# Check container status
docker-compose ps
```

---

## 🧪 Testing the API

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Single translation
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Hello\", \"source_lang\": \"en\", \"target_lang\": \"ibani\"}"

# Batch translation
curl -X POST "http://localhost:8000/batch-translate" \
  -H "Content-Type: application/json" \
  -d "{\"texts\": [\"Hello\", \"Thank you\"], \"source_lang\": \"en\", \"target_lang\": \"ibani\"}"
```

### Using Python

```python
import requests

# Single translation
response = requests.post(
    "http://localhost:8000/translate",
    json={
        "text": "Hello, how are you?",
        "source_lang": "en",
        "target_lang": "ibani"
    }
)

print(response.json())
```

### Using Browser

1. Open http://localhost:8000/docs
2. Try the interactive API documentation
3. Click "Try it out" on any endpoint
4. Enter your text and click "Execute"

---

## 🎯 Complete Workflow Summary

```bash
# 1. Train the model (one time)
docker-train.bat

# 2. Start the API server
docker-compose up -d

# 3. Test the API
curl http://localhost:8000/health

# 4. Use the API
# Visit http://localhost:8000/docs

# 5. Stop when done
docker-compose down
```

---

## 📁 File Locations

### On Your Computer (Host)
```
c:\Users\PC\Documents\GitHub\ibani-byt5-model\
├── models/                    # Trained model (created during training)
├── logs/                      # Training logs
└── ibani_eng_training_data.json  # Your data
```

### Inside Docker Container
```
/app/
├── models/                    # Mounted from host
├── logs/                      # Mounted from host
├── ibani_eng_training_data.json  # Mounted from host
├── train.py
├── app.py
└── requirements.txt
```

**Note:** The `models/` folder is shared between your computer and Docker, so the trained model persists even after containers are stopped.

---

## 🚀 Quick Reference

| Task | Command |
|------|---------|
| Train model | `docker-train.bat` |
| Start API | `docker-compose up -d` |
| View logs | `docker-compose logs -f` |
| Stop API | `docker-compose down` |
| Restart API | `docker-compose restart` |
| Health check | `curl http://localhost:8000/health` |
| API docs | http://localhost:8000/docs |

---

## 💡 Pro Tips

1. **Keep Docker Desktop running** while using the containers
2. **Training only needs to run once** - the model is saved
3. **Use `-d` flag** to run containers in background
4. **Check logs** if something doesn't work
5. **The model folder is shared** - changes persist
6. **Use `--build` flag** if you change code: `docker-compose up -d --build`

---

## 🎉 You're Ready!

Now you can use Python 3.11 through Docker even though you have Python 3.10!

**Start now:**
```bash
docker-train.bat
```

Then wait for training to complete, and start the server:
```bash
docker-compose up -d
```

**Happy Translating! 🐳🚀**
