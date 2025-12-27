# Training on Google Colab & Kaggle - Complete Guide

## 🎯 Why Train on Colab/Kaggle?

- ✅ **Free GPU access** (Tesla T4, P100, or better)
- ✅ **Python 3.11** already installed
- ✅ **20-30x faster** than CPU training
- ✅ **No local setup** required
- ✅ **Free compute hours** every week

---

## 🚀 Quick Start

### **Option 1: Google Colab (Recommended)**

1. **Upload your data** to Google Drive
2. **Open the Colab notebook** (see below)
3. **Run all cells**
4. **Download trained model**

### **Option 2: Kaggle**

1. **Upload your data** as a Kaggle dataset
2. **Create a new notebook**
3. **Copy the training code** (see below)
4. **Run and download model**

---

## 📓 Google Colab Notebook

### Step-by-Step Instructions

#### 1. Upload Data to Google Drive

1. Go to [Google Drive](https://drive.google.com)
2. Create folder: `ibani-translation`
3. Upload `ibani_eng_training_data.json` to this folder

#### 2. Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. Create new notebook: `File` → `New notebook`
3. Enable GPU: `Runtime` → `Change runtime type` → `GPU` → `Save`

#### 3. Copy This Code to Colab

```python
# ============================================
# Ibani-English ByT5 Training on Google Colab
# ============================================

# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Install dependencies
!pip install -q transformers datasets accelerate evaluate sacrebleu tensorboard sentencepiece

# 3. Import libraries
import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import evaluate
import numpy as np
from tqdm import tqdm

print("✅ All libraries imported successfully!")
print(f"🐍 Python version: {torch.__version__}")
print(f"🔥 PyTorch version: {torch.__version__}")
print(f"💻 GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

# 4. Configuration
@dataclass
class TrainingConfig:
    """Configuration for training"""
    model_name: str = "google/byt5-small"  # Can use byt5-base or byt5-large
    data_path: str = "/content/drive/MyDrive/ibani-translation/ibani_eng_training_data.json"
    output_dir: str = "/content/ibani-byt5-finetuned"
    
    # Training hyperparameters
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    
    # Generation parameters
    max_source_length: int = 256
    max_target_length: int = 256
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100
    eval_split: float = 0.1
    
    # Other
    seed: int = 42
    fp16: bool = torch.cuda.is_available()

config = TrainingConfig()

# 5. Load data
print(f"\n📊 Loading data from {config.data_path}...")
with open(config.data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

english_texts = []
ibani_texts = []

for item in data:
    translation = item.get('translation', {})
    en_text = translation.get('en', '').strip()
    ibani_text = translation.get('ibani', '').strip()
    
    if en_text and ibani_text:
        english_texts.append(en_text)
        ibani_texts.append(ibani_text)

print(f"✅ Loaded {len(english_texts)} translation pairs")

# 6. Create datasets
dataset_dict = {
    'english': english_texts,
    'ibani': ibani_texts
}
dataset = Dataset.from_dict(dataset_dict)
split_dataset = dataset.train_test_split(test_size=config.eval_split, seed=42)

train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

print(f"📚 Train size: {len(train_dataset)}")
print(f"📚 Validation size: {len(eval_dataset)}")

# 7. Load model and tokenizer
print(f"\n🤖 Loading model: {config.model_name}")
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)

print(f"✅ Model loaded with {model.num_parameters():,} parameters")

# 8. Preprocessing function
def preprocess_function(examples):
    inputs = [f"translate English to Ibani: {text}" for text in examples['english']]
    targets = examples['ibani']
    
    model_inputs = tokenizer(
        inputs,
        max_length=config.max_source_length,
        truncation=True,
        padding=False,
    )
    
    labels = tokenizer(
        targets,
        max_length=config.max_target_length,
        truncation=True,
        padding=False,
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 9. Preprocess datasets
print("\n🔄 Preprocessing datasets...")
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Preprocessing train dataset"
)

eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_dataset.column_names,
    desc="Preprocessing eval dataset"
)

# 10. Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)

# 11. Metrics
bleu = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    
    result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {"bleu": result["score"]}

# 12. Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=config.output_dir,
    evaluation_strategy="steps",
    eval_steps=config.eval_steps,
    save_steps=config.save_steps,
    logging_steps=config.logging_steps,
    learning_rate=config.learning_rate,
    per_device_train_batch_size=config.per_device_train_batch_size,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    weight_decay=config.weight_decay,
    save_total_limit=3,
    num_train_epochs=config.num_train_epochs,
    predict_with_generate=True,
    fp16=config.fp16,
    warmup_steps=config.warmup_steps,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    push_to_hub=False,
    report_to=["tensorboard"],
    seed=config.seed,
)

# 13. Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 14. Train!
print("\n🚀 Starting training...")
print(f"⏱️  Training for {config.num_train_epochs} epochs")
print(f"📊 Total steps: {len(train_dataset) // config.per_device_train_batch_size * config.num_train_epochs}")

trainer.train()

# 15. Save model
print(f"\n💾 Saving model to {config.output_dir}")
trainer.save_model(config.output_dir)
tokenizer.save_pretrained(config.output_dir)

# 16. Final evaluation
print("\n📊 Running final evaluation...")
metrics = trainer.evaluate()
print(f"✅ Final BLEU score: {metrics['eval_bleu']:.2f}")

# Save metrics
with open(os.path.join(config.output_dir, "final_metrics.json"), 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n🎉 Training completed successfully!")
print(f"📁 Model saved to: {config.output_dir}")

# 17. Test the model
print("\n🧪 Testing the model...")

def translate(text, source_lang="en", target_lang="ibani"):
    if source_lang == "en" and target_lang == "ibani":
        prompt = f"translate English to Ibani: {text}"
    else:
        prompt = f"translate Ibani to English: {text}"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256, num_beams=4, early_stopping=True)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# Test translations
test_sentences = [
    "Hello, how are you?",
    "Good morning",
    "Thank you very much"
]

print("\n📝 Test translations:")
for sentence in test_sentences:
    translation = translate(sentence)
    print(f"  EN: {sentence}")
    print(f"  IB: {translation}\n")

# 18. Download model to your computer
print("\n📥 To download the model:")
print("1. Click the folder icon on the left sidebar")
print("2. Navigate to /content/ibani-byt5-finetuned/")
print("3. Right-click the folder → Download")
print("\nOr run this to save to Google Drive:")

# Save to Google Drive
!cp -r /content/ibani-byt5-finetuned /content/drive/MyDrive/ibani-translation/

print("✅ Model saved to Google Drive!")
print("\n🎉 All done! You can now download the model and use it locally.")
```

#### 4. Run the Notebook

1. Click `Runtime` → `Run all`
2. Authorize Google Drive access when prompted
3. Wait for training to complete (30-60 minutes)
4. Download the model from Google Drive

---

## 📓 Kaggle Notebook

### Step-by-Step Instructions

#### 1. Upload Data to Kaggle

1. Go to [Kaggle Datasets](https://www.kaggle.com/datasets)
2. Click `New Dataset`
3. Upload `ibani_eng_training_data.json`
4. Name it: `ibani-english-translation-data`
5. Click `Create`

#### 2. Create Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click `New Notebook`
3. Settings → Accelerator → `GPU T4 x2`
4. Add your dataset: `Add Data` → Search for your dataset

#### 3. Use This Code

```python
# ============================================
# Ibani-English ByT5 Training on Kaggle
# ============================================

# 1. Install dependencies
!pip install -q transformers datasets accelerate evaluate sacrebleu tensorboard sentencepiece

# 2. Import libraries
import json
import os
from dataclasses import dataclass
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import evaluate
import numpy as np

print("✅ Setup complete!")
print(f"🐍 Python version: {torch.__version__}")
print(f"💻 GPU: {torch.cuda.get_device_name(0)}")

# 3. Configuration
@dataclass
class TrainingConfig:
    model_name: str = "google/byt5-small"
    data_path: str = "/kaggle/input/ibani-english-translation-data/ibani_eng_training_data.json"
    output_dir: str = "/kaggle/working/ibani-byt5-finetuned"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    learning_rate: float = 5e-5
    max_source_length: int = 256
    max_target_length: int = 256
    eval_split: float = 0.1
    fp16: bool = True

config = TrainingConfig()

# [Rest of the code is the same as Colab version above]
# ... (copy the same training code from Colab)

# At the end, the model will be in /kaggle/working/ibani-byt5-finetuned/
# Click "Save Version" to save your work
# Download from the Output tab
```

---

## 📥 After Training: Download & Use Locally

### 1. Download the Model

**From Google Colab:**
- Model is saved to Google Drive: `ibani-translation/ibani-byt5-finetuned/`
- Download the entire folder to your computer

**From Kaggle:**
- Click `Save Version` → `Save & Run All`
- After completion, go to `Output` tab
- Download `ibani-byt5-finetuned` folder

### 2. Place Model in Your Project

```
c:\Users\PC\Documents\GitHub\ibani-byt5-model\
└── models\
    └── ibani-byt5-finetuned\    ← Put downloaded folder here
        ├── config.json
        ├── pytorch_model.bin
        ├── tokenizer_config.json
        └── ...
```

### 3. Start the API Server Locally

```bash
# With Docker
docker-compose up -d

# Or with Python 3.10 (should work fine for inference)
pip install -r requirements.txt
python app.py
```

---

## 🎯 Comparison: Colab vs Kaggle

| Feature | Google Colab | Kaggle |
|---------|-------------|---------|
| **GPU** | Tesla T4, P100 | Tesla T4 x2, P100 |
| **Free Hours** | ~12 hours/day | 30 hours/week |
| **RAM** | 12-25 GB | 13-30 GB |
| **Storage** | 15 GB (Drive) | 20 GB |
| **Setup** | Mount Drive | Upload dataset |
| **Best For** | Quick experiments | Longer training |

**Recommendation:** Start with **Google Colab** for simplicity!

---

## ⚡ Performance Tips

### For Faster Training:

1. **Use GPU** (already enabled in Colab/Kaggle)
2. **Increase batch size** if you have enough memory:
   ```python
   per_device_train_batch_size: int = 16  # or 32
   ```
3. **Use larger model** for better quality:
   ```python
   model_name: str = "google/byt5-base"  # or byt5-large
   ```

### For Better Quality:

1. **Train longer**:
   ```python
   num_train_epochs: int = 20
   ```
2. **Use more beams** during generation:
   ```python
   num_beams=8  # instead of 4
   ```

---

## 🐛 Troubleshooting

### Issue: "Out of memory"

**Solution:**
```python
per_device_train_batch_size: int = 4  # Reduce batch size
```

### Issue: "Runtime disconnected"

**Solutions:**
- Colab: Keep the tab active, interact occasionally
- Kaggle: Enable "Always-on" in notebook settings

### Issue: "Can't find data file"

**Solutions:**
- Colab: Check Google Drive path
- Kaggle: Make sure dataset is added to notebook

---

## ✅ Checklist

- [ ] Data uploaded to Colab/Kaggle
- [ ] GPU enabled
- [ ] All cells run successfully
- [ ] Training completed (BLEU score shown)
- [ ] Model downloaded to local computer
- [ ] Model placed in `models/` folder
- [ ] API server tested locally

---

## 🎉 You're Done!

Now you have:
1. ✅ Trained model using Python 3.11 on free GPU
2. ✅ Model downloaded to your computer
3. ✅ Ready to run API server locally

**Next step:** Start your API server!
```bash
docker-compose up -d
```

---

**Happy Training! 🚀📓**
