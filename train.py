"""
ByT5 Training Script for Ibani-English Translation
This script fine-tunes a ByT5 model on the Ibani-English dataset.
"""

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


@dataclass
class TrainingConfig:
    """Configuration for training"""
    model_name: str = "google/byt5-small"  # Can use byt5-base or byt5-large for better results
    data_path: str = "ibani_eng_training_data.json"
    output_dir: str = "models/ibani-byt5-finetuned"
    
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


def load_data(data_path: str):
    """Load the Ibani-English training data"""
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract English and Ibani pairs
    english_texts = []
    ibani_texts = []
    
    for item in data:
        translation = item.get('translation', {})
        en_text = translation.get('en', '').strip()
        ibani_text = translation.get('ibani', '').strip()
        
        if en_text and ibani_text:
            english_texts.append(en_text)
            ibani_texts.append(ibani_text)
    
    print(f"Loaded {len(english_texts)} translation pairs")
    return english_texts, ibani_texts


def create_dataset(english_texts, ibani_texts, eval_split=0.1):
    """Create train and validation datasets"""
    # Create dataset
    dataset_dict = {
        'english': english_texts,
        'ibani': ibani_texts
    }
    dataset = Dataset.from_dict(dataset_dict)
    
    # Split into train and validation
    split_dataset = dataset.train_test_split(test_size=eval_split, seed=42)
    
    print(f"Train size: {len(split_dataset['train'])}")
    print(f"Validation size: {len(split_dataset['test'])}")
    
    return split_dataset['train'], split_dataset['test']


def preprocess_function(examples, tokenizer, config, direction='en_to_ibani'):
    """Preprocess the data for training"""
    if direction == 'en_to_ibani':
        # English to Ibani
        inputs = [f"translate English to Ibani: {text}" for text in examples['english']]
        targets = examples['ibani']
    else:
        # Ibani to English
        inputs = [f"translate Ibani to English: {text}" for text in examples['ibani']]
        targets = examples['english']
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=config.max_source_length,
        truncation=True,
        padding=False,  # We'll pad in the data collator
    )
    
    # Tokenize targets
    labels = tokenizer(
        targets,
        max_length=config.max_target_length,
        truncation=True,
        padding=False,
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_preds, tokenizer):
    """Compute BLEU score for evaluation"""
    bleu = evaluate.load("sacrebleu")
    
    preds, labels = eval_preds
    
    # Decode predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Replace -100 in labels (used for padding)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Clean up predictions and labels
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    
    # Compute BLEU
    result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {
        "bleu": result["score"],
    }


def train_model(config: TrainingConfig):
    """Main training function"""
    print("="*50)
    print("ByT5 Ibani Translation Model Training")
    print("="*50)
    
    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Load data
    english_texts, ibani_texts = load_data(config.data_path)
    
    # Create datasets
    train_dataset, eval_dataset = create_dataset(
        english_texts, 
        ibani_texts, 
        eval_split=config.eval_split
    )
    
    # Load tokenizer and model
    print(f"\nLoading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    
    print(f"Model loaded with {model.num_parameters():,} parameters")
    
    # Preprocess datasets
    print("\nPreprocessing datasets...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, config, direction='en_to_ibani'),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Preprocessing train dataset"
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, config, direction='en_to_ibani'),
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Preprocessing eval dataset"
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Training arguments
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
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )
    
    # Train
    print("\nStarting training...")
    print(f"Training for {config.num_train_epochs} epochs")
    print(f"Total training steps: {len(train_dataset) // config.per_device_train_batch_size * config.num_train_epochs}")
    
    trainer.train()
    
    # Save final model
    print(f"\nSaving final model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # Final evaluation
    print("\nRunning final evaluation...")
    metrics = trainer.evaluate()
    print(f"Final BLEU score: {metrics['eval_bleu']:.2f}")
    
    # Save metrics
    with open(os.path.join(config.output_dir, "final_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)
    
    return trainer, model, tokenizer


if __name__ == "__main__":
    # Initialize config
    config = TrainingConfig()
    
    # Check if data file exists
    if not os.path.exists(config.data_path):
        raise FileNotFoundError(f"Data file not found: {config.data_path}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Train model
    trainer, model, tokenizer = train_model(config)
    
    print("\nModel is ready for inference!")
    print(f"Model saved at: {config.output_dir}")
