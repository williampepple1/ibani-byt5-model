"""
Simple inference script for testing the trained model locally
without running the full API server
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_model(model_path="models/ibani-byt5-finetuned"):
    """Load the trained model"""
    print(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please train the model first using: python train.py"
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!\n")
    return model, tokenizer, device


def translate(text, model, tokenizer, device, source_lang="en", target_lang="ibani", max_length=256, num_beams=4):
    """Translate text"""
    # Create prompt
    if source_lang == "en" and target_lang == "ibani":
        prompt = f"translate English to Ibani: {text}"
    else:
        prompt = f"translate Ibani to English: {text}"
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )
    
    # Decode
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated.strip()


def interactive_mode(model, tokenizer, device):
    """Interactive translation mode"""
    print("="*60)
    print("Interactive Translation Mode")
    print("="*60)
    print("Commands:")
    print("  - Type 'switch' to change translation direction")
    print("  - Type 'quit' or 'exit' to quit")
    print("="*60)
    print()
    
    source_lang = "en"
    target_lang = "ibani"
    
    while True:
        direction = f"{source_lang} → {target_lang}"
        text = input(f"\n[{direction}] Enter text: ").strip()
        
        if not text:
            continue
        
        if text.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if text.lower() == 'switch':
            source_lang, target_lang = target_lang, source_lang
            print(f"Switched to: {source_lang} → {target_lang}")
            continue
        
        try:
            translation = translate(text, model, tokenizer, device, source_lang, target_lang)
            print(f"Translation: {translation}")
        except Exception as e:
            print(f"Error: {str(e)}")


def batch_test(model, tokenizer, device):
    """Test with some example sentences"""
    print("="*60)
    print("Testing with example sentences")
    print("="*60)
    print()
    
    examples = [
        ("Hello, how are you?", "en", "ibani"),
        ("Good morning", "en", "ibani"),
        ("Thank you very much", "en", "ibani"),
    ]
    
    for text, src, tgt in examples:
        print(f"\nOriginal ({src}): {text}")
        translation = translate(text, model, tokenizer, device, src, tgt)
        print(f"Translation ({tgt}): {translation}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ibani Translation Inference")
    parser.add_argument("--model-path", type=str, default="models/ibani-byt5-finetuned", help="Path to trained model")
    parser.add_argument("--text", type=str, help="Text to translate")
    parser.add_argument("--source", type=str, default="en", choices=["en", "ibani"], help="Source language")
    parser.add_argument("--target", type=str, default="ibani", choices=["en", "ibani"], help="Target language")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--test", action="store_true", help="Run batch test")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, device = load_model(args.model_path)
    
    if args.interactive:
        # Interactive mode
        interactive_mode(model, tokenizer, device)
    elif args.test:
        # Batch test mode
        batch_test(model, tokenizer, device)
    elif args.text:
        # Single translation
        translation = translate(args.text, model, tokenizer, device, args.source, args.target)
        print(f"\nOriginal ({args.source}): {args.text}")
        print(f"Translation ({args.target}): {translation}")
    else:
        # Default: interactive mode
        interactive_mode(model, tokenizer, device)
