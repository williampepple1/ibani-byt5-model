"""
Test script to verify ByT5 migration is working correctly.
Tests model loading, character preservation, and basic translation.
"""

import sys
from huggingface_translator import IbaniHuggingFaceTranslator


def test_model_loading():
    """Test that ByT5 model loads correctly."""
    print("=" * 60)
    print("TEST 1: Model Loading")
    print("=" * 60)
    
    try:
        translator = IbaniHuggingFaceTranslator()
        print("✓ Model loaded successfully")
        print(f"✓ Model type: {type(translator.model).__name__}")
        print(f"✓ Tokenizer type: {type(translator.tokenizer).__name__}")
        print(f"✓ Task prefix: '{translator.task_prefix}'")
        
        # Verify it's ByT5
        assert "T5ForConditionalGeneration" in str(type(translator.model)), "Model should be T5ForConditionalGeneration"
        assert "ByT5Tokenizer" in str(type(translator.tokenizer)), "Tokenizer should be ByT5Tokenizer"
        
        print("✓ All assertions passed")
        return translator
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_character_preservation(translator):
    """Test that special Ibani characters are preserved."""
    print("\n" + "=" * 60)
    print("TEST 2: Character Preservation")
    print("=" * 60)
    
    if translator is None:
        print("✗ Skipping test - translator not loaded")
        return
    
    # Test tokenization of Ibani characters
    test_words = [
        "ḅẹlẹma",  # Contains ḅ and ẹ
        "ọ́rụ́ḅọ́",  # Contains ọ́, ụ́, ḅ
        "árị",      # Contains á, ị
        "fíị",      # Contains í, ị
        "ẹ́kị́rị́kị́"  # Multiple special characters
    ]
    
    print("\nTesting tokenization of Ibani words:")
    for word in test_words:
        # Tokenize and decode
        tokens = translator.tokenizer.tokenize(word)
        token_ids = translator.tokenizer.encode(word, add_special_tokens=False)
        decoded = translator.tokenizer.decode(token_ids, skip_special_tokens=True)
        
        print(f"\nWord: '{word}'")
        print(f"  Tokens: {tokens[:10]}...")  # Show first 10 tokens
        print(f"  Token count: {len(tokens)}")
        print(f"  Decoded: '{decoded}'")
        
        # Verify character preservation
        if word == decoded:
            print(f"  ✓ Characters preserved correctly")
        else:
            print(f"  ✗ Character mismatch!")
            print(f"    Expected: '{word}'")
            print(f"    Got: '{decoded}'")


def test_task_prefix(translator):
    """Test that task prefix is added correctly."""
    print("\n" + "=" * 60)
    print("TEST 3: Task Prefix")
    print("=" * 60)
    
    if translator is None:
        print("✗ Skipping test - translator not loaded")
        return
    
    test_text = "I eat fish"
    expected_prefix = "translate English to Ibani: "
    
    # Manually create input with prefix
    input_with_prefix = expected_prefix + test_text
    
    print(f"Original text: '{test_text}'")
    print(f"Expected prefix: '{expected_prefix}'")
    print(f"Input with prefix: '{input_with_prefix}'")
    
    # Tokenize
    tokens = translator.tokenizer.encode(input_with_prefix, add_special_tokens=False)
    decoded = translator.tokenizer.decode(tokens, skip_special_tokens=True)
    
    print(f"Decoded: '{decoded}'")
    
    if decoded == input_with_prefix:
        print("✓ Task prefix preserved correctly")
    else:
        print("✗ Task prefix not preserved")


def test_basic_translation(translator):
    """Test basic translation (will be random with untrained model)."""
    print("\n" + "=" * 60)
    print("TEST 4: Basic Translation")
    print("=" * 60)
    
    if translator is None:
        print("✗ Skipping test - translator not loaded")
        return
    
    test_sentences = [
        "I eat fish",
        "Good morning",
        "Thank you"
    ]
    
    print("\nNote: With untrained base model, translations will be random.")
    print("After training, these should produce proper Ibani translations.\n")
    
    for sentence in test_sentences:
        try:
            translation = translator.translate(sentence)
            print(f"EN: {sentence}")
            print(f"IBANI: {translation}")
            print(f"✓ Translation completed (length: {len(translation)} chars)")
            print()
        except Exception as e:
            print(f"✗ Translation failed for '{sentence}': {e}")
            import traceback
            traceback.print_exc()


def test_preprocessing():
    """Test data preprocessing with task prefix."""
    print("\n" + "=" * 60)
    print("TEST 5: Data Preprocessing")
    print("=" * 60)
    
    try:
        translator = IbaniHuggingFaceTranslator()
        
        # Create sample data
        sample_data = {
            "translation": [
                {"en": "I eat fish", "ibani": "ịrị olokpó fíị"},
                {"en": "Good morning", "ibani": "ụ́tarị ọ́ma"}
            ]
        }
        
        # Preprocess
        processed = translator.preprocess_data(sample_data)
        
        print("Sample data preprocessed successfully")
        print(f"✓ Keys in processed data: {list(processed.keys())}")
        print(f"✓ Input IDs shape: {len(processed['input_ids'])}")
        print(f"✓ Labels shape: {len(processed['labels'])}")
        
        # Decode first example to verify prefix
        first_input = translator.tokenizer.decode(processed['input_ids'][0], skip_special_tokens=True)
        print(f"\nFirst input (decoded): '{first_input}'")
        
        if first_input.startswith("translate English to Ibani:"):
            print("✓ Task prefix added correctly")
        else:
            print("✗ Task prefix missing or incorrect")
            
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ByT5 MIGRATION VERIFICATION TESTS")
    print("=" * 60)
    print("\nThis script tests the migration from MarianMT to ByT5")
    print("It verifies model loading, character preservation, and translation.\n")
    
    # Run tests
    translator = test_model_loading()
    test_character_preservation(translator)
    test_task_prefix(translator)
    test_preprocessing()
    test_basic_translation(translator)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("\n✓ If all tests passed, ByT5 migration is successful!")
    print("\nNext steps:")
    print("1. Train the model: python train_from_ibani_eng.py")
    print("2. Test with trained model for proper Ibani translations")
    print("3. Upload to HuggingFace Hub: python upload_to_hf.py")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
