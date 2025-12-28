"""
Test script for Kaggle - Run this in a Kaggle notebook cell after training.
Tests the model for ḅ and á character preservation before downloading.
"""

from huggingface_translator import IbaniHuggingFaceTranslator

def test_model_on_kaggle():
    """Test the trained model on Kaggle."""
    print("🧪 Testing Model on Kaggle")
    print("="*70)
    
    # Load the model from the output directory
    print("\n📂 Loading model from ./ibani_model...")
    translator = IbaniHuggingFaceTranslator(model_path="./ibani_model")
    
    # Test cases focusing on ḅ and á
    test_cases = [
        ("love", "ḅẹlẹma"),  # Should contain ḅ
        ("woman", "ọ́rụ́ḅọ́"),  # Should contain ḅ
        ("she loves you", None),  # Check for spacing issues
        ("I love you", None),
        ("good morning", None),
        ("thank you", None),
    ]
    
    print("\n📝 Translation Tests:")
    print("-"*70)
    
    all_good = True
    
    for english, expected_ibani in test_cases:
        translation = translator.translate(english)
        
        # Check for spacing issues with ḅ and á
        has_space_issue = ' ḅ ' in translation or ' á ' in translation
        has_b = 'ḅ' in translation
        has_a = 'á' in translation
        
        # Determine status
        if expected_ibani:
            matches = translation == expected_ibani
            status = "✅" if matches and not has_space_issue else "⚠️"
        else:
            status = "✅" if not has_space_issue else "⚠️"
        
        print(f"\n{status} EN: {english}")
        print(f"   IBANI: {translation}")
        
        if expected_ibani:
            print(f"   Expected: {expected_ibani}")
            if translation != expected_ibani:
                print(f"   ⚠️  Mismatch!")
                all_good = False
        
        if has_space_issue:
            print(f"   ❌ SPACING ISSUE DETECTED!")
            all_good = False
        
        if has_b:
            print(f"   ✓ Contains ḅ")
        if has_a:
            print(f"   ✓ Contains á")
    
    # Test tokenization directly
    print("\n\n🔍 Tokenization Tests:")
    print("-"*70)
    
    test_words = ['ḅẹlẹma', 'ọ́rụ́ḅọ́', 'árị', 'ḅ', 'á']
    
    for word in test_words:
        tokens = translator.tokenizer.tokenize(word)
        decoded = translator.tokenizer.decode(
            translator.tokenizer.convert_tokens_to_ids(tokens),
            skip_special_tokens=True
        )
        
        preserved = (decoded.replace(' ', '') == word.replace(' ', ''))
        status = "✅" if preserved else "❌"
        
        print(f"\n{status} Word: '{word}'")
        print(f"   Tokens: {tokens}")
        print(f"   Decoded: '{decoded}'")
        
        if not preserved:
            print(f"   ❌ Character loss detected!")
            all_good = False
    
    # Final verdict
    print("\n" + "="*70)
    if all_good:
        print("✅ ALL TESTS PASSED!")
        print("✅ Model is ready to download and deploy!")
        print("\nNo spacing issues detected with ḅ and á characters.")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("⚠️  Review the output above before downloading.")
        print("\nYou may need to retrain with adjusted parameters.")
    
    print("\n" + "="*70)
    return all_good


# Run the test
if __name__ == "__main__":
    test_model_on_kaggle()
