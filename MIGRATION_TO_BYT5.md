# Migration from MarianMT to ByT5

## Overview

This document describes the migration from MarianMT to ByT5 for the Ibani translation model.

## Why ByT5?

### Advantages of ByT5 over MarianMT:

1. **Character-Level Tokenization**: ByT5 works at the byte/character level, which is perfect for Ibani's special characters (á, ḅ, ẹ, ị, ọ, ụ, etc.)
   - No need to add special tokens to vocabulary
   - Natural handling of all Unicode characters
   - Prevents spacing issues with tonal marks

2. **Better for Low-Resource Languages**: ByT5 is specifically designed for languages with limited training data

3. **No Tokenization Issues**: Since it works at character level, there's no risk of:
   - Characters being split incorrectly
   - Tonal marks being separated from base characters
   - Unknown tokens for special characters

4. **Multilingual by Design**: ByT5 is trained on 101 languages and handles diverse scripts naturally

## Key Changes Made

### 1. Model and Tokenizer (`huggingface_translator.py`)

**Before (MarianMT):**
```python
from transformers import MarianMTModel, MarianTokenizer

self.tokenizer = MarianTokenizer.from_pretrained(model_name)
self.model = MarianMTModel.from_pretrained(model_name)
```

**After (ByT5):**
```python
from transformers import T5ForConditionalGeneration, ByT5Tokenizer

self.tokenizer = ByT5Tokenizer.from_pretrained(model_name)
self.model = T5ForConditionalGeneration.from_pretrained(model_name)
self.task_prefix = "translate English to Ibani: "
```

### 2. Base Model

**Before:** `Helsinki-NLP/opus-mt-en-mul` (English to multilingual)
**After:** `google/byt5-small` (Character-level multilingual)

### 3. Input Preprocessing

ByT5 requires a task prefix to specify the translation task:

```python
# Add task prefix to input
input_text = self.task_prefix + text  # "translate English to Ibani: " + text
```

### 4. Sequence Lengths

Increased max_length from 128 to 256 to accommodate character-level tokenization:

```python
# Before
max_length=128

# After
max_length=256  # Character-level needs more tokens
```

### 5. Tokenizer Augmentation Removed

**Before:** Had to add special characters to MarianMT vocabulary
```python
self._augment_tokenizer_from_data(dataset)  # Add á, ḅ, etc.
```

**After:** No augmentation needed - ByT5 handles all characters natively
```python
# ByT5 works at byte/character level, so no need to augment tokenizer
# It naturally handles all Unicode characters including á, ḅ, etc.
```

### 6. Label Tokenization

**Before (MarianMT):**
```python
labels = self.tokenizer(
    text_target=targets,  # MarianMT uses text_target parameter
    max_length=128
)
```

**After (ByT5):**
```python
labels = self.tokenizer(
    targets,  # ByT5 tokenizes labels directly
    max_length=256
)
```

## Files Modified

1. **`huggingface_translator.py`** - Core translator implementation
   - Changed imports from MarianMT to ByT5
   - Added task prefix
   - Removed tokenizer augmentation
   - Updated sequence lengths

2. **`api_server.py`** - API description updated
   - Changed description from "MarianMT model" to "ByT5 model"

3. **`README.md`** - Documentation updated
   - Updated model information
   - Changed base model references
   - Added ByT5 advantages

## Training Considerations

### Recommended Settings for ByT5:

```python
translator.train_model(
    training_data_file="ibani_eng_training_data.json",
    output_dir="./ibani_model",
    num_epochs=10,  # ByT5 may need more epochs
    batch_size=2,   # Smaller batch size due to longer sequences
    learning_rate=5e-5
)
```

### Why Different Settings?

- **More Epochs**: Character-level models need more training iterations
- **Smaller Batch Size**: Character sequences are longer, requiring more memory
- **Same Learning Rate**: 5e-5 works well for both models

## Expected Improvements

1. **Better Character Preservation**: Tonal marks (á, ḅ) will be preserved correctly
2. **No Spacing Issues**: Characters won't be separated by spaces
3. **Better Generalization**: Character-level understanding helps with unseen words
4. **More Robust**: Less likely to produce gibberish for unknown inputs

## Testing the Migration

Run the test script to verify ByT5 is working:

```bash
python test_byt5_migration.py
```

This will test:
- Model loading
- Character preservation (á, ḅ, etc.)
- Translation quality
- API compatibility

## Deployment Notes

### HuggingFace Hub

When uploading to HuggingFace Hub, use a new repository name:
- Old: `williampepple1/ibani-translator` (MarianMT)
- New: `williampepple1/ibani-byt5-translator` (ByT5)

### Environment Variables

Update your environment variables:
```bash
HF_MODEL_REPO=williampepple1/ibani-byt5-translator
```

## Backward Compatibility

⚠️ **Important**: Models trained with MarianMT are NOT compatible with ByT5.

You must:
1. Retrain your model using the new ByT5 implementation
2. Update any deployed models
3. Update environment variables to point to the new model

## Next Steps

1. **Train New Model**: Run `python train_from_ibani_eng.py`
2. **Test Thoroughly**: Verify special characters are preserved
3. **Upload to HuggingFace**: Share your trained ByT5 model
4. **Update Deployment**: Deploy the new model to your API

## Resources

- [ByT5 Paper](https://arxiv.org/abs/2105.13626)
- [ByT5 on HuggingFace](https://huggingface.co/docs/transformers/model_doc/byt5)
- [Google ByT5 Models](https://huggingface.co/google/byt5-small)
