"""
FastAPI Application for Ibani-English Translation using ByT5
"""

import os
import time
from typing import List, Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Global variables for model and tokenizer
model = None
tokenizer = None
device = None


class TranslationRequest(BaseModel):
    """Request model for translation"""
    text: str = Field(..., description="Text to translate", min_length=1, max_length=5000)
    source_lang: str = Field("en", description="Source language (en or ibani)")
    target_lang: str = Field("ibani", description="Target language (en or ibani)")
    max_length: int = Field(256, description="Maximum length of generated translation", ge=1, le=512)
    num_beams: int = Field(4, description="Number of beams for beam search", ge=1, le=10)
    temperature: float = Field(1.0, description="Sampling temperature", ge=0.1, le=2.0)


class BatchTranslationRequest(BaseModel):
    """Request model for batch translation"""
    texts: List[str] = Field(..., description="List of texts to translate", max_items=100)
    source_lang: str = Field("en", description="Source language (en or ibani)")
    target_lang: str = Field("ibani", description="Target language (en or ibani)")
    max_length: int = Field(256, description="Maximum length of generated translation", ge=1, le=512)
    num_beams: int = Field(4, description="Number of beams for beam search", ge=1, le=10)


class TranslationResponse(BaseModel):
    """Response model for translation"""
    translated_text: str
    source_lang: str
    target_lang: str
    processing_time: float


class BatchTranslationResponse(BaseModel):
    """Response model for batch translation"""
    translations: List[str]
    source_lang: str
    target_lang: str
    total_processing_time: float
    count: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    model_path: str


def load_model():
    """Load the trained ByT5 model"""
    global model, tokenizer, device
    
    model_path = os.getenv("MODEL_PATH", "models/ibani-byt5-finetuned")
    
    print(f"Loading model from {model_path}...")
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please train the model first using train.py"
        )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    print("Starting up...")
    load_model()
    yield
    # Shutdown
    print("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Ibani-English Translation API",
    description="ByT5-based translation API for Ibani and English languages",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    max_length: int = 256,
    num_beams: int = 4,
    temperature: float = 1.0,
) -> str:
    """
    Translate text using the ByT5 model
    
    Args:
        text: Text to translate
        source_lang: Source language code (en or ibani)
        target_lang: Target language code (en or ibani)
        max_length: Maximum length of generated translation
        num_beams: Number of beams for beam search
        temperature: Sampling temperature
        
    Returns:
        Translated text
    """
    # Validate languages
    valid_langs = ["en", "ibani"]
    if source_lang not in valid_langs or target_lang not in valid_langs:
        raise ValueError(f"Languages must be one of {valid_langs}")
    
    if source_lang == target_lang:
        return text
    
    # Create prompt based on direction
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
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            early_stopping=True,
            do_sample=temperature != 1.0,
        )
    
    # Decode
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return translated_text.strip()


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Ibani-English Translation API",
        "version": "1.0.0",
        "endpoints": {
            "translate": "/translate",
            "batch_translate": "/batch-translate",
            "health": "/health",
            "docs": "/docs",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        model_path=os.getenv("MODEL_PATH", "models/ibani-byt5-finetuned"),
    )


@app.post("/translate", response_model=TranslationResponse, tags=["Translation"])
async def translate(request: TranslationRequest):
    """
    Translate text from one language to another
    
    - **text**: Text to translate
    - **source_lang**: Source language (en or ibani)
    - **target_lang**: Target language (en or ibani)
    - **max_length**: Maximum length of generated translation
    - **num_beams**: Number of beams for beam search
    - **temperature**: Sampling temperature
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        start_time = time.time()
        
        translated_text = translate_text(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            max_length=request.max_length,
            num_beams=request.num_beams,
            temperature=request.temperature,
        )
        
        processing_time = time.time() - start_time
        
        return TranslationResponse(
            translated_text=translated_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            processing_time=processing_time,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation error: {str(e)}"
        )


@app.post("/batch-translate", response_model=BatchTranslationResponse, tags=["Translation"])
async def batch_translate(request: BatchTranslationRequest):
    """
    Translate multiple texts at once
    
    - **texts**: List of texts to translate (max 100)
    - **source_lang**: Source language (en or ibani)
    - **target_lang**: Target language (en or ibani)
    - **max_length**: Maximum length of generated translation
    - **num_beams**: Number of beams for beam search
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    if not request.texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No texts provided for translation"
        )
    
    try:
        start_time = time.time()
        
        translations = []
        for text in request.texts:
            translated = translate_text(
                text=text,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                max_length=request.max_length,
                num_beams=request.num_beams,
            )
            translations.append(translated)
        
        total_processing_time = time.time() - start_time
        
        return BatchTranslationResponse(
            translations=translations,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            total_processing_time=total_processing_time,
            count=len(translations),
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch translation error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
