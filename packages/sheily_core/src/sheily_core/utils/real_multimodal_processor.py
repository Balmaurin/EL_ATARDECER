"""
Real Multimodal Processor - NO SIMULATIONS
Uses actual AI models: Whisper, CLIP, Tesseract
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class RealMultiModalProcessor:
    """
    Real multimodal processing using actual AI models
    - Whisper for speech-to-text
    - CLIP for vision understanding
    - Tesseract for OCR
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🎯 Initializing Real Multimodal Processor on {self.device}")
        
        # Models will be loaded on first use (lazy loading)
        self.whisper_model = None
        self.clip_model = None
        self.clip_processor = None
        self.tesseract_available = False
        
        self._check_tesseract()
    
    def _check_tesseract(self):
        """Check if Tesseract is available"""
        try:
            import pytesseract
            # Try to get version
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info("✅ Tesseract OCR available")
        except Exception as e:
            logger.warning(f"⚠️ Tesseract not available: {e}")
            logger.warning("   Install from: https://github.com/UB-Mannheim/tesseract/wiki")
    
    def _load_whisper(self):
        """Load Whisper model for speech recognition"""
        if self.whisper_model is None:
            try:
                import whisper
                logger.info("📥 Loading Whisper model...")
                self.whisper_model = whisper.load_model("base", device=self.device)
                logger.info("✅ Whisper model loaded")
            except ImportError:
                logger.error("❌ Whisper not installed. Run: pip install openai-whisper")
                raise
    
    def _load_clip(self):
        """Load CLIP model for vision understanding"""
        if self.clip_model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor
                logger.info("📥 Loading CLIP model...")
                
                model_name = "openai/clip-vit-base-patch32"
                self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
                self.clip_processor = CLIPProcessor.from_pretrained(model_name)
                
                logger.info("✅ CLIP model loaded")
            except ImportError:
                logger.error("❌ Transformers not installed. Run: pip install transformers")
                raise
    
    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Real speech-to-text using Whisper
        
        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., 'en', 'es')
            
        Returns:
            Dict with transcription and metadata
        """
        try:
            self._load_whisper()
            
            logger.info(f"🎤 Transcribing audio: {audio_path}")
            
            # Real Whisper transcription
            result = self.whisper_model.transcribe(
                str(audio_path),
                language=language,
                fp16=(self.device == "cuda")
            )
            
            logger.info(f"✅ Transcription complete ({len(result['text'])} chars)")
            
            return {
                "success": True,
                "text": result["text"],
                "language": result.get("language"),
                "segments": result.get("segments", []),
                "duration": sum(seg["end"] - seg["start"] for seg in result.get("segments", []))
            }
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }
    
    def extract_text_from_image(
        self,
        image_path: Union[str, Path]
    ) -> Dict[str, any]:
        """
        Real OCR using Tesseract
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict with extracted text
        """
        try:
            if not self.tesseract_available:
                raise RuntimeError("Tesseract not available")
            
            import pytesseract
            
            logger.info(f"📄 Extracting text from image: {image_path}")
            
            # Load image
            image = Image.open(image_path)
            
            # Real OCR
            text = pytesseract.image_to_string(image)
            
            # Get detailed data
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            confidence = sum(int(c) for c in data['conf'] if c != '-1') / len([c for c in data['conf'] if c != '-1'])
            
            logger.info(f"✅ OCR complete ({len(text)} chars, {confidence:.1f}% confidence)")
            
            return {
                "success": True,
                "text": text.strip(),
                "confidence": confidence,
                "word_count": len(text.split())
            }
            
        except Exception as e:
            logger.error(f"❌ OCR failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }
    
    def analyze_image(
        self,
        image_path: Union[str, Path],
        text_queries: List[str]
    ) -> Dict[str, any]:
        """
        Real image analysis using CLIP
        
        Args:
            image_path: Path to image
            text_queries: List of text descriptions to match against
            
        Returns:
            Dict with similarity scores for each query
        """
        try:
            self._load_clip()
            
            logger.info(f"🖼️ Analyzing image: {image_path}")
            logger.info(f"   Queries: {text_queries}")
            
            # Load image
            image = Image.open(image_path)
            
            # Process with CLIP
            inputs = self.clip_processor(
                text=text_queries,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get similarity scores
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Format results
            results = {}
            for i, query in enumerate(text_queries):
                results[query] = float(probs[0][i])
            
            # Find best match
            best_match = max(results.items(), key=lambda x: x[1])
            
            logger.info(f"✅ Analysis complete. Best match: '{best_match[0]}' ({best_match[1]:.2%})")
            
            return {
                "success": True,
                "scores": results,
                "best_match": {
                    "query": best_match[0],
                    "score": best_match[1]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Image analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "scores": {}
            }
    
    def generate_image_caption(
        self,
        image_path: Union[str, Path]
    ) -> Dict[str, any]:
        """
        Generate caption for image using CLIP
        
        Args:
            image_path: Path to image
            
        Returns:
            Dict with generated caption
        """
        # Common image descriptions
        queries = [
            "a photo of a person",
            "a photo of an animal",
            "a photo of nature",
            "a photo of food",
            "a photo of a building",
            "a photo of a vehicle",
            "a photo of text or document",
            "a photo of art or painting",
            "a screenshot of software",
            "a diagram or chart"
        ]
        
        result = self.analyze_image(image_path, queries)
        
        if result["success"]:
            caption = result["best_match"]["query"]
            confidence = result["best_match"]["score"]
            
            return {
                "success": True,
                "caption": caption,
                "confidence": confidence
            }
        
        return result


# Singleton instance
_real_multimodal_processor: Optional[RealMultiModalProcessor] = None


def get_real_multimodal_processor() -> RealMultiModalProcessor:
    """Get singleton instance"""
    global _real_multimodal_processor
    
    if _real_multimodal_processor is None:
        _real_multimodal_processor = RealMultiModalProcessor()
    
    return _real_multimodal_processor


# Demo
if __name__ == "__main__":
    processor = get_real_multimodal_processor()
    
    print("🎯 Real Multimodal Processor Demo")
    print("=" * 50)
    print(f"Device: {processor.device}")
    print(f"Tesseract: {'✅' if processor.tesseract_available else '❌'}")
