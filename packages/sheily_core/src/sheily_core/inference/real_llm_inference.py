"""
Real LLM Inference - NO SIMULATIONS
Uses actual transformer models for text generation
"""

import logging
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class RealLLMInference:
    """
    Real LLM inference using transformers
    NO MOCKS - Actual model loading and generation
    """
    
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        """
        Initialize LLM inference
        
        Args:
            model_name: HuggingFace model to use
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        logger.info(f"🤖 Real LLM Inference initialized")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Device: {self.device}")
    
    def load_model(self):
        """Load model and tokenizer"""
        try:
            logger.info(f"📥 Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated texts
        """
        try:
            if self.model is None:
                self.load_model()
            
            logger.info(f"🎯 Generating from prompt: '{prompt[:50]}...'")
            
            # Generate using pipeline
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Extract generated texts
            results = [output["generated_text"] for output in outputs]
            
            logger.info(f"✅ Generated {len(results)} sequences")
            return results
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return [f"Error: {str(e)}"]
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 200,
        temperature: float = 0.7
    ) -> str:
        """
        Chat-style generation
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        try:
            # Format messages into prompt
            prompt = self._format_chat_prompt(messages)
            
            # Generate
            results = self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=1
            )
            
            # Extract response (remove prompt)
            response = results[0][len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Chat failed: {e}")
            return f"Error: {str(e)}"
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a prompt"""
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    def embed_text(self, text: str) -> torch.Tensor:
        """
        Get embeddings for text
        
        Args:
            text: Input text
            
        Returns:
            Embedding tensor
        """
        try:
            if self.model is None:
                self.load_model()
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use last hidden state, mean pooling
                embeddings = outputs.hidden_states[-1].mean(dim=1)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Embedding failed: {e}")
            return torch.zeros(1, 768)  # Default size


# Singleton
_real_llm_inference: Optional[RealLLMInference] = None


def get_real_llm_inference(model_name: str = "microsoft/Phi-3-mini-4k-instruct") -> RealLLMInference:
    """Get singleton instance"""
    global _real_llm_inference
    
    if _real_llm_inference is None:
        _real_llm_inference = RealLLMInference(model_name)
    
    return _real_llm_inference


# Demo
if __name__ == "__main__":
    print("🤖 Real LLM Inference Demo")
    print("=" * 50)
    
    # Initialize
    llm = get_real_llm_inference()
    
    # Generate
    prompt = "Explain machine learning in simple terms:"
    results = llm.generate(prompt, max_new_tokens=50)
    
    print(f"\n📝 Generated:")
    print(results[0])
