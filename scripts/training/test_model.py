
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def test_model():
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("models/sheily_simple")
        model = GPT2LMHeadModel.from_pretrained("models/sheily_simple")
        
        test_input = "Pregunta: Hola"
        inputs = tokenizer.encode(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=50, pad_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Entrada: {test_input}")
        print(f"Respuesta: {response}")
        
        return True
    except Exception as e:
        print(f"Error en test: {e}")
        return False

if __name__ == "__main__":
    test_model()
