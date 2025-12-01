
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class SimpleDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=128):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"Pregunta: {item['input']} Respuesta: {item['output']}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

def train_model():
    # Usar modelo pequeño para prueba
    model_name = "gpt2"
    
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Agregar pad token si no existe
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Cargar dataset
        dataset = SimpleDataset("data/training_data.json", tokenizer)
        
        # Configurar entrenamiento
        training_args = TrainingArguments(
            output_dir="models/sheily_simple",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            save_steps=10,
            save_total_limit=2,
            logging_steps=5,
            learning_rate=5e-5,
            warmup_steps=10,
            logging_dir="logs/training",
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        
        print("Iniciando entrenamiento...")
        trainer.train()
        
        # Guardar modelo
        trainer.save_model()
        tokenizer.save_pretrained("models/sheily_simple")
        
        print("Entrenamiento completado exitosamente!")
        return True
        
    except Exception as e:
        print(f"Error en entrenamiento: {e}")
        return False

if __name__ == "__main__":
    train_model()
