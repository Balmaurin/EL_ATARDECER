"""
Hippocampus Neural - Memoria Episódica Neural
==============================================

Transformer encoder pequeño + FAISS para memoria episódica.
Genera embeddings de experiencias y permite búsqueda rápida.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging
import numpy as np
import json
from datetime import datetime

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, memory search will be limited")

logger = logging.getLogger(__name__)


class TransformerEncoderSmall(nn.Module):
    """
    Transformer encoder pequeño para codificar experiencias.
    
    Arquitectura:
    - 2-3 layers
    - 128-256 dim
    - Attention heads: 4-8
    """
    
    def __init__(self, vocab_size: int = 1000, d_model: int = 128, nhead: int = 4, 
                 num_layers: int = 2, dim_feedforward: int = 256, max_seq_len: int = 128):
        """
        Inicializa el transformer encoder.
        
        Args:
            vocab_size: Tamaño del vocabulario
            d_model: Dimensión del modelo
            nhead: Número de heads de atención
            num_layers: Número de capas
            dim_feedforward: Dimensión de feedforward
            max_seq_len: Longitud máxima de secuencia
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pooling (CLS token o mean pooling)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len] (token IDs)
            mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Embedding [batch_size, d_model]
        """
        batch_size, seq_len = x.shape
        
        # Embedding
        x = self.embedding(x) * np.sqrt(self.d_model)
        
        # Posicional encoding
        if seq_len <= self.max_seq_len:
            x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        else:
            x = x + self.pos_encoding[-1].unsqueeze(0).unsqueeze(0)
        
        # Transformer
        if mask is not None:
            # Convertir mask a formato de transformer (True = attend, False = mask)
            mask = ~mask.bool()
        
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Mean pooling
        x = x.mean(dim=1)  # [batch_size, d_model]
        
        # Output projection
        x = self.output_proj(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalization
        
        return x


class SimpleTokenizer:
    """
    Tokenizador simple para experiencias.
    En producción usar sentence-transformers o similar.
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.unk_id = 0
        self.pad_id = 1
        
        # Inicializar con tokens especiales
        self.word_to_id["<UNK>"] = self.unk_id
        self.word_to_id["<PAD>"] = self.pad_id
        self.id_to_word[self.unk_id] = "<UNK>"
        self.id_to_word[self.pad_id] = "<PAD>"
        
        self.next_id = 2
    
    def encode(self, text: str, max_length: int = 128) -> List[int]:
        """
        Codifica texto a IDs de tokens.
        
        Args:
            text: Texto a codificar
            max_length: Longitud máxima
            
        Returns:
            Lista de token IDs
        """
        words = text.lower().split()
        token_ids = []
        
        for word in words:
            if word not in self.word_to_id:
                if self.next_id < self.vocab_size:
                    self.word_to_id[word] = self.next_id
                    self.id_to_word[self.next_id] = word
                    self.next_id += 1
                else:
                    token_ids.append(self.unk_id)
            else:
                token_ids.append(self.word_to_id[word])
        
        # Padding o truncado
        if len(token_ids) < max_length:
            token_ids = token_ids + [self.pad_id] * (max_length - len(token_ids))
        else:
            token_ids = token_ids[:max_length]
        
        return token_ids


class HippocampusNeural:
    """
    Sistema completo de memoria episódica neural.
    """
    
    def __init__(self, model_path: Optional[str] = None, index_path: Optional[str] = None, 
                 device: str = "cpu", embedding_dim: int = 128):
        """
        Inicializa el sistema de memoria.
        
        Args:
            model_path: Ruta al modelo guardado
            index_path: Ruta al índice FAISS
            device: Dispositivo
            embedding_dim: Dimensión de embeddings
        """
        self.device = torch.device(device)
        self.embedding_dim = embedding_dim
        
        # Modelo
        self.model = TransformerEncoderSmall(d_model=embedding_dim).to(self.device)
        self.tokenizer = SimpleTokenizer()
        
        # FAISS index
        self.index = None
        self.memories = []  # Lista de memorias con metadata
        self.index_path = index_path
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        if FAISS_AVAILABLE:
            self._init_faiss_index()
            if index_path and Path(index_path).exists():
                self.load_index(index_path)
        else:
            logger.warning("FAISS not available, using simple list-based search")
    
    def _init_faiss_index(self):
        """Inicializa el índice FAISS."""
        if FAISS_AVAILABLE:
            # IndexFlatL2 para CPU (búsqueda exacta)
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info(f"FAISS index initialized: {self.embedding_dim}D")
    
    def encode_experience(self, experience: Dict[str, Any]) -> np.ndarray:
        """
        Codifica una experiencia en un embedding.
        
        Args:
            experience: Dict con información de la experiencia
                - content: str (texto de la experiencia)
                - timestamp: str
                - context: dict
                
        Returns:
            Embedding [embedding_dim]
        """
        self.model.eval()
        
        with torch.no_grad():
            # Tokenizar contenido
            content = experience.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content)
            
            token_ids = self.tokenizer.encode(content, max_length=128)
            
            # Convertir a tensor
            x = torch.LongTensor([token_ids]).to(self.device)  # [1, seq_len]
            
            # Forward pass
            embedding = self.model(x)
            embedding = embedding.squeeze(0).cpu().numpy()
            
            return embedding
    
    def store_memory(self, experience: Dict[str, Any]) -> int:
        """
        Almacena una experiencia en memoria.
        
        Args:
            experience: Información de la experiencia
            
        Returns:
            ID de la memoria almacenada
        """
        # Codificar
        embedding = self.encode_experience(experience)
        
        # Añadir metadata
        memory_id = len(self.memories)
        memory_entry = {
            "id": memory_id,
            "embedding": embedding,
            "content": experience.get("content", ""),
            "timestamp": experience.get("timestamp", datetime.now().isoformat()),
            "context": experience.get("context", {}),
            "relevance": experience.get("relevance", 1.0)
        }
        
        self.memories.append(memory_entry)
        
        # Añadir a FAISS
        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(np.array([embedding], dtype=np.float32))
        
        logger.debug(f"Memory stored: {memory_id}")
        return memory_id
    
    def retrieve(self, query: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recupera memorias relevantes dado un query.
        
        Args:
            query: Query de búsqueda
            top_k: Número de resultados
            
        Returns:
            Lista de memorias relevantes
        """
        if len(self.memories) == 0:
            return []
        
        # Codificar query
        query_embedding = self.encode_experience(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Búsqueda
        if FAISS_AVAILABLE and self.index is not None:
            # Búsqueda FAISS
            k = min(top_k, len(self.memories))
            distances, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.memories):
                    memory = self.memories[idx].copy()
                    memory["distance"] = float(distances[0][i])
                    memory["similarity"] = 1.0 / (1.0 + memory["distance"])  # Convertir distancia a similitud
                    results.append(memory)
        else:
            # Búsqueda simple por similitud coseno
            query_vec = query_embedding[0]
            similarities = []
            
            for memory in self.memories:
                mem_vec = memory["embedding"]
                similarity = np.dot(query_vec, mem_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(mem_vec) + 1e-8)
                similarities.append((similarity, memory))
            
            # Ordenar por similitud
            similarities.sort(reverse=True, key=lambda x: x[0])
            
            results = []
            for similarity, memory in similarities[:top_k]:
                result = memory.copy()
                result["similarity"] = float(similarity)
                result["distance"] = 1.0 - similarity
                results.append(result)
        
        return results
    
    def save_model(self, path: str) -> bool:
        """Guarda el modelo."""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'tokenizer': {
                    'word_to_id': self.tokenizer.word_to_id,
                    'id_to_word': self.tokenizer.id_to_word,
                    'next_id': self.tokenizer.next_id
                },
                'embedding_dim': self.embedding_dim
            }, path)
            logger.info(f"Hippocampus model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving hippocampus model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Carga el modelo."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Restaurar tokenizer
            if 'tokenizer' in checkpoint:
                self.tokenizer.word_to_id = checkpoint['tokenizer']['word_to_id']
                self.tokenizer.id_to_word = checkpoint['tokenizer']['id_to_word']
                self.tokenizer.next_id = checkpoint['tokenizer']['next_id']
            
            logger.info(f"Hippocampus model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading hippocampus model: {e}")
            return False
    
    def save_index(self, path: str) -> bool:
        """Guarda el índice FAISS y memorias."""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Guardar índice FAISS
            if FAISS_AVAILABLE and self.index is not None:
                faiss.write_index(self.index, str(path) + ".faiss")
            
            # Guardar memorias (sin embeddings, ya están en FAISS)
            memories_metadata = []
            for mem in self.memories:
                metadata = {
                    "id": mem["id"],
                    "content": mem["content"],
                    "timestamp": mem["timestamp"],
                    "context": mem["context"],
                    "relevance": mem["relevance"]
                }
                memories_metadata.append(metadata)
            
            with open(str(path) + ".json", 'w', encoding='utf-8') as f:
                json.dump(memories_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Hippocampus index saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving hippocampus index: {e}")
            return False
    
    def load_index(self, path: str) -> bool:
        """Carga el índice FAISS y memorias."""
        try:
            # Cargar índice FAISS
            if FAISS_AVAILABLE:
                index_file = Path(str(path) + ".faiss")
                if index_file.exists():
                    self.index = faiss.read_index(str(index_file))
                    logger.info(f"FAISS index loaded from {index_file}")
            
            # Cargar metadata
            metadata_file = Path(str(path) + ".json")
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    memories_metadata = json.load(f)
                
                # Reconstruir memorias (necesitamos re-embedding o cargar desde checkpoint)
                # Por ahora, solo cargamos metadata
                self.memories = []
                for meta in memories_metadata:
                    # Necesitaríamos re-embedding aquí, pero por simplicidad
                    # asumimos que el índice FAISS ya tiene los embeddings
                    self.memories.append({
                        "id": meta["id"],
                        "content": meta["content"],
                        "timestamp": meta["timestamp"],
                        "context": meta["context"],
                        "relevance": meta["relevance"],
                        "embedding": None  # Se recuperaría del índice si necesario
                    })
                
                logger.info(f"Loaded {len(self.memories)} memories from {metadata_file}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading hippocampus index: {e}")
            return False
