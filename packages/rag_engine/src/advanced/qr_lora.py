#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QR-LoRA: QR-Based Low-Rank Adaptation for Efficient Fine-Tuning
Based on "QR-LoRA: QR-Based Low-Rank Adaptation for Efficient Fine-Tuning of Large Language Models"

Implements ultra-efficient parameter adaptation using QR decomposition:
- QR decomposition with column pivoting for orthonormal basis
- Scalar coefficient training only (massive parameter reduction)
- Interpretable basis ordering by importance
- 1000x parameter reduction vs full fine-tuning
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        PreTrainedModel,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class QRLoRAConfig:
    """Configuration for QR-LoRA adaptation"""

    r: int = 8  # Rank for QR decomposition
    lora_alpha: float = 16.0  # LoRA scaling factor
    lora_dropout: float = 0.0  # Dropout probability
    target_modules: List[str] = None  # Modules to adapt
    qr_threshold: float = 0.5  # Energy threshold for rank selection
    use_column_pivoting: bool = True  # Use pivoted QR
    trainable_scalars_only: bool = True  # Train only scalar coefficients

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


@dataclass
class QRDecompositionResult:
    """Result of QR decomposition with pivoting"""

    Q: torch.Tensor  # Orthonormal basis
    R: torch.Tensor  # Upper triangular matrix
    P: torch.Tensor  # Permutation matrix (for pivoting)
    rank: int  # Effective rank based on threshold
    energy_retained: float  # Fraction of energy retained


class QRLoRALayer(nn.Module):
    """
    QR-LoRA layer that adapts pre-trained weights using QR decomposition

    Key innovation: Uses QR decomposition to create orthonormal basis,
    then trains only scalar coefficients for massive parameter reduction.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: QRLoRAConfig,
        pretrained_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Store frozen pre-trained weight
        self.pretrained_weight = nn.Parameter(
            (
                pretrained_weight.clone()
                if pretrained_weight is not None
                else torch.randn(out_features, in_features)
            ),
            requires_grad=False,
        )

        # Initialize QR decomposition
        self._initialize_qr_decomposition()

        # Trainable scalar coefficients (main innovation!)
        if config.trainable_scalars_only:
            # Only train scalar coefficients λ_i
            self.lambda_coeffs = nn.Parameter(
                torch.zeros(self.qr_result.rank), requires_grad=True
            )
        else:
            # Traditional LoRA matrices (fallback)
            self.lora_A = nn.Parameter(torch.randn(config.r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, config.r))

        # Scaling and dropout
        self.scaling = config.lora_alpha / config.r
        self.dropout = (
            nn.Dropout(config.lora_dropout)
            if config.lora_dropout > 0
            else nn.Identity()
        )

    def _initialize_qr_decomposition(self):
        """Initialize QR decomposition with column pivoting"""
        W = self.pretrained_weight.data

        if self.config.use_column_pivoting:
            # QR with column pivoting (main innovation from paper)
            Q, R, P = self._qr_with_pivoting(W)
        else:
            # Standard QR
            Q, R = torch.linalg.qr(W)
            P = torch.eye(W.shape[1], device=W.device)

        # Select rank based on energy threshold
        diagonal_elements = torch.abs(torch.diag(R))
        total_energy = torch.sum(diagonal_elements**2)
        cumulative_energy = torch.cumsum(diagonal_elements**2, dim=0)
        energy_fraction = cumulative_energy / total_energy

        # Find minimum rank that retains threshold fraction of energy
        rank = torch.sum(energy_fraction < self.config.qr_threshold).item() + 1
        rank = min(rank, min(W.shape))  # Don't exceed matrix dimensions

        # Keep only top-rank components
        Q_selected = Q[:, :rank]
        R_selected = R[:rank, :]

        self.qr_result = QRDecompositionResult(
            Q=Q_selected,
            R=R_selected,
            P=P,
            rank=rank,
            energy_retained=energy_fraction[rank - 1].item(),
        )

    def _qr_with_pivoting(
        self, W: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute QR decomposition with column pivoting

        This orders columns by importance (diagonal elements of R in decreasing order)
        """
        # Use PyTorch's QR with pivoting if available, otherwise implement
        try:
            Q, R, P = torch.linalg.qr(W, mode="complete", pivoting=True)
            return Q, R, P
        except:
            # Fallback: approximate pivoted QR
            m, n = W.shape
            P = torch.arange(n, device=W.device)

            # Simple pivoting based on column norms
            col_norms = torch.norm(W, dim=0)
            _, P = torch.sort(col_norms, descending=True)

            # Reorder columns
            W_pivoted = W[:, P]

            # Standard QR on pivoted matrix
            Q, R = torch.linalg.qr(W_pivoted)

            return Q, R, P

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with QR-LoRA adaptation"""
        # Validar dimensiones de entrada
        if x.dim() > 2:
            # Si x tiene más de 2 dimensiones (batch, seq_len, hidden), aplanar
            original_shape = x.shape
            x = x.view(-1, x.shape[-1])  # (batch * seq_len, hidden)
            needs_reshape = True
        else:
            needs_reshape = False
        
        # Validar que las dimensiones coincidan
        if x.shape[-1] != self.in_features:
            raise RuntimeError(
                f"Dimension mismatch in QRLoRALayer: "
                f"input has {x.shape[-1]} features but layer expects {self.in_features}. "
                f"Input shape: {x.shape}, Weight shape: {self.pretrained_weight.shape}"
            )
        
        # Original weight matrix multiplication
        original_output = torch.matmul(x, self.pretrained_weight.t())

        # QR-LoRA adaptation
        if self.config.trainable_scalars_only:
            # Main innovation: Train only scalar coefficients
            delta_W = self._compute_delta_W_from_scalars()
        else:
            # Traditional LoRA (fallback)
            delta_W = self._compute_delta_W_lora()

        # Validar dimensiones de delta_W
        if delta_W.shape != self.pretrained_weight.shape:
            raise RuntimeError(
                f"Dimension mismatch in delta_W: "
                f"delta_W shape {delta_W.shape} != weight shape {self.pretrained_weight.shape}"
            )

        # Apply adaptation
        adapted_weight = self.pretrained_weight + delta_W
        adapted_output = torch.matmul(x, adapted_weight.t())

        # Combine with scaling and dropout
        lora_output = self.dropout(adapted_output - original_output)
        result = original_output + self.scaling * lora_output
        
        # Restaurar forma original si fue necesario
        if needs_reshape:
            result = result.view(original_shape[:-1] + (result.shape[-1],))
        
        return result

    def _compute_delta_W_from_scalars(self) -> torch.Tensor:
        """
        Compute weight update from scalar coefficients only

        This is the key innovation: ∆W = Q * Λ * R
        Where:
        - Q: (out_features, rank) - orthonormal basis
        - Λ: (rank, rank) - diagonal matrix of scalar coefficients
        - R: (rank, in_features) - upper triangular matrix
        
        Result: (out_features, in_features)
        """
        Q = self.qr_result.Q  # (out_features, rank)
        R = self.qr_result.R  # (rank, in_features)

        # Expand scalar coefficients to diagonal matrix
        lambda_diag = torch.diag(self.lambda_coeffs)  # (rank, rank)

        # Compute update: Q * Λ * R
        # Q * lambda_diag: (out_features, rank) * (rank, rank) = (out_features, rank)
        # (Q * lambda_diag) * R: (out_features, rank) * (rank, in_features) = (out_features, in_features)
        delta_W = torch.matmul(torch.matmul(Q, lambda_diag), R)

        return delta_W

    def _compute_delta_W_lora(self) -> torch.Tensor:
        """Traditional LoRA computation (fallback)"""
        return torch.matmul(self.lora_B, self.lora_A)

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        if self.config.trainable_scalars_only:
            return self.qr_result.rank  # Only scalar coefficients
        else:
            return (
                self.config.r * self.in_features + self.out_features * self.config.r
            )  # Traditional LoRA

    def get_parameter_reduction_ratio(self) -> float:
        """Get parameter reduction ratio vs full fine-tuning"""
        full_params = self.in_features * self.out_features
        lora_params = self.get_trainable_parameters()
        return full_params / lora_params if lora_params > 0 else float("inf")


class QRLoRAModel(nn.Module):
    """
    Complete model with QR-LoRA adaptation applied to target modules
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        config: QRLoRAConfig,
        target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.target_modules = target_modules or config.target_modules

        # Replace target modules with QR-LoRA versions
        self._replace_modules_with_qr_lora()

        # Freeze base model parameters
        self._freeze_base_model()

    def _replace_modules_with_qr_lora(self):
        """Replace target linear layers with QR-LoRA versions"""
        replaced_count = 0
        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    try:
                        # Create QR-LoRA replacement
                        qr_lora_layer = QRLoRALayer(
                            in_features=module.in_features,
                            out_features=module.out_features,
                            config=self.config,
                            pretrained_weight=module.weight.data.clone(),
                        )

                        # Replace in parent module
                        parent_name = ".".join(name.split(".")[:-1])
                        child_name = name.split(".")[-1]

                        parent = self.base_model
                        if parent_name:
                            for part in parent_name.split("."):
                                parent = getattr(parent, part)

                        setattr(parent, child_name, qr_lora_layer)
                        replaced_count += 1
                    except Exception as e:
                        import warnings
                        warnings.warn(
                            f"Failed to replace {name} with QRLoRA: {e}. "
                            f"Keeping original layer."
                        )
        
        if replaced_count == 0:
            import warnings
            warnings.warn(
                f"No layers were replaced with QRLoRA. "
                f"Target modules: {self.target_modules}. "
                f"Check if the model architecture matches the target module names."
            )

    def _freeze_base_model(self):
        """Freeze all base model parameters except QR-LoRA layers"""
        for name, param in self.base_model.named_parameters():
            # Only train QR-LoRA parameters
            if "lambda_coeffs" in name or "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, **kwargs):
        """Forward pass through adapted model"""
        return self.base_model(**kwargs)

    def get_trainable_parameter_count(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_stats(self) -> Dict[str, Any]:
        """Get detailed parameter statistics"""
        total_params = sum(p.numel() for p in self.base_model.parameters())
        trainable_params = self.get_trainable_parameter_count()

        # Count parameters by layer type
        qr_lora_params = 0
        traditional_lora_params = 0

        for name, module in self.named_modules():
            if isinstance(module, QRLoRALayer):
                if module.config.trainable_scalars_only:
                    qr_lora_params += module.get_trainable_parameters()
                else:
                    traditional_lora_params += module.get_trainable_parameters()

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "qr_lora_parameters": qr_lora_params,
            "traditional_lora_parameters": traditional_lora_params,
            "parameter_reduction_ratio": (
                total_params / trainable_params
                if trainable_params > 0
                else float("inf")
            ),
            "trainable_percentage": (
                (trainable_params / total_params) * 100 if total_params > 0 else 0
            ),
        }


class QRLoRATrainer:
    """
    Trainer for QR-LoRA fine-tuning
    """

    def __init__(
        self,
        model: QRLoRAModel,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        # Optimizer (AdamW recommended for transformers)
        self.optimizer = AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = None

    def train_epoch(
        self,
        dataloader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> Dict[str, float]:
        """Train for one epoch"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"   Iniciando época - Device: {device}, Batches: {len(dataloader)}")
        self.model.train()
        self.model.to(device)
        logger.info("   Modelo movido a device")

        total_loss = 0.0
        num_batches = 0

        logger.info("   Iniciando loop de entrenamiento...")
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Log cada 10 batches para ver progreso
                if batch_idx % 10 == 0:
                    logger.info(f"   Procesando batch {batch_idx + 1}/{len(dataloader)}")
                
                # Move batch to device
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                total_loss += loss.item()
                num_batches += 1
                
            except Exception as batch_error:
                logger.error(f"   ❌ Error en batch {batch_idx + 1}: {batch_error}")
                raise

        logger.info(f"   ✅ Época completada: {num_batches} batches procesados")
        return {
            "train_loss": total_loss / num_batches if num_batches > 0 else 0.0,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def evaluate(
        self,
        dataloader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        self.model.to(device)

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                outputs = self.model(**batch)
                loss = outputs.loss

                total_loss += loss.item()
                num_batches += 1

        return {"eval_loss": total_loss / num_batches}


def create_qr_lora_model(
    model_name: str, config: QRLoRAConfig, cache_dir: Optional[str] = None
) -> QRLoRAModel:
    """
    Create a QR-LoRA adapted model

    Args:
        model_name: HuggingFace model name
        config: QR-LoRA configuration
        cache_dir: Cache directory for models

    Returns:
        QR-LoRA adapted model
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library required for QR-LoRA")

    # Load base model - Use AutoModelForCausalLM for language models
    # This fixes dimension mismatch errors when using causal language models like Phi-3
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
    except Exception as e:
        # Fallback to sequence classification if causal fails
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, cache_dir=cache_dir
            )
        except Exception:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

    # Create QR-LoRA model
    qr_lora_model = QRLoRAModel(model, config)

    return qr_lora_model


def benchmark_qr_lora_efficiency(
    model_name: str = "microsoft/DialoGPT-small",
    tasks: List[str] = ["mnli", "mrpc", "sst2"],
) -> Dict[str, Any]:
    """
    Benchmark QR-LoRA parameter efficiency

    Args:
        model_name: Base model name
        tasks: Tasks to benchmark

    Returns:
        Benchmarking results
    """
    results = {}

    # Different QR-LoRA configurations
    configs = [
        QRLoRAConfig(r=8, qr_threshold=0.5, trainable_scalars_only=True),
        QRLoRAConfig(r=16, qr_threshold=0.7, trainable_scalars_only=True),
        QRLoRAConfig(r=32, qr_threshold=0.8, trainable_scalars_only=True),
        QRLoRAConfig(r=2, trainable_scalars_only=False),  # Traditional LoRA baseline
    ]

    for config in configs:
        try:
            # Create model
            model = create_qr_lora_model(model_name, config)

            # Get parameter statistics
            stats = model.get_parameter_stats()

            config_name = (
                f"QR-LoRA_r{config.r}_thresh{config.qr_threshold}"
                if config.trainable_scalars_only
                else f"LoRA_r{config.r}"
            )

            results[config_name] = {
                "total_parameters": stats["total_parameters"],
                "trainable_parameters": stats["trainable_parameters"],
                "parameter_reduction_ratio": stats["parameter_reduction_ratio"],
                "trainable_percentage": stats["trainable_percentage"],
                "config": config.__dict__,
            }

        except Exception as e:
            print(f"Error benchmarking config {config}: {e}")
            continue

    return results


# Example usage and demonstration
def demo_qr_lora():
    """
    Demonstrate QR-LoRA capabilities
    """
    print("🔬 QR-LoRA Demonstration")
    print("=" * 50)

    # Benchmark parameter efficiency
    print("📊 Parameter Efficiency Benchmark:")
    print("-" * 40)

    try:
        results = benchmark_qr_lora_efficiency()

        for config_name, stats in results.items():
            print(f"{config_name}:")
            print(".1f")
            print(".0f")
            print(".2f")
            print()

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        print("Note: Requires transformers and torch to be installed")

    # Show theoretical advantages
    print("🎯 QR-LoRA Key Advantages:")
    print("-" * 30)
    print("• 1000× parameter reduction vs full fine-tuning")
    print("• Orthonormal basis ensures numerical stability")
    print("• Column pivoting provides interpretable importance ordering")
    print("• Scalar-only training minimizes memory footprint")
    print("• Maintains or exceeds full fine-tuning performance")
    print("• Particularly effective on GLUE benchmark tasks")


if __name__ == "__main__":
    demo_qr_lora()
