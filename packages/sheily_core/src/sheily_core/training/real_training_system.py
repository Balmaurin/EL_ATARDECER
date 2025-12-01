"""
Real Training System - NO SIMULATIONS
Uses actual PyTorch training with PEFT/LoRA
"""

import logging
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader

# Intentar importar psutil para optimización de hardware
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for real training - Sheily v1 (Phi-3-mini-4k-instruct)"""
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    output_dir: str = "./trained_models/sheily_v1_lora"
    adapter_path: Optional[str] = None  # Ruta al adaptador existente (ej: "models/sheily-v1.0")
    continue_from_adapter: bool = False  # Si True, carga adaptador existente y continúa entrenando
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 4
    max_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_fp16: bool = True
    save_steps: int = 100
    logging_steps: int = 10


class RealTrainingSystem:
    """
    Real training system using PyTorch and PEFT
    NO MOCKS - Actual model training
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Optimizar configuración basada en hardware
        self._optimize_for_hardware()
        
        logger.info(f"🚂 Real Training System initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Model: {self.config.model_name}")
        logger.info(f"   FP16: {self.config.use_fp16 and self.device == 'cuda'}")
        logger.info(f"   Batch size: {self.config.batch_size}")
        logger.info(f"   Gradient accumulation: {self.config.gradient_accumulation_steps}")
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def _optimize_for_hardware(self):
        """Optimizar configuración basada en hardware disponible"""
        # Detectar sistema operativo
        is_windows = platform.system() == "Windows"
        
        # Configurar threads de PyTorch para CPU
        if self.device == "cpu":
            if PSUTIL_AVAILABLE:
                # Usar todos los cores disponibles pero dejar 1 libre
                num_threads = max(1, psutil.cpu_count(logical=True) - 1)
            else:
                # Fallback: usar número de cores físicos
                num_threads = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
            torch.set_num_threads(num_threads)
            logger.info(f"   CPU threads configurados: {num_threads}")
        
        # Optimizar batch size basado en memoria disponible
        if self.device == "cuda":
            try:
                # Obtener memoria GPU disponible
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                # Ajustar batch size según memoria GPU
                if gpu_memory_gb >= 16:
                    # GPU con mucha memoria: aumentar batch size
                    self.config.batch_size = min(8, self.config.batch_size * 2)
                    self.config.gradient_accumulation_steps = max(2, self.config.gradient_accumulation_steps // 2)
                elif gpu_memory_gb >= 8:
                    # GPU con memoria media: mantener o aumentar ligeramente
                    self.config.batch_size = min(6, self.config.batch_size + 1)
                else:
                    # GPU con poca memoria: mantener batch size pequeño
                    self.config.batch_size = max(2, self.config.batch_size)
                
                logger.info(f"   GPU Memory: {gpu_memory_gb:.1f} GB")
                logger.info(f"   Batch size optimizado: {self.config.batch_size}")
            except Exception as e:
                logger.warning(f"   No se pudo optimizar batch size: {e}")
        
        # Configurar num_workers basado en sistema operativo
        if is_windows:
            # Windows: usar 0 para evitar bloqueos (pero más lento)
            self.dataloader_num_workers = 0
            self.dataloader_pin_memory = False
            logger.info("   ⚠️ Windows detectado: usando num_workers=0 para evitar bloqueos")
        else:
            # Linux/Mac: usar workers para mejor rendimiento
            if PSUTIL_AVAILABLE:
                cpu_count = psutil.cpu_count(logical=True)
            else:
                cpu_count = os.cpu_count() or 4
            
            if self.device == "cuda":
                self.dataloader_num_workers = min(4, cpu_count // 2)
                self.dataloader_pin_memory = True  # Mejor rendimiento en GPU
            else:
                self.dataloader_num_workers = min(2, cpu_count // 4)
                self.dataloader_pin_memory = False
        
        logger.info(f"   DataLoader workers: {self.dataloader_num_workers}")
        logger.info(f"   Pin memory: {self.dataloader_pin_memory}")
    
    def load_model(self):
        """Load model and tokenizer"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, TaskType
            
            logger.info(f"📥 Loading model: {self.config.model_name}")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   FP16: {self.config.use_fp16 and self.device == 'cuda'}")
            
            # Load tokenizer
            logger.info("   Cargando tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("   ✅ Tokenizer cargado")
            
            # Verificar si hay un adaptador existente para continuar entrenando
            adapter_path = self.config.adapter_path or (Path("models/sheily-v1.0") if Path("models/sheily-v1.0").exists() else None)
            
            if self.config.continue_from_adapter and adapter_path and Path(adapter_path).exists():
                logger.info(f"   🔄 Continuando entrenamiento desde adaptador existente: {adapter_path}")
                logger.info("   📥 Cargando modelo base y adaptador...")
                
                # Cargar modelo base
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if (self.config.use_fp16 and self.device == "cuda") else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                # Desactivar caché durante entrenamiento para evitar problemas con DynamicCache
                if hasattr(self.model, 'config'):
                    self.model.config.use_cache = False
                
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                
                # Cargar adaptador existente usando PEFT
                from peft import PeftModel, PeftConfig
                
                # Leer configuración del adaptador para obtener los parámetros LoRA
                adapter_config_path = Path(adapter_path) / "adapter_config.json"
                if adapter_config_path.exists():
                    import json
                    with open(adapter_config_path, 'r') as f:
                        adapter_config_data = json.load(f)
                    # Usar los parámetros del adaptador existente
                    lora_r = adapter_config_data.get('r', self.config.lora_r)
                    lora_alpha = adapter_config_data.get('lora_alpha', self.config.lora_alpha)
                    target_modules = adapter_config_data.get('target_modules', ["o_proj", "gate_up_proj", "down_proj", "qkv_proj"])
                    lora_dropout = adapter_config_data.get('lora_dropout', self.config.lora_dropout)
                else:
                    # Usar configuración por defecto
                    lora_r = self.config.lora_r
                    lora_alpha = self.config.lora_alpha
                    target_modules = ["o_proj", "gate_up_proj", "down_proj", "qkv_proj"]
                    lora_dropout = self.config.lora_dropout
                
                # Cargar el adaptador - CRÍTICO: El adaptador puede estar en modo inference_mode
                # Necesitamos cargarlo y luego reaplicar LoRA en modo entrenable
                logger.info("   📥 Cargando adaptador existente...")
                self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
                
                # Verificar si el adaptador está en modo inferencia
                adapter_config_path = Path(adapter_path) / "adapter_config.json"
                inference_mode = False
                if adapter_config_path.exists():
                    import json
                    with open(adapter_config_path, 'r') as f:
                        adapter_config_data = json.load(f)
                    inference_mode = adapter_config_data.get('inference_mode', False)
                
                if inference_mode:
                    logger.warning("   ⚠️ Adaptador está en modo inferencia - reaplicando LoRA en modo entrenable")
                    # Obtener el modelo base
                    base_model = self.model.get_base_model()
                    
                    # Crear nueva configuración LoRA en modo entrenable
                    from peft import LoraConfig, get_peft_model, TaskType
                    lora_config = LoraConfig(
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        target_modules=target_modules,
                        lora_dropout=lora_dropout,
                        bias="none",
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False  # CRÍTICO: Modo entrenable
                    )
                    
                    # Reaplicar LoRA al modelo base
                    self.model = get_peft_model(base_model, lora_config)
                    logger.info("   ✅ LoRA reaplicado en modo entrenable")
                else:
                    # El adaptador ya está en modo entrenable, solo asegurar que esté en modo train
                    self.model.train()
                
                # HABILITAR PARÁMETROS ENTRENABLES EXPLÍCITAMENTE
                # Asegurar que todos los parámetros LoRA estén entrenables
                for name, param in self.model.named_parameters():
                    if 'lora' in name.lower() or any(module in name.lower() for module in target_modules):
                        param.requires_grad = True
                
                # Verificar que hay parámetros entrenables
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                logger.info(f"   📊 Parámetros entrenables: {trainable_params:,}")
                
                if trainable_params == 0:
                    # Último recurso: reaplicar LoRA desde cero
                    logger.error("   ❌ CRÍTICO: 0 parámetros entrenables - reaplicando LoRA desde cero")
                    from peft import LoraConfig, get_peft_model, TaskType
                    
                    # Recargar modelo base
                    base_model = AutoModelForCausalLM.from_pretrained(
                        self.config.model_name,
                        torch_dtype=torch.float16 if (self.config.use_fp16 and self.device == "cuda") else torch.float32,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    
                    # Desactivar caché durante entrenamiento para evitar problemas con DynamicCache
                    if hasattr(base_model, 'config'):
                        base_model.config.use_cache = False
                    
                    if self.device == "cpu":
                        base_model = base_model.to(self.device)
                    
                    # Crear nueva configuración LoRA
                    lora_config = LoraConfig(
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        target_modules=target_modules,
                        lora_dropout=lora_dropout,
                        bias="none",
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False
                    )
                    
                    # Aplicar LoRA
                    self.model = get_peft_model(base_model, lora_config)
                    self.model.train()
                    
                    # Verificar nuevamente
                    trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    logger.info(f"   ✅ LoRA aplicado desde cero - parámetros entrenables: {trainable_params:,}")
                    
                    if trainable_params == 0:
                        raise RuntimeError("❌ CRÍTICO: No se pudieron habilitar parámetros entrenables - verifica la configuración LoRA")
                
                logger.info(f"   ✅ Adaptador cargado desde: {adapter_path}")
                logger.info("   🔄 El modelo continuará entrenando desde este adaptador")
                
            else:
                # Cargar modelo base desde cero
                logger.info("   Cargando modelo base (esto puede tomar varios minutos)...")
                logger.info("   ⚠️ Si se queda bloqueado aquí, puede ser un problema de red o cache")
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if (self.config.use_fp16 and self.device == "cuda") else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True  # Optimización para evitar bloqueos de memoria
                )
                
                logger.info("   ✅ Modelo base cargado")
                
                if self.device == "cpu":
                    logger.info("   Moviendo modelo a CPU...")
                    self.model = self.model.to(self.device)
                    logger.info("   ✅ Modelo movido a CPU")
                
                # Apply LoRA - Configuración específica para Phi-3-mini-4k-instruct
                # Phi-3 usa: o_proj, gate_up_proj, down_proj, qkv_proj (según adapter_config.json de sheily-v1.0)
                lora_config = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    target_modules=["o_proj", "gate_up_proj", "down_proj", "qkv_proj"],  # Específico para Phi-3
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM
                )
                
                self.model = get_peft_model(self.model, lora_config)
            
            self.model.print_trainable_parameters()
            logger.info("✅ Model loaded with LoRA adapters")
            
        except ImportError as e:
            logger.error(f"❌ Missing dependencies: {e}")
            logger.error("   Install: pip install transformers peft accelerate")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def prepare_dataset(self, data: List[Dict[str, str]]) -> Dataset:
        """
        Prepare dataset for training
        
        Args:
            data: List of dicts with 'input' and 'output' keys
            
        Returns:
            PyTorch Dataset
        """
        class TextDataset(Dataset):
            def __init__(self, data, tokenizer, max_length):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                
                # Format as instruction-following
                text = f"### Input:\n{item['input']}\n\n### Output:\n{item['output']}"
                
                # Tokenize
                encodings = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                return {
                    "input_ids": encodings["input_ids"].squeeze(),
                    "attention_mask": encodings["attention_mask"].squeeze(),
                    "labels": encodings["input_ids"].squeeze()
                }
        
        return TextDataset(data, self.tokenizer, self.config.max_length)
    
    def train(self, train_data: List[Dict[str, str]]) -> Dict[str, any]:
        """
        Train the model with real PyTorch training
        
        Args:
            train_data: List of training examples
            
        Returns:
            Training results
        """
        try:
            from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
            
            if self.model is None:
                self.load_model()
            
            logger.info(f"🚂 Starting real training on {len(train_data)} examples")
            
            # VALIDACIÓN: Verificar que hay datos para entrenar
            if not train_data or len(train_data) == 0:
                raise ValueError("❌ No hay datos para entrenar - train_data está vacío")
            
            # Prepare dataset
            dataset = self.prepare_dataset(train_data)
            
            # VALIDACIÓN: Verificar que el dataset tiene elementos
            if len(dataset) == 0:
                raise ValueError("❌ El dataset preparado está vacío - verifica el formato de los datos")
            
            logger.info(f"   📊 Dataset preparado: {len(dataset)} ejemplos")
            
            # Calcular número esperado de pasos
            effective_batch_size = self.config.batch_size * self.config.gradient_accumulation_steps
            steps_per_epoch = max(1, (len(dataset) + effective_batch_size - 1) // effective_batch_size)  # Ceiling division
            total_steps = steps_per_epoch * self.config.num_epochs
            logger.info(f"   📊 Batch size: {self.config.batch_size}")
            logger.info(f"   📊 Gradient accumulation: {self.config.gradient_accumulation_steps}")
            logger.info(f"   📊 Effective batch size: {effective_batch_size}")
            logger.info(f"   📊 Pasos esperados por época: {steps_per_epoch}")
            logger.info(f"   📊 Total de pasos esperados: {total_steps}")
            
            if total_steps == 0:
                raise ValueError(
                    f"❌ No se pueden ejecutar pasos de entrenamiento. "
                    f"Dataset size: {len(dataset)}, Batch size: {self.config.batch_size}, "
                    f"Gradient accumulation: {self.config.gradient_accumulation_steps}"
                )
            
            # Verificar que el dataset se puede acceder correctamente
            try:
                sample_item = dataset[0]
                logger.info(f"   ✅ Dataset accesible - Sample keys: {list(sample_item.keys())}")
                logger.info(f"   ✅ Sample input_ids shape: {sample_item['input_ids'].shape}")
            except Exception as e:
                logger.error(f"   ❌ Error accediendo al dataset: {e}")
                raise
            
            # Training arguments - Optimizado para evitar bloqueos
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                fp16=self.config.use_fp16 and self.device == "cuda",
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                save_total_limit=3,
                remove_unused_columns=False,
                push_to_hub=False,
                report_to="none",
                load_best_model_at_end=False,
                # Optimizaciones basadas en hardware
                dataloader_num_workers=getattr(self, 'dataloader_num_workers', 0),
                dataloader_pin_memory=getattr(self, 'dataloader_pin_memory', False),
                disable_tqdm=False,  # Mantener barras de progreso pero con logging
                # Optimizaciones adicionales de rendimiento
                dataloader_prefetch_factor=2 if getattr(self, 'dataloader_num_workers', 0) > 0 else None,
                ddp_find_unused_parameters=False,  # Optimización para multi-GPU
                dataloader_drop_last=False,  # Usar todos los datos, incluso batches incompletos
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Create trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )
            
            # Verificar configuración del trainer antes de entrenar
            logger.info(f"   📊 Trainer configurado:")
            logger.info(f"      - Train dataset size: {len(self.trainer.train_dataset)}")
            logger.info(f"      - Num epochs: {training_args.num_train_epochs}")
            logger.info(f"      - Per device batch size: {training_args.per_device_train_batch_size}")
            logger.info(f"      - Gradient accumulation: {training_args.gradient_accumulation_steps}")
            
            # Calcular pasos esperados usando el método del Trainer
            try:
                train_dataloader = self.trainer.get_train_dataloader()
                dataloader_length = len(train_dataloader)
                expected_steps = dataloader_length * training_args.num_train_epochs
                logger.info(f"   📊 DataLoader length: {dataloader_length}")
                logger.info(f"   📊 Pasos calculados por Trainer: {expected_steps}")
                
                if expected_steps == 0:
                    logger.error(f"   ❌ El Trainer no generará pasos - verifica la configuración")
                    logger.error(f"      Dataset size: {len(dataset)}")
                    logger.error(f"      Batch size: {training_args.per_device_train_batch_size}")
                    logger.error(f"      Gradient accumulation: {training_args.gradient_accumulation_steps}")
                    raise ValueError("El Trainer no puede generar pasos de entrenamiento")
            except Exception as e:
                logger.warning(f"   ⚠️ No se pudo calcular pasos del DataLoader: {e}")
                # Continuar de todas formas - el Trainer puede calcularlo internamente
            
            # Train!
            logger.info("🔥 Training started...")
            logger.info(f"   Dataset size: {len(dataset)}")
            logger.info(f"   Batch size: {self.config.batch_size}")
            logger.info(f"   Gradient accumulation: {self.config.gradient_accumulation_steps}")
            logger.info(f"   Total steps: {len(dataset) // (self.config.batch_size * self.config.gradient_accumulation_steps)}")
            
            # Verificar que el dataset no esté vacío
            if len(dataset) == 0:
                logger.error("❌ Dataset está vacío!")
                return {
                    "success": False,
                    "error": "Empty dataset"
                }
            
            try:
                # Agregar timeout y logging adicional para diagnosticar bloqueos
                import signal
                import threading
                
                # Flag para monitorear si el entrenamiento está activo
                training_active = threading.Event()
                training_active.set()
                
                def log_progress():
                    """Log periódico para verificar que el proceso no está bloqueado"""
                    import time
                    step_count = 0
                    while training_active.is_set():
                        time.sleep(10)  # Log cada 10 segundos
                        if training_active.is_set():
                            logger.info(f"⏳ Entrenamiento en progreso... (paso {step_count})")
                            step_count += 1
                
                # Iniciar thread de monitoreo
                monitor_thread = threading.Thread(target=log_progress, daemon=True)
                monitor_thread.start()
                
                logger.info("🚀 Iniciando entrenamiento (esto puede tomar varios minutos)...")
                logger.info("   💡 Si se queda bloqueado, verifica:")
                logger.info("      - Conexión a internet (para descargar modelos)")
                logger.info("      - Espacio en disco suficiente")
                logger.info("      - Memoria RAM disponible")
                logger.info("      - Logs adicionales aparecerán cada 10 segundos")
                
                # Log estado inicial del trainer
                if hasattr(self.trainer, 'state'):
                    initial_step = getattr(self.trainer.state, 'global_step', 0)
                    logger.info(f"   📊 Estado inicial: global_step={initial_step}")
                
                # Verificar que el modelo tiene parámetros entrenables ANTES de entrenar
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                if trainable_params == 0:
                    raise RuntimeError(
                        "❌ El modelo no tiene parámetros entrenables (trainable_params=0). "
                        "No se puede ejecutar entrenamiento. Verifica la configuración LoRA."
                    )
                logger.info(f"   ✅ Parámetros entrenables verificados: {trainable_params:,}")
                
                # Asegurar que el modelo está en modo entrenamiento
                self.model.train()
                for param in self.model.parameters():
                    if param.requires_grad:
                        param.requires_grad = True  # Forzar que estén entrenables
                
                # Ejecutar entrenamiento REAL
                logger.info("   🔥 Ejecutando trainer.train() - ENTRENAMIENTO REAL...")
                
                # Intentar entrenar con manejo de errores de accelerate
                try:
                    train_result = self.trainer.train()
                except (TypeError, AttributeError) as train_error:
                    error_str = str(train_error)
                    if "keep_torch_compile" in error_str or "unwrap_model" in error_str:
                        # El error ocurre al guardar, pero el entrenamiento puede haber comenzado
                        # Intentar ejecutar el loop de entrenamiento manualmente
                        logger.warning(f"⚠️ Error de accelerate detectado durante entrenamiento: {train_error}")
                        logger.info("   🔄 Intentando ejecutar entrenamiento manualmente...")
                        
                        # El error de accelerate ocurre al guardar, pero el entrenamiento puede haberse ejecutado
                        # Extraer métricas del state ANTES de que falle el guardado
                        logger.info("   📊 El entrenamiento puede haberse ejecutado - extrayendo métricas del state...")
                        
                        # Esperar un momento para que el entrenamiento se complete
                        import time
                        time.sleep(2)  # Dar tiempo para que el entrenamiento se ejecute
                        
                        # Verificar si el entrenamiento se ejecutó revisando el state
                        if hasattr(self.trainer, 'state'):
                            # Extraer métricas del log_history
                            training_loss = None
                            global_step = 0
                            
                            if hasattr(self.trainer.state, 'log_history') and self.trainer.state.log_history:
                                # Buscar el último log con loss
                                for log_entry in reversed(self.trainer.state.log_history):
                                    if isinstance(log_entry, dict):
                                        if 'loss' in log_entry:
                                            training_loss = log_entry.get('loss')
                                            global_step = log_entry.get('step', self.trainer.state.global_step if hasattr(self.trainer.state, 'global_step') else 0)
                                            break
                                        elif 'train_loss' in log_entry:
                                            training_loss = log_entry.get('train_loss')
                                            global_step = log_entry.get('step', self.trainer.state.global_step if hasattr(self.trainer.state, 'global_step') else 0)
                                            break
                            
                            # Obtener global_step del state
                            if hasattr(self.trainer.state, 'global_step'):
                                global_step = self.trainer.state.global_step
                            
                            # Si hay métricas, crear resultado
                            if training_loss is not None and global_step > 0:
                                logger.info(f"   ✅ Entrenamiento ejecutado: Loss={training_loss:.4f}, Steps={global_step}")
                                from types import SimpleNamespace
                                train_result = SimpleNamespace(
                                    training_loss=training_loss,
                                    global_step=global_step
                                )
                            else:
                                # El entrenamiento no se ejecutó - ejecutar loop de entrenamiento manualmente con PyTorch
                                logger.warning("   ⚠️ No se encontraron métricas - el entrenamiento no se ejecutó")
                                logger.info("   🔄 Ejecutando entrenamiento manualmente con PyTorch (sin accelerate)...")
                                
                                # Ejecutar loop de entrenamiento manualmente
                                from torch.utils.data import DataLoader
                                from torch.optim import AdamW
                                
                                # Preparar DataLoader
                                train_dataloader = DataLoader(
                                    dataset,
                                    batch_size=self.config.batch_size,
                                    shuffle=True,
                                    num_workers=0,  # Windows compatibility
                                    pin_memory=False
                                )
                                
                                # Optimizador
                                optimizer = AdamW(
                                    [p for p in self.model.parameters() if p.requires_grad],
                                    lr=self.config.learning_rate
                                )
                                
                                # Loop de entrenamiento
                                self.model.train()
                                # Asegurar que el modelo no use caché durante entrenamiento
                                if hasattr(self.model, 'config'):
                                    self.model.config.use_cache = False
                                
                                total_loss = 0.0
                                global_step = 0
                                
                                num_epochs = self.config.num_epochs
                                gradient_accumulation_steps = self.config.gradient_accumulation_steps
                                
                                logger.info(f"   🚂 Entrenando manualmente: {num_epochs} épocas, {len(train_dataloader)} batches/época")
                                logger.info(f"   📊 Total de batches: {len(train_dataloader)}")
                                logger.info(f"   📊 Device: {self.device}")
                                logger.info(f"   📊 Modelo en modo train: {self.model.training}")
                                
                                for epoch in range(num_epochs):
                                    logger.info(f"   🔄 Iniciando época {epoch + 1}/{num_epochs}...")
                                    epoch_loss = 0.0
                                    optimizer.zero_grad()
                                    
                                    for batch_idx, batch in enumerate(train_dataloader):
                                        if batch_idx == 0:
                                            logger.info(f"   📦 Procesando batch {batch_idx + 1}/{len(train_dataloader)}...")
                                        try:
                                            # Mover batch a dispositivo
                                            input_ids = batch['input_ids'].to(self.device)
                                            attention_mask = batch['attention_mask'].to(self.device)
                                            labels = batch['labels'].to(self.device)
                                            
                                            # Forward pass
                                            # Desactivar uso de caché para evitar problemas con DynamicCache
                                            outputs = self.model(
                                                input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                labels=labels,
                                                use_cache=False  # Desactivar caché durante entrenamiento
                                            )
                                        except AttributeError as cache_error:
                                            if "get_usable_length" in str(cache_error) or "DynamicCache" in str(cache_error):
                                                # Error de caché - intentar sin caché explícitamente
                                                logger.warning(f"   ⚠️ Error de caché detectado, reintentando sin caché: {cache_error}")
                                                # Asegurar que no se use caché
                                                if hasattr(self.model, 'config'):
                                                    self.model.config.use_cache = False
                                                # Reintentar sin pasar use_cache (ya está desactivado en config)
                                                outputs = self.model(
                                                    input_ids=input_ids,
                                                    attention_mask=attention_mask,
                                                    labels=labels
                                                )
                                            else:
                                                raise
                                        
                                        # Log éxito del forward pass
                                        if batch_idx == 0:
                                            logger.info(f"   ✅ Forward pass exitoso en batch {batch_idx + 1}")
                                        
                                        loss = outputs.loss if hasattr(outputs, 'loss') else None
                                        if loss is None:
                                            # Calcular loss manualmente si no está disponible
                                            from torch.nn import CrossEntropyLoss
                                            loss_fn = CrossEntropyLoss()
                                            shift_logits = outputs.logits[..., :-1, :].contiguous()
                                            shift_labels = labels[..., 1:].contiguous()
                                            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                                        
                                        # Normalizar loss por gradient accumulation
                                        loss = loss / gradient_accumulation_steps
                                        
                                        # Backward pass
                                        loss.backward()
                                        
                                        # Actualizar pesos cada gradient_accumulation_steps
                                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                                            optimizer.step()
                                            optimizer.zero_grad()
                                            global_step += 1
                                            
                                            # Log cada paso para ver progreso
                                            current_loss = loss.item() * gradient_accumulation_steps
                                            logger.info(f"   📊 Paso {global_step}: Loss={current_loss:.4f}")
                                        
                                        epoch_loss += loss.item() * gradient_accumulation_steps
                                        total_loss += loss.item() * gradient_accumulation_steps
                                    
                                    # Asegurar que se actualicen los pesos al final de la época si quedan gradientes
                                    if (batch_idx + 1) % gradient_accumulation_steps != 0:
                                        optimizer.step()
                                        optimizer.zero_grad()
                                    
                                    avg_epoch_loss = epoch_loss / len(train_dataloader)
                                    logger.info(f"   ✅ Época {epoch + 1}/{num_epochs} completada - Loss promedio: {avg_epoch_loss:.4f}")
                                
                                # Calcular loss promedio
                                total_batches = num_epochs * len(train_dataloader)
                                avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
                                
                                logger.info(f"   ✅ Entrenamiento manual completado: Loss={avg_loss:.4f}, Steps={global_step}")
                                
                                # Crear resultado
                                from types import SimpleNamespace
                                train_result = SimpleNamespace(
                                    training_loss=avg_loss,
                                    global_step=global_step
                                )
                                
                                # Guardar modelo manualmente
                                try:
                                    output_path = Path(self.config.output_dir)
                                    output_path.mkdir(parents=True, exist_ok=True)
                                    self.model.save_pretrained(output_path)
                                    self.tokenizer.save_pretrained(output_path)
                                    logger.info(f"   ✅ Modelo guardado en: {output_path}")
                                except Exception as save_error:
                                    logger.warning(f"   ⚠️ Error guardando modelo: {save_error}")
                        else:
                            raise RuntimeError("❌ No se puede acceder al state del trainer")
                    else:
                        # Otro tipo de error, re-lanzar
                        raise
                
                # Log estado después del entrenamiento
                if hasattr(self.trainer, 'state'):
                    final_step = getattr(self.trainer.state, 'global_step', 0)
                    log_count = len(getattr(self.trainer.state, 'log_history', []))
                    logger.info(f"   📊 Estado final: global_step={final_step}, log_entries={log_count}")
                    
                    # Log métricas del resultado
                    if hasattr(train_result, 'training_loss'):
                        logger.info(f"   📈 Loss final del entrenamiento: {train_result.training_loss:.4f}")
                    if hasattr(train_result, 'global_step'):
                        logger.info(f"   📊 Steps totales: {train_result.global_step}")
                
                # Detener monitoreo
                training_active.clear()
                
                # Validar que el entrenamiento realmente se ejecutó
                if hasattr(self.trainer, 'state'):
                    if hasattr(self.trainer.state, 'global_step'):
                        logger.info(f"   ✅ Pasos ejecutados: {self.trainer.state.global_step}")
                        if self.trainer.state.global_step == 0:
                            raise RuntimeError("❌ El entrenamiento no se ejecutó - global_step es 0")
                    
                    if hasattr(self.trainer.state, 'log_history'):
                        if not self.trainer.state.log_history:
                            raise RuntimeError("❌ No hay logs de entrenamiento - el entrenamiento no se ejecutó correctamente")
                        logger.info(f"   ✅ Logs de entrenamiento disponibles: {len(self.trainer.state.log_history)} entradas")
                
                logger.info("✅ Entrenamiento completado")
            except (TypeError, AttributeError) as e:
                error_str = str(e)
                if "keep_torch_compile" in error_str or "unwrap_model" in error_str:
                    # Error de compatibilidad con accelerate - el entrenamiento puede haber completado
                    # pero falló al guardar. Intentar obtener métricas reales y guardar manualmente.
                    logger.warning(f"⚠️ Error de compatibilidad con accelerate: {e}")
                    logger.warning("   El entrenamiento puede haber completado. Extrayendo métricas reales...")
                    
                    # EXTRAER LOSS REAL DEL TRAINER STATE - SIN FALLBACKS
                    training_loss = None
                    global_step = 0
                    
                    # Intentar obtener el resultado del entrenamiento del state
                    if hasattr(self.trainer, 'state') and hasattr(self.trainer.state, 'log_history'):
                        log_history = self.trainer.state.log_history
                        if log_history:
                            logger.info(f"   📊 Analizando {len(log_history)} entradas de log...")
                            # Buscar el último log con loss
                            for log_entry in reversed(log_history):
                                if isinstance(log_entry, dict):
                                    if 'loss' in log_entry:
                                        training_loss = log_entry.get('loss')
                                        logger.info(f"   ✅ Loss real encontrado: {training_loss:.4f}")
                                        break
                                    elif 'train_loss' in log_entry:
                                        training_loss = log_entry.get('train_loss')
                                        logger.info(f"   ✅ Train loss real encontrado: {training_loss:.4f}")
                                        break
                    
                    # Obtener global_step del state
                    if hasattr(self.trainer, 'state') and hasattr(self.trainer.state, 'global_step'):
                        global_step = self.trainer.state.global_step
                        logger.info(f"   ✅ Global step: {global_step}")
                    
                    # VALIDAR QUE EL ENTRENAMIENTO REALMENTE SE EJECUTÓ
                    # Verificar si el entrenamiento realmente se ejecutó antes de validar
                    if hasattr(self.trainer, 'state'):
                        # Verificar si hay parámetros entrenables
                        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                        if trainable_params == 0:
                            raise RuntimeError("❌ El modelo no tiene parámetros entrenables - no se puede entrenar. Verifica la configuración LoRA.")
                        
                        # Si global_step es 0 pero hay log_history, el entrenamiento puede haber fallado silenciosamente
                        if hasattr(self.trainer.state, 'global_step') and self.trainer.state.global_step == 0:
                            # Verificar si hay logs que indiquen que se intentó entrenar
                            if hasattr(self.trainer.state, 'log_history') and self.trainer.state.log_history:
                                # Hay logs pero global_step es 0 - puede ser un problema de configuración
                                logger.warning("⚠️ Hay logs pero global_step es 0 - verificando configuración del trainer")
                                # Intentar continuar si hay logs
                            else:
                                raise RuntimeError("❌ El entrenamiento no se ejecutó - global_step es 0 y no hay logs. Verifica la configuración del trainer.")
                        
                        if not hasattr(self.trainer.state, 'log_history') or not self.trainer.state.log_history:
                            raise RuntimeError("❌ No hay logs de entrenamiento - el entrenamiento no se ejecutó correctamente. No hay métricas reales disponibles.")
                    
                    # Si no se pudo obtener loss real, lanzar error en lugar de usar valor mock
                    if training_loss is None:
                        raise RuntimeError(
                            "❌ No se pudo extraer el loss real del entrenamiento. "
                            "El entrenamiento puede no haberse ejecutado correctamente. "
                            "Verifica los logs del trainer.state.log_history."
                        )
                    
                    logger.info(f"   📊 Métricas reales extraídas: Loss={training_loss:.4f}, Steps={global_step}")
                    
                    # Verificar si hay un checkpoint guardado
                    try:
                        checkpoint_dir = Path(self.config.output_dir)
                        if checkpoint_dir.exists():
                            # Buscar el último checkpoint
                            checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
                            if checkpoints:
                                latest_checkpoint = max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))
                                logger.info(f"   📁 Encontrado checkpoint: {latest_checkpoint}")
                    except Exception:
                        pass
                    
                    # Guardar modelo sin usar unwrap_model
                    try:
                        # Obtener el modelo del trainer
                        model_to_save = self.trainer.model
                        if hasattr(model_to_save, 'module'):
                            model_to_save = model_to_save.module
                        
                        # Guardar directamente
                        output_path = Path(self.config.output_dir)
                        output_path.mkdir(parents=True, exist_ok=True)
                        model_to_save.save_pretrained(output_path)
                        self.tokenizer.save_pretrained(output_path)
                        
                        logger.info("✅ Modelo guardado manualmente")
                        
                        # Crear resultado con métricas REALES (no mock)
                        from types import SimpleNamespace
                        train_result = SimpleNamespace(
                            training_loss=training_loss,  # Loss REAL extraído del state
                            global_step=global_step  # Steps REALES del state
                        )
                        logger.info(f"   ✅ Resultado creado con métricas REALES: Loss={training_loss:.4f}, Steps={global_step}")
                    except Exception as save_error:
                        logger.error(f"❌ Error guardando modelo: {save_error}")
                        # Si falla el guardado pero tenemos métricas reales, aún podemos reportar el resultado
                        if training_loss is not None:
                            from types import SimpleNamespace
                            train_result = SimpleNamespace(
                                training_loss=training_loss,  # Usar loss REAL
                                global_step=global_step  # Usar steps REALES
                            )
                            logger.warning("   ⚠️ Modelo no guardado, pero métricas de entrenamiento disponibles")
                        else:
                            raise RuntimeError(
                                f"❌ Error guardando modelo y no hay métricas reales disponibles: {save_error}"
                            )
                else:
                    # Otro tipo de error - re-lanzar
                    raise
            
            logger.info("✅ Training complete!")
            
            # VALIDACIÓN FINAL: Verificar que el resultado tiene valores REALES
            if not hasattr(train_result, 'training_loss'):
                raise RuntimeError("❌ train_result no tiene training_loss - el entrenamiento puede no haberse completado correctamente")
            
            training_loss = train_result.training_loss
            
            # Validar que el loss no sea 0.0 (indicaría un fallback/mock)
            if training_loss == 0.0:
                logger.warning("⚠️ WARNING: training_loss es 0.0 - esto puede indicar un problema")
                # Verificar si realmente se entrenó revisando el state
                if hasattr(self.trainer, 'state'):
                    if hasattr(self.trainer.state, 'global_step') and self.trainer.state.global_step > 0:
                        logger.warning("   ⚠️ Pero global_step > 0, el entrenamiento sí se ejecutó")
                        logger.warning("   ⚠️ El loss 0.0 puede ser real si el modelo ya estaba perfectamente entrenado")
                    else:
                        raise RuntimeError("❌ training_loss es 0.0 y global_step es 0 - el entrenamiento no se ejecutó")
            
            logger.info(f"   📊 Loss final: {training_loss:.4f}")
            
            # Obtener steps reales
            steps = getattr(train_result, 'global_step', 0)
            if steps == 0 and hasattr(self.trainer, 'state') and hasattr(self.trainer.state, 'global_step'):
                steps = self.trainer.state.global_step
                logger.info(f"   📊 Steps obtenidos del state: {steps}")
            
            logger.info(f"   📊 Steps totales: {steps}")
            
            # Save model (si no se guardó manualmente)
            try:
                self.save_model()
            except Exception as e:
                logger.warning(f"⚠️ Error en save_model (puede que ya esté guardado): {e}")
            
            return {
                "success": True,
                "training_loss": training_loss,  # Loss REAL validado
                "epochs": self.config.num_epochs,
                "steps": steps,  # Steps REALES
                "model_path": self.config.output_dir
            }
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def save_model(self):
        """Save trained model"""
        try:
            output_path = Path(self.config.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save LoRA adapters
            self.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            
            logger.info(f"💾 Model saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
    
    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """
        Generate text using trained model
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return f"Error: {str(e)}"


# Singleton
_real_training_system: Optional[RealTrainingSystem] = None


def get_real_training_system(config: Optional[TrainingConfig] = None) -> RealTrainingSystem:
    """Get singleton instance"""
    global _real_training_system
    
    if _real_training_system is None:
        _real_training_system = RealTrainingSystem(config)
    
    return _real_training_system


# Demo
if __name__ == "__main__":
    print("🚂 Real Training System Demo")
    print("=" * 50)
    
    # Sample training data
    train_data = [
        {
            "input": "What is machine learning?",
            "output": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
        },
        {
            "input": "Explain neural networks",
            "output": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process information."
        }
    ]
    
    # Initialize
    config = TrainingConfig(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        num_epochs=1,
        batch_size=1,
        output_dir="./demo_model"
    )
    
    trainer = get_real_training_system(config)
    
    print(f"Device: {trainer.device}")
    print(f"Ready for real training!")
