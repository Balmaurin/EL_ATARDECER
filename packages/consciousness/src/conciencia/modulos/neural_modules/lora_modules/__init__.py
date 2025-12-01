"""
LoRA Modules for LLM Modulation
==============================

Módulos LoRA para modular el comportamiento del LLM según estados del cerebro.
"""

from .lora_vmpfc import LoRAVMPFC
from .lora_ofc import LoRAOFC
from .lora_ras import LoRARAS
from .lora_metacog import LoRAMetaCog

__all__ = ["LoRAVMPFC", "LoRAOFC", "LoRARAS", "LoRAMetaCog"]
