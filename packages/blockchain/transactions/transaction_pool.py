#!/usr/bin/env python3
"""
SHEILYS Transaction Pool - Pool de transacciones pendiente para SHEILYS Blockchain

Gestiona las transacciones pendientes antes de ser incluidas en bloques.
Implementa prioridades, límites de memoria, y políticas de limpieza.
"""

import heapq
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .sheilys_blockchain import SHEILYSTransaction

# Configurar logging
logger = logging.getLogger(__name__)


class TransactionPriority(Enum):
    """Prioridades de transacciones en el pool"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass(order=True)
class PrioritizedTransaction:
    """Transacción con prioridad para heap management"""

    priority: int
    timestamp: float
    transaction: SHEILYSTransaction


class TransactionPool:
    """
    Pool de transacciones SHEILYS - Gestiona transacciones pendientes

    Funcionalidades:
    - Gestión de prioridades de transacciones
    - Límites de memoria y limpieza automática
    - Estadísticas de pool
    - Políticas de reemplazo inteligente
    """

    def __init__(self, max_size: int = 10000, max_age_seconds: int = 300):
        """
        Inicializar transaction pool

        Args:
            max_size: Máximo número de transacciones en pool
            max_age_seconds: Máximo tiempo de vida de transacciones (5 minutos)
        """
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds

        # Pool principal - heap de transacciones priorizadas
        self.transaction_heap: List[PrioritizedTransaction] = []

        # Index rápido para lookup por transaction_id
        self.transaction_index: Dict[str, PrioritizedTransaction] = {}

        # Estadísticas del pool
        self.stats = {
            "added_transactions": 0,
            "removed_transactions": 0,
            "expired_transactions": 0,
            "replaced_transactions": 0,
            "current_size": 0,
            "max_size_reached": 0,
            "avg_priority": 0.0,
            "oldest_transaction": 0.0,
            "newest_transaction": 0.0,
        }

        # Políticas de prioridad por tipo de transacción
        self.priority_policies = {
            "transfer": TransactionPriority.NORMAL,
            "stake": TransactionPriority.HIGH,
            "unstake": TransactionPriority.HIGH,
            "reward": TransactionPriority.LOW,
            "nft_mint": TransactionPriority.NORMAL,
            "nft_transfer": TransactionPriority.NORMAL,
            "gamification": TransactionPriority.LOW,
        }

    def add_transaction(self, transaction: SHEILYSTransaction) -> bool:
        """
        Agregar transacción al pool

        Args:
            transaction: Transacción a agregar

        Returns:
            bool: True si fue agregada exitosamente
        """
        try:
            # Verificar si ya existe
            if transaction.transaction_id in self.transaction_index:
                return False

            # Limpiar transacciones expiradas antes de agregar
            self._clean_expired_transactions()

            # Verificar límites de pool
            if self._is_pool_full():
                # Intentar reemplazar transacción de baja prioridad
                if not self._try_replace_low_priority(transaction):
                    self.stats["max_size_reached"] += 1
                    return False

            # Calcular prioridad
            priority = self._calculate_priority(transaction)

            # Crear entrada priorizada
            prioritized_tx = PrioritizedTransaction(
                priority=priority.value,
                timestamp=transaction.timestamp,
                transaction=transaction,
            )

            # Agregar al heap y al índice
            heapq.heappush(self.transaction_heap, prioritized_tx)
            self.transaction_index[transaction.transaction_id] = prioritized_tx

            # Actualizar estadísticas
            self.stats["added_transactions"] += 1
            self.stats["current_size"] += 1
            self._update_stats()

            return True

        except Exception as e:
            logger.error(f"Error agregando transacción al pool: {e}", exc_info=True)
            return False

    def _calculate_priority(
        self, transaction: SHEILYSTransaction
    ) -> TransactionPriority:
        """Calcular prioridad de la transacción basada en su tipo"""
        tx_type = transaction.transaction_type.value.lower()

        # Buscar política por tipo
        if tx_type in self.priority_policies:
            return self.priority_policies[tx_type]

        # Default priority
        return TransactionPriority.NORMAL

    def _is_pool_full(self) -> bool:
        """Verificar si el pool está lleno"""
        return len(self.transaction_heap) >= self.max_size

    def _try_replace_low_priority(self, new_transaction: SHEILYSTransaction) -> bool:
        """
        Intentar reemplazar transacción de baja prioridad

        Returns:
            bool: True si se pudo reemplazar
        """
        # Buscar la transacción de más baja prioridad
        lowest_priority_tx = None
        lowest_priority = TransactionPriority.URGENT.value + 1

        for tx_entry in self.transaction_heap:
            if tx_entry.priority < lowest_priority:
                lowest_priority = tx_entry.priority
                lowest_priority_tx = tx_entry

        # Si encontramos una de baja prioridad, reemplazar
        if lowest_priority_tx and lowest_priority <= TransactionPriority.LOW.value:
            self.remove_transaction(lowest_priority_tx.transaction.transaction_id)
            self.stats["replaced_transactions"] += 1
            return True

        return False

    def remove_transaction(self, transaction_id: str) -> bool:
        """
        Remover transacción del pool

        Args:
            transaction_id: ID de la transacción a remover

        Returns:
            bool: True si fue removida exitosamente
        """
        try:
            if transaction_id not in self.transaction_index:
                return False

            # Remover del índice
            removed_tx = self.transaction_index.pop(transaction_id)

            # Remover del heap (más complicado - recrear heap)
            self.transaction_heap = [
                tx
                for tx in self.transaction_heap
                if tx.transaction.transaction_id != transaction_id
            ]
            heapq.heapify(self.transaction_heap)  # Re-heapify

            # Actualizar estadísticas
            self.stats["removed_transactions"] += 1
            self.stats["current_size"] -= 1
            self._update_stats()

            return True

        except Exception as e:
            logger.error(f"Error removiendo transacción del pool: {e}", exc_info=True)
            return False

    def get_transactions_for_block(
        self, max_count: int = 1000
    ) -> List[SHEILYSTransaction]:
        """
        Obtener transacciones para incluir en el próximo bloque

        Args:
            max_count: Máximo número de transacciones a retornar

        Returns:
            List[SHEILYSTransaction]: Lista de transacciones ordenadas por prioridad
        """
        try:
            # Limpiar transacciones expiradas
            self._clean_expired_transactions()

            # Obtener las transacciones de más alta prioridad
            # heapq es min-heap, pero nuestros prioridades están invertidas
            # donde mayor número = mayor prioridad
            transactions = []

            # Extraer y volver a ordenar por prioridad descendente
            temp_transactions = []
            while self.transaction_heap and len(temp_transactions) < max_count:
                tx_entry = heapq.heappop(self.transaction_heap)
                temp_transactions.append(tx_entry)

            # Ordenar por prioridad descendente (más alta primero)
            temp_transactions.sort(key=lambda x: x.priority, reverse=True)
            transactions = [
                tx_entry.transaction for tx_entry in temp_transactions[:max_count]
            ]

            # Regresar las demás al heap
            remaining = temp_transactions[max_count:]
            for tx_entry in remaining:
                heapq.heappush(self.transaction_heap, tx_entry)

            return transactions

        except Exception as e:
            logger.error(f"Error obteniendo transacciones para bloque: {e}", exc_info=True)
            return []

    def _clean_expired_transactions(self):
        """Limpiar transacciones expiradas del pool"""
        current_time = time.time()
        expired_count = 0

        # Buscar transacciones expiradas
        expired_transactions = []
        for tx_entry in self.transaction_heap:
            if current_time - tx_entry.timestamp > self.max_age_seconds:
                expired_transactions.append(tx_entry.transaction.transaction_id)
                expired_count += 1

        # Remover transacciones expiradas
        for tx_id in expired_transactions:
            self.remove_transaction(tx_id)

        if expired_count > 0:
            self.stats["expired_transactions"] += expired_count

    def _update_stats(self):
        """Actualizar estadísticas del pool"""
        if not self.transaction_heap:
            self.stats["avg_priority"] = 0.0
            self.stats["oldest_transaction"] = 0.0
            self.stats["newest_transaction"] = 0.0
            return

        # Calcular prioridad promedio
        total_priority = sum(tx.priority for tx in self.transaction_heap)
        self.stats["avg_priority"] = total_priority / len(self.transaction_heap)

        # Encontrar timestamps más viejos y nuevos
        timestamps = [tx.timestamp for tx in self.transaction_heap]
        self.stats["oldest_transaction"] = min(timestamps)
        self.stats["newest_transaction"] = max(timestamps)

    def get_pool_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del transaction pool"""
        return {
            "current_size": len(self.transaction_heap),
            "max_size": self.max_size,
            "utilization_percentage": (len(self.transaction_heap) / self.max_size)
            * 100,
            "avg_priority": self.stats["avg_priority"],
            "oldest_transaction_age": (
                time.time() - self.stats["oldest_transaction"]
                if self.stats["oldest_transaction"] > 0
                else 0
            ),
            "newest_transaction_age": (
                time.time() - self.stats["newest_transaction"]
                if self.stats["newest_transaction"] > 0
                else 0
            ),
            "stats": self.stats.copy(),
        }

    def get_transaction(self, transaction_id: str) -> Optional[SHEILYSTransaction]:
        """
        Obtener transacción específica por ID

        Args:
            transaction_id: ID de la transacción

        Returns:
            SHEILYSTransaction: Transacción encontrada, o None
        """
        if transaction_id in self.transaction_index:
            return self.transaction_index[transaction_id].transaction
        return None

    def get_transactions_by_sender(self, sender: str) -> List[SHEILYSTransaction]:
        """
        Obtener transacciones de un remitente específico

        Args:
            sender: Dirección del remitente

        Returns:
            List[SHEILYSTransaction]: Lista de transacciones del remitente
        """
        return [
            tx_entry.transaction
            for tx_entry in self.transaction_heap
            if tx_entry.transaction.sender == sender
        ]

    def update_transaction_priority(
        self, transaction_id: str, new_priority: TransactionPriority
    ) -> bool:
        """
        Actualizar prioridad de una transacción

        Args:
            transaction_id: ID de la transacción
            new_priority: Nueva prioridad

        Returns:
            bool: True si fue actualizada exitosamente
        """
        try:
            if transaction_id not in self.transaction_index:
                return False

            tx_entry = self.transaction_index[transaction_id]

            # Remover del heap temporalmente
            temp_heap = [
                tx
                for tx in self.transaction_heap
                if tx.transaction.transaction_id != transaction_id
            ]
            heapq.heapify(temp_heap)

            # Actualizar prioridad
            tx_entry.priority = new_priority.value

            # Volver a agregar
            heapq.heappush(temp_heap, tx_entry)
            self.transaction_heap = temp_heap

            # Actualizar estadísticas
            self._update_stats()

            return True

        except Exception as e:
            logger.error(f"Error actualizando prioridad de transacción: {e}", exc_info=True)
            return False

    def clear_pool(self):
        """Limpiar completamente el transaction pool"""
        self.transaction_heap.clear()
        self.transaction_index.clear()
        self.stats["current_size"] = 0
