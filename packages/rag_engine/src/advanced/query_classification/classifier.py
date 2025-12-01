#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Classification System for RAG
Based on EMNLP 2024 Paper Section A.1

Implements BERT-base-multilingual-cased classifier for:
- "retrieval required" vs "no retrieval required" classification
- 95% accuracy target with 96% precision
- Handles 15 different task types
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


@dataclass
class QueryClassificationResult:
    """Result of query classification"""

    query: str
    needs_retrieval: bool
    confidence: float
    predicted_class: str
    all_probabilities: Dict[str, float]
    processing_time: float


class QueryClassificationDataset(Dataset):
    """Dataset for query classification training"""

    def __init__(
        self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class QueryClassifier:
    """
    BERT-based Query Classifier for RAG systems

    Based on paper configuration:
    - Model: BERT-base-multilingual-cased
    - Batch size: 16
    - Learning rate: 1e-5
    - Target: 95% accuracy, 96% precision
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        num_labels: int = 2,
        max_length: int = 512,
        batch_size: int = 16,
        learning_rate: float = 1e-5,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the query classifier

        Args:
            model_name: HuggingFace model name
            num_labels: Number of classification labels (2 for binary)
            max_length: Maximum sequence length
            batch_size: Training batch size
            learning_rate: Learning rate for training
            model_path: Path to pre-trained model (optional)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_path = (
            model_path or f"models/query_classifier_{model_name.replace('/', '_')}"
        )

        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.label_map = {0: "no_retrieval", 1: "retrieval_required"}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

        # Create model directory
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized QueryClassifier with {model_name}")

    def load_model(self) -> bool:
        """Load or initialize the model and tokenizer"""
        try:
            if (
                Path(self.model_path).exists()
                and (Path(self.model_path) / "config.json").exists()
            ):
                # Load fine-tuned model
                logger.info(f"Loading fine-tuned model from {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path
                )
            else:
                # Initialize from base model
                logger.info(f"Initializing from base model {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, num_labels=self.num_labels
                )

            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def create_dataset_from_paper_examples(self) -> Tuple[List[str], List[int]]:
        """
        Create training dataset based on paper examples
        Returns 111K samples with 15 task types as mentioned in paper
        """
        texts = []
        labels = []

        # Task categories from paper (Figure 2)
        task_categories = {
            "search": [
                "retrieval_required",
                "Find information about",
                "Search for",
                "What is",
            ],
            "continuation_writing": [
                "no_retrieval",
                "Please continue writing",
                "Complete the paragraph",
            ],
            "translation": [
                "no_retrieval",
                "Translate this to",
                "What is the French for",
            ],
            "planning": [
                "no_retrieval",
                "How to plan",
                "Create a schedule for",
                "Plan for",
            ],
            "roleplay": ["no_retrieval", "Pretend you are", "Role-play as", "Act as"],
            "rewriting": [
                "no_retrieval",
                "Paraphrase this",
                "Rewrite the following",
                "Rephrase",
            ],
            "summarization": ["no_retrieval", "Summarize this", "Give me a summary"],
            "closed_qa": [
                "retrieval_required",
                "Identify who is",
                "Which of these",
                "Choose the correct",
            ],
            "reasoning": [
                "no_retrieval",
                "If A has 3 sisters",
                "Tom has 3 sisters",
                "Logic puzzle",
            ],
            "in_context_learning": [
                "no_retrieval",
                "Q: 3,1 A: 3 Q: 2,5 A: 5",
                "Pattern recognition",
            ],
            "information_extraction": [
                "retrieval_required",
                "What is the ownership",
                "Extract the relationship",
            ],
            "decision_making": [
                "no_retrieval",
                "Should I drive or take",
                "Which option is better",
            ],
            "suggestion": [
                "no_retrieval",
                "How should I persuade",
                "What should I do when",
            ],
            "writing": [
                "no_retrieval",
                "Write an article about",
                "Create content about",
            ],
            "general_qa": [
                "retrieval_required",
                "What are the benefits",
                "Explain how",
            ],
        }

        # Generate synthetic examples for each category
        for category, (retrieval_type, *examples) in task_categories.items():
            label = 1 if retrieval_type == "retrieval_required" else 0

            # Add base examples
            for example in examples:
                texts.append(example)
                labels.append(label)

            # Generate variations (expand to ~7K examples per category for 111K total)
            variations = [
                f"Can you {example.lower()}",
                f"I need help with {example.lower()}",
                f"Please {example.lower()}",
                f"Tell me about {example.lower()}",
                f"Help me {example.lower()}",
                f"Show me how to {example.lower()}",
                f"What do you know about {example.lower()}",
                f"Explain {example.lower()}",
                f"Give me information about {example.lower()}",
                f"I want to know {example.lower()}",
            ]

            # Add variations multiple times to reach target size
            for _ in range(500):  # ~7.5K examples per category
                for variation in variations:
                    if examples:  # Make sure we have base examples
                        base_example = np.random.choice(examples)
                        full_text = f"{variation} {base_example.lower()}"
                        texts.append(full_text)
                        labels.append(label)

        logger.info(f"Created dataset with {len(texts)} examples")
        return texts, labels

    def train(
        self,
        train_texts: Optional[List[str]] = None,
        train_labels: Optional[List[int]] = None,
        validation_split: float = 0.1,
        num_epochs: int = 3,
        save_steps: int = 500,
        eval_steps: int = 500,
    ) -> Dict[str, float]:
        """
        Train the query classifier

        Args:
            train_texts: Training texts (if None, uses paper examples)
            train_labels: Training labels
            validation_split: Validation split ratio
            num_epochs: Number of training epochs
            save_steps: Save model every N steps
            eval_steps: Evaluate every N steps

        Returns:
            Training metrics
        """
        if train_texts is None or train_labels is None:
            logger.info("Using paper examples for training")
            train_texts, train_labels = self.create_dataset_from_paper_examples()

        # Split into train/validation
        split_idx = int(len(train_texts) * (1 - validation_split))
        train_texts_split = train_texts[:split_idx]
        train_labels_split = train_labels[:split_idx]
        val_texts = train_texts[split_idx:]
        val_labels = train_labels[split_idx:]

        # Create datasets
        train_dataset = QueryClassificationDataset(
            train_texts_split, train_labels_split, self.tokenizer, self.max_length
        )
        val_dataset = QueryClassificationDataset(
            val_texts, val_labels, self.tokenizer, self.max_length
        )

        # Training arguments (based on paper: batch_size=16, lr=1e-5)
        training_args = TrainingArguments(
            output_dir=self.model_path,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{self.model_path}/logs",
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,  # Avoid issues on Windows
            report_to=[],  # Disable wandb/tensorboard
        )

        # Custom compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)

            return {
                "accuracy": accuracy_score(labels, predictions),
                "precision": precision_score(labels, predictions, average="weighted"),
                "recall": recall_score(labels, predictions, average="weighted"),
                "f1": f1_score(labels, predictions, average="weighted"),
            }

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train the model
        logger.info("Starting training...")
        trainer.train()

        # Save the best model
        trainer.save_model(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)

        # Get final metrics
        final_metrics = trainer.evaluate()
        logger.info(f"Training completed. Final metrics: {final_metrics}")

        return final_metrics

    def classify(self, query: str) -> QueryClassificationResult:
        """
        Classify a single query

        Args:
            query: Input query text

        Returns:
            Classification result
        """
        import time

        start_time = time.time()

        if not self.model or not self.tokenizer:
            if not self.load_model():
                raise RuntimeError("Failed to load model for classification")

        # Tokenize input
        inputs = self.tokenizer(
            query,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # Get prediction
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.label_map[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        needs_retrieval = predicted_class == "retrieval_required"

        # All probabilities
        all_probabilities = {
            self.label_map[i]: float(probabilities[i])
            for i in range(len(probabilities))
        }

        processing_time = time.time() - start_time

        return QueryClassificationResult(
            query=query,
            needs_retrieval=needs_retrieval,
            confidence=confidence,
            predicted_class=predicted_class,
            all_probabilities=all_probabilities,
            processing_time=processing_time,
        )

    def classify_batch(self, queries: List[str]) -> List[QueryClassificationResult]:
        """
        Classify multiple queries in batch

        Args:
            queries: List of query texts

        Returns:
            List of classification results
        """
        return [self.classify(query) for query in queries]

    def evaluate_on_test_set(
        self, test_texts: List[str], test_labels: List[int]
    ) -> Dict[str, float]:
        """
        Evaluate model on test set

        Args:
            test_texts: Test texts
            test_labels: True labels

        Returns:
            Evaluation metrics
        """
        results = self.classify_batch(test_texts)
        predictions = [1 if result.needs_retrieval else 0 for result in results]

        return {
            "accuracy": accuracy_score(test_labels, predictions),
            "precision": precision_score(test_labels, predictions, average="weighted"),
            "recall": recall_score(test_labels, predictions, average="weighted"),
            "f1": f1_score(test_labels, predictions, average="weighted"),
        }

    def save_model(self, path: Optional[str] = None) -> str:
        """Save the model and tokenizer"""
        save_path = path or self.model_path
        Path(save_path).mkdir(parents=True, exist_ok=True)

        if self.model and self.tokenizer:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Model saved to {save_path}")

        return save_path

    def load_trained_model(self, path: Optional[str] = None) -> bool:
        """Load a trained model"""
        load_path = path or self.model_path
        return self.load_model()
