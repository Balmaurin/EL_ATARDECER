#!/usr/bin/env python3
"""
Train Real ML Model with Hack Memori Responses
====================================================

Uses real_training_system.py to train Phi-3-mini-4k-instruct with LoRA
on Hack Memori response data (accepted_for_training=true)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_hack_memori_training_data(responses_dir: str, accepted_only: bool = True) -> List[Dict[str, str]]:
    """
    Load Hack Memori responses for training

    Args:
        responses_dir: Path to responses directory
        accepted_only: Only load responses marked accepted_for_training=True

    Returns:
        List of training examples with 'input' and 'output' keys
    """
    responses_path = Path(responses_dir)
    training_data = []

    if not responses_path.exists():
        raise FileNotFoundError(f"Responses directory not found: {responses_path}")

    json_files = list(responses_path.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON response files")

    loaded_count = 0
    accepted_count = 0

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            loaded_count += 1

            # Check if accepted for training
            if accepted_only:
                accepted = data.get('accepted_for_training', False)
                if not accepted:
                    continue

            accepted_count += 1

            # Extract prompt and response
            prompt = data.get('prompt', '')
            response = data.get('response', '')

            if not prompt or not response:
                logger.warning(f"Skipping {json_file}: missing prompt or response")
                continue

            # Format for instruction-following training
            training_example = {
                'input': prompt.strip(),
                'output': response.strip()
            }

            training_data.append(training_example)

        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            continue

    logger.info(f"Loaded {loaded_count} total responses, {accepted_count} accepted")
    logger.info(f"Created {len(training_data)} training examples")
    return training_data


def filter_by_model(data: List[Dict[str, str]], model_ids: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Filter training data by model ID

    Args:
        data: Training data with model metadata
        model_ids: List of model IDs to include (None for all)

    Returns:
        Filtered training data
    """
    if model_ids is None:
        return data

    # Note: We don't have model_id in the training examples currently
    # This is a placeholder for future enhancement
    return data


def create_training_batches(training_data: List[Dict[str, str]], batch_size: int = 50) -> List[List[Dict[str, str]]]:
    """
    Create training batches to avoid memory issues

    Args:
        training_data: Full training dataset
        batch_size: Size of each batch

    Returns:
        List of training batches
    """
    batches = []
    for i in range(0, len(training_data), batch_size):
        batch = training_data[i:i + batch_size]
        batches.append(batch)

    logger.info(f"Created {len(batches)} training batches (batch_size={batch_size})")
    return batches


def train_on_hack_memori(
    responses_dir: str = "data/hack_memori/responses",
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    batch_size: int = 50,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    max_batches: Optional[int] = None
):
    """
    Train model on Hack Memori responses

    Args:
        responses_dir: Path to responses directory
        model_name: Model to train
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        max_batches: Maximum number of batches to train (None for all)
    """

    logger.info("🚀 Starting Hack Memori Training")
    logger.info("="*50)

    # Import here to avoid import errors - add the training directory to path
    training_dir = Path(__file__).parent.parent.parent / "packages" / "sheily_core" / "src" / "sheily_core" / "training"
    sys.path.insert(0, str(training_dir))
    from real_training_system import get_real_training_system, TrainingConfig

    # Load training data
    logger.info("📁 Loading Hack Memori responses...")
    training_data = load_hack_memori_training_data(responses_dir)

    if not training_data:
        logger.error("❌ No training data found!")
        return False

    logger.info(f"✅ Loaded {len(training_data)} training examples")

    # Create batches
    batches = create_training_batches(training_data, batch_size)
    if max_batches:
        batches = batches[:max_batches]
        logger.info(f"📊 Training on first {max_batches} batches ({len(batches) * batch_size} examples)")

    # Configure training
    config = TrainingConfig(
        model_name=model_name,
        output_dir="./trained_models/hack_memori_sheily_v1",
        num_epochs=num_epochs,
        batch_size=min(batch_size, 4),  # Small batch size for safety
        learning_rate=learning_rate,
        max_length=1024  # Longer for complex responses
    )

    # Initialize trainer
    trainer = get_real_training_system(config)

    total_examples = 0
    successful_batches = 0

    # Train in batches or all at once based on size
    if len(training_data) <= 100:
        # Small dataset - train all at once
        logger.info("🎯 Training on complete dataset")
        result = trainer.train(training_data)
        if result.get('success'):
            successful_batches += 1
            total_examples += len(training_data)
    else:
        # Large dataset - train in batches
        logger.info("🎯 Training in batches")
        for i, batch in enumerate(batches):
            logger.info(f"🏋️ Training batch {i+1}/{len(batches)} ({len(batch)} examples)")

            try:
                result = trainer.train(batch)
                if result.get('success'):
                    successful_batches += 1
                    total_examples += len(batch)
                    logger.info(f"✅ Batch {i+1} completed - Loss: {result.get('training_loss', 'N/A'):.4f}")
                else:
                    logger.error(f"❌ Batch {i+1} failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"❌ Batch {i+1} exception: {e}")
                continue

    # Summary
    logger.info("\n" + "="*50)
    logger.info("🎉 Training Complete!")
    logger.info(f"   Successful batches: {successful_batches}")
    logger.info(f"   Total examples trained: {total_examples}")
    logger.info(f"   Model saved to: {config.output_dir}")
    logger.info("="*50)

    return successful_batches > 0


def test_trained_model(model_path: str = "./trained_models/hack_memori_sheily_v1"):
    """
    Test the trained model with Hack Memori-style questions

    Args:
        model_path: Path to trained model
    """
    logger.info("🧪 Testing trained model")

    # Import training system
    training_dir = Path(__file__).parent.parent.parent / "packages" / "sheily_core" / "src" / "sheily_core" / "training"
    sys.path.insert(0, str(training_dir))
    from real_training_system import get_real_training_system, TrainingConfig

    # Load trained model
    config = TrainingConfig(output_dir=model_path)
    trainer = get_real_training_system(config)

    # Load the saved model
    try:
        trainer.model = None  # Reset to load from path
        # Note: We'd need to modify real_training_system.py to support loading from path
        logger.info("Model loading not yet implemented in test function")
        return
    except Exception as e:
        logger.warning(f"Could not load trained model: {e}")
        return

    # Test prompts
    test_prompts = [
        "Explain consciousness from a neuroscience perspective.",
        "How would you implement memory systems in distributed AI?",
        "What are the key principles of multimodal learning?"
    ]

    logger.info("🔮 Generating responses:")
    for prompt in test_prompts:
        try:
            response = trainer.generate(prompt, max_new_tokens=200)
            logger.info(f"\nPrompt: {prompt[:100]}...")
            logger.info(f"Response: {response[:200]}...")
            logger.info("-"*50)
        except Exception as e:
            logger.error(f"Error generating for prompt '{prompt[:50]}...': {e}")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Train model on Hack Memori responses")
    parser.add_argument("--responses-dir", default="data/hack_memori/responses", help="Path to responses directory")
    parser.add_argument("--model-name", default="microsoft/Phi-3-mini-4k-instruct", help="Model to train")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for training")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-batches", type=int, help="Maximum batches to train")
    parser.add_argument("--test-only", action="store_true", help="Only run testing, not training")

    args = parser.parse_args()

    if args.test_only:
        test_trained_model()
    else:
        success = train_on_hack_memori(
            responses_dir=args.responses_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            max_batches=args.max_batches
        )

        if success:
            print("🎉 Training completed successfully!")
            print("💡 To test the trained model, run with --test-only flag")
        else:
            print("❌ Training failed!")
            sys.exit(1)
