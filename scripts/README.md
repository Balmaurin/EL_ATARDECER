# Scripts Directory

This directory contains all the scripts for the EL-AMANECER project, organized by function.

## Directory Structure

### `core/`
Contains the main execution scripts for the system's core loops and evolution cycles.
- `execute_real_evolution_cycle.py`: The main entry point for the evolution cycle.

### `maintenance/`
Scripts for system maintenance, repair, and updates.
- `repair_corruption_v2.py`: Tool to fix file corruption (specifically the `]c]h]a]r` pattern).
- `update_containers.py`: Updates Docker containers.
- `fix_encoding.py`: Fixes file encoding issues.
- `repair_rag_file.py`: Specific repair for RAG files.

### `services/`
Scripts that run persistent services or applications.
- `serve_frontend.py`: Serves the frontend application.
- `sheily_chat.py`: Runs the chat interface.

### `testing/`
Scripts for testing and validating system components.
- `test_pegamento_completo.py`: Integration tests.
- `validate_dashboard.py`: Validates dashboard functionality.
- `validate_frontend_connection.py`: Checks frontend connectivity.
- `validate_trained_model.py`: Validates model performance.

### `training/`
Scripts related to model training and data generation.
- `real_functional_training.py`: Main training script.
- `generate_training_data.py`: Generates data for training.
- `simple_training.py`: Simplified training routine.
- `train_embeddings.py`: Trains embeddings.
- `data/`: Training datasets.
- `modelsLLM/`: LLM adapters and related files.

### `setup/`
Initialization and setup scripts.
- `model_tools/`: Scripts for downloading and managing local LLMs.

### `launchers/`
Scripts to launch various parts of the system.

### `deprecated/`
Old, obsolete, or duplicate scripts. These are kept for archival purposes but should not be used.

## Usage

Run scripts from the project root directory using `python -m scripts.<category>.<script_name>` or `python scripts/<category>/<script_name>.py`.
