#  Lab2-BuggyCode: Bug Detection MLOps Pipeline

This project implements an end-to-end MLOps pipeline for generating synthetic buggy code and training a model to detect bugs using **Apache Airflow**.

The pipeline orchestrates the entire workflow from data ingestion and synthetic data generation to model training and evaluation.

##  Quick Start

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

### Setup & Run

1.  **Start Airflow services**:
    ```bash
    docker-compose up -d
    ```
    *This will initialize the database and start the Webserver, Scheduler, and Postgres containers.*

2.  **Access Airflow UI**:
    - **URL**: `http://localhost:8081`
    - **Username**: `admin`
    - **Password**: `admin`

3.  **Trigger the Pipeline**:
    - In the Airflow UI, enable the DAGs.
    - Trigger `stage1_dataset_analysis` to start the process.

## Project Structure

The project is structured as a standard Airflow environment:

- **`dags/`**: Contains the Python files defining the Airflow Directed Acyclic Graphs (DAGs) for each stage of the pipeline.
- **`plugins/`**: Custom Airflow plugins and utility scripts used by the DAGs.
    - `utils/code_generator.py`: Logic for generating synthetic buggy code.
    - `utils/data_loader.py`: Handles loading CodeXGLUE dataset.
    - `utils/quality_filters.py`: Implements filters to ensure data quality.
    - `utils/model_trainer.py`: Encapsulates model training logic (PyTorch/Transformers).
    - `utils/llm_wrapper.py`: Interface for LLM interactions.
- **`data/`**: Directory mapped to the Docker container for storing datasets (persists between runs).
- **`models/`**: Directory for saving trained model checkpoints.
- **`logs/`**: Airflow logs for debugging task execution.
- **`docker-compose.yaml`**: Defines the services and network configuration for the Airflow environment.

##  Pipeline Stages

The MLOps pipeline is divided into 6 distinct stages, each represented by an Airflow DAG:

### 1. Data Analysis (`stage1_dataset_analysis`)
- **Goal**: Load the base CodeXGLUE defect detection dataset.
- **Tasks**:
    - `load_dataset`: Downloads dataset splits from HuggingFace.
    - `generate_analysis`: Computes statistics on the base data.
    - `print_summary`: Outputs dataset distribution.

### 2. Synthetic Generation (`stage2_simple_generation`)
- **Goal**: Augment the dataset with synthetic buggy code samples.
- **Mechanism**: Runs parallel tasks to generate batches of buggy code using an LLM.
- **Output**: Raw synthetic code samples in `data/synthetic_batches/`.

### 3. Quality Filtering (`stage3_quality_filtering`)
- **Goal**: Ensure the quality of synthetic data before training.
- **Filters**:
    - Syntax validation (AST parsing).
    - Minimum code length.
    - Function definition check.
- **Output**: Filtered dataset `data/synthetic_filtered.csv`.

### 4. Merge & Split (`stage4_merge_datasets`)
- **Goal**: Combine real and synthetic data and prepare splits.
- **Process**:
    - Merges base CodeXGLUE data with filtered synthetic data.
    - Creates stratified Train (80%), Validation (10%), and Test (10%) splits.
- **Output**: `final_train.csv`, `final_val.csv`, `final_test.csv`.

### 5. Model Training (`stage5_train_model`)
- **Goal**: Fine-tune a CodeBERT model for bug detection.
- **Tasks**:
    - `prepare_and_tokenize_data`: Tokenizes the text data.
    - `train_model`: Fine-tunes the model (supports GPU if available).
    - *Note*: Includes a `stage5_quick_setup` DAG for downloading a pre-trained model for demonstration purposes.

### 6. Evaluation (`stage6_evaluation`)
- **Goal**: Assess model performance against a baseline.
- **Metrics**: Accuracy, Precision, Recall, F1 Score.
- **Comparisons**: Compares the trained model against a simple majority-class classifier baseline.
- **Output**: A final comparison report in `results/`.

##  Configuration

- **Environment Variables**: Managed via `.env` file (if present) or `docker-compose.yaml`.
- **Dependencies**: Python packages are listed in `requirements.txt`.
