# AI VisionLab

Automated Image Classification, Multi-Model Benchmarking, and Explainable AI Platform.

---

## Overview

AI VisionLab is an automated machine learning (AutoML) and deep learning evaluation platform designed for image classification experiments. The application provides an end-to-end experimental workflow: users can upload image datasets, inspect visual characteristics, receive automated preprocessing recommendations, train candidate models across classical ML and deep neural architectures, generate Explainable AI (XAI) activation heatmaps, and export reproducible reports.

The platform is designed to run in containerized environments via Docker as well as native host environments.

---

## Project Objectives & Key Capabilities

- **Automated Pipeline Exploration:** Eliminate repetitive boilerplate code when comparing classical machine learning algorithms, convolutional neural networks, and transfer learning backbones on image datasets.
- **Leakage-Safe Validation:** Enforce strict stratified train/test split generation and duplicate file detection before model fitting to prevent data contamination between training and evaluation partitions.
- **Data-Driven Preprocessing:** Analyze native dataset characteristics (resolutions, aspect ratios, color modes, and class imbalance ratios) to provide actionable preprocessing recommendations.
- **Model Explainability:** Provide transparent visual explanations for convolutional architectures using Grad-CAM and Grad-CAM++ to reveal spatial focus areas during prediction.
- **Reproducible Experimentation:** Export self-contained experiment artifacts, including structured CSV leaderboards, publication-ready PDF evaluation summaries, and reproducible Jupyter Notebooks (`.ipynb`) containing recorded training history and source snapshots.

---

## Features

- **Dataset Ingestion and Structure Detection:**
  - Automatic detection of standard `ImageFolder` directory structures (`dataset/class_name/image.png`).
  - Automatic detection of pre-partitioned split datasets containing `seg_train`, `seg_test`, and unlabeled `seg_pred` folders.
  - Safe extraction with automatic pruning of nested archive wrappers and operating system metadata artifacts (`__MACOSX`, `._*`, `.DS_Store`).
  - SHA-256/MD5 exact duplicate detection to prevent identical images from appearing across both training and test partitions.
- **Dataset Profiling and Preprocessing:**
  - Dimensional analysis: average, minimum, and maximum dimensions; uniform resolution detection.
  - Distribution metrics: per-class image counts, minority/majority class identification, and imbalance ratios.
  - Preprocessing configuration: target resizing (e.g., 64x64, 96x96, 128x128), channel normalization (ImageNet Z-score or Min-Max), class balancing via weighted cross-entropy loss, and data augmentations (flips, rotations, color jitter, random crops, Gaussian blur).
  - Dynamic preview of augmented samples before initiating training.
- **Multi-Model Search & Training Telemetry:**
  - Benchmarks 16 candidate model configurations across Classical ML, Deep Learning, Transfer Learning, and Ensembles.
  - Background asynchronous execution decoupled from web request threads.
  - Real-time epoch telemetry: live training loss, validation accuracy, elapsed time, ETA calculation, and streaming execution logs.
  - Early-stopping cancellation support via the user interface.
- **Comprehensive Evaluation & Visualizations:**
  - Metric tracking: Top-1 Accuracy, Precision (Macro), Recall (Macro), Macro F1-score, ROC-AUC score, and Generalization Gap (Train Acc vs. Holdout Acc).
  - Interactive confusion matrix display and ROC curve plots.
  - Side-by-side multi-model comparison matrix highlighting the top-performing configuration.
- **Explainable AI (XAI):**
  - Integrated Grad-CAM and Grad-CAM++ activation heatmaps for convolutional architectures.
  - Visual overlay on input images showing discriminative regions driving class predictions.
- **Export Formats:**
  - Structured CSV leaderboard export.
  - PDF experiment summary generated via ReportLab with embedded confusion matrices and activation heatmaps.
  - Self-contained, executable Jupyter Notebook containing runtime configurations and execution logs.

---

## Technology Stack

| Layer | Technologies |
|---|---|
| **Programming Language** | Python 3.10 / 3.11 |
| **Web Framework** | Flask 3.x, Werkzeug 3.x |
| **Production WSGI Server** | Gunicorn 21.x / 27.x |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Deep Learning** | PyTorch, Torchvision |
| **Image Processing** | Pillow (PIL), NumPy, SciPy |
| **Reporting & Export** | ReportLab (PDF), PyYAML, JSON |
| **Frontend** | HTML5, CSS3 (Modern Flex/Grid layout), Vanilla JavaScript (ES6+ fetch polling) |
| **Containerization** | Docker, Docker Compose (Debian Bookworm base) |

---

## Project Architecture & Directory Structure

```text
AI_vision_lab/
├── .dockerignore                # Docker build context exclusions
├── .env.example                 # Environment configuration template
├── .gitignore                   # Git repository ignore patterns
├── Dockerfile                   # Multi-platform container build file
├── docker-compose.yml           # Docker Compose service definition
├── requirements.txt             # Python dependencies
├── config.py                    # Centralized environment configuration loader
├── app.py                       # Flask application factory, routing & job coordinator
├── wsgi.py                      # WSGI entrypoint for production application servers
├── run_pipeline.py              # Standalone CLI entrypoint for AutoML search
├── make_sample_data.py          # Synthetic dataset generator for local testing
├── example_config.yaml          # Sample search-space definition for CLI runs
├── shapes_test.zip              # Pre-packaged sample dataset (circle, square, triangle)
├── README.md                    # Project documentation
│
├── stage1_pipeline/             # AutoML core engine
│   ├── __init__.py
│   ├── data.py                  # Dataset discovery, validation, and characteristic profiling
│   ├── splits.py                # Stratified, leakage-safe train/test split generator
│   ├── models.py                # Model registry (16 models), feature extraction, and training routines
│   ├── search.py                # Grid search execution loop, callbacks, and cancellation
│   ├── preprocessing.py         # Heuristic recommendations and image transform previews
│   ├── pipeline.py              # Main Stage 1 pipeline orchestrator (`run_stage1`)
│   └── xai.py                   # Grad-CAM and Grad-CAM++ heatmap generation engine
│
├── static/                      # Frontend static assets
│   ├── app.js                   # Client-side UI interactions and progress polling
│   └── style.css                # Application styles and dark theme design system
│
├── templates/                   # Jinja2 server-rendered templates
│   ├── base.html                # Base layout, sidebar navigation, and shell
│   ├── home.html                # Landing page & dashboard
│   ├── upload.html              # Dataset upload dropzone
│   ├── preview.html             # Dataset characteristic profiling and class distribution
│   ├── preprocessing.html       # Preprocessing recommendations & transformation preview
│   ├── models.html              # Multi-model architecture selection
│   ├── config.html              # Training hyperparameter configuration
│   ├── processing.html          # Real-time training progress and execution log terminal
│   ├── results.html             # Evaluation dashboard, confusion matrix, and XAI display
│   ├── comparison.html          # Side-by-side leaderboard comparison table
│   └── report.html              # Full printable experiment report view
│
├── sample_data/                 # Synthetic image directory created by make_sample_data.py
└── webapp_data/                 # Persistent storage (excluded from git)
    ├── uploads/                 # Uploaded archives and extracted datasets
    └── results/                 # Output JSON experiment results and cached artifacts
```

---

## Supported Models

The application defines 16 model architectures registered in `stage1_pipeline/models.py`:

| Model Identifier | Family | Architecture / Type | Epochs Configurable | Supports Grad-CAM |
|---|---|---|:---:|:---:|
| `small_cnn` | Deep Learning (`dl`) | Custom 3-block Conv2D network with ReLU and MaxPooling | Yes | Yes |
| `deep_cnn` | Deep Learning (`dl`) | 4-block Conv2D with BatchNorm, Dropout (0.3), and Dense head | Yes | Yes |
| `resnet18` | Deep Learning (`dl`) | ResNet-18 pretrained on ImageNet with fine-tuned classifier | Yes | Yes |
| `mobilenet` | Deep Learning (`dl`) | MobileNetV2 pretrained on ImageNet with fine-tuned classifier | Yes | Yes |
| `vgg16` | Deep Learning (`dl`) | VGG-16 backbone with custom dense classification head | Yes | Yes |
| `efficientnet` | Deep Learning (`dl`) | EfficientNet-B0 backbone with custom linear head | Yes | Yes |
| `random_forest` | Classical ML (`ml`) | Scikit-learn RandomForestClassifier (100–200 estimators) | No | No |
| `xgboost` | Classical ML (`ml`) | XGBoost gradient boosted trees (histogram-based) | No | No |
| `svm` | Classical ML (`ml`) | Support Vector Classifier with RBF kernel and probability calibration | No | No |
| `logistic_regression` | Classical ML (`ml`) | L2-regularized multinomial Logistic Regression | No | No |
| `knn` | Classical ML (`ml`) | K-Nearest Neighbors Classifier (k=5) | No | No |
| `gradient_boosting` | Classical ML (`ml`) | Scikit-learn GradientBoostingClassifier | No | No |
| `hybrid_attention` | Hybrid (`hybrid`) | CNN backbone with Convolutional Block Attention Module (CBAM) | Yes | Yes |
| `hybrid_transformer` | Hybrid (`hybrid`) | CNN stem coupled with Multi-Head Self-Attention Transformer encoder | Yes | Yes |
| `ensemble_voting` | Ensemble (`ensemble`) | Soft-voting ensemble combining Random Forest, Gradient Boosting, and SVM | No | No |
| `ensemble_stacking` | Ensemble (`ensemble`) | Stacking classifier using RF, GB, and SVM with Logistic Regression meta-learner | No | No |

---

## System Requirements & Prerequisites

### Docker Environment
- [Docker Desktop](https://docs.docker.com/get-docker/) (macOS 11+, Windows 10/11 64-bit) or [Docker Engine](https://docs.docker.com/engine/install/) 24.0+ with [Docker Compose](https://docs.docker.com/compose/install/) v2.20+.
- Hardware Allocation: Minimum 4 GB RAM (8 GB recommended for Deep Learning models), 2 CPU cores, and 10 GB free disk space.

### Native Host Environments
- **Python:** Python 3.10.x or 3.11.x (Python 3.12+ may encounter compatibility issues with specific compiled C-extensions).
- **Package Manager:** `pip` version 23.0+.
- **Compilers / Libraries:**
  - macOS: Xcode Command Line Tools (`xcode-select --install`) and OpenMP runtime (`libomp`).
  - Linux: `build-essential`, `libgomp1`, `libglib2.0-0`, `curl`.
  - Windows: Microsoft Visual C++ 2015–2022 Redistributable (x64).

---

## Setup & Execution: Docker (Recommended)

Running AI VisionLab inside Docker ensures an isolated, cross-platform runtime where all system libraries (including OpenMP for XGBoost and headless graphics libraries) are pre-installed.

### 1. Clone the Repository

```bash
git clone https://github.com/apoorvaww/AI_vision_lab.git
cd AI_vision_lab
```

### 2. Configure Environment Variables

Copy the example environment template:

```bash
cp .env.example .env
```

Review and adjust variables in `.env` if necessary (e.g., custom port or secret key).

### 3. Build and Start the Application

Build the container image and launch the service in the background:

```bash
docker compose up --build -d
```

### 4. Verify Application Health

Confirm that the application container is running and healthy:

```bash
curl -f http://localhost:5000/api/health
```

Expected response:
```json
{"env":"production","service":"AI VisionLab","status":"healthy"}
```

Access the user interface by navigating to **[http://localhost:5000](http://localhost:5000)** in your browser.

### 5. Managing the Container Lifecycle

- **View real-time logs:**
  ```bash
  docker compose logs -f visionlab
  ```
- **Restart the container:**
  ```bash
  docker compose restart visionlab
  ```
- **Stop the container (preserving uploaded datasets and results):**
  ```bash
  docker compose down
  ```
- **Stop the container and delete persistent volumes:**
  > [!WARNING]
  > Running `docker compose down -v` permanently removes the `ai_visionlab_data` Docker volume, deleting all uploaded datasets, training logs, and saved experiment outputs.
  ```bash
  docker compose down -v
  ```

---

## Setup & Execution: macOS Native

This environment has been tested and verified on macOS 14+ (Apple Silicon arm64) using Python 3.10.21.

### 1. Prerequisites and Python Installation

Ensure Python 3.10 or 3.11 is installed. You can install it via [Homebrew](https://brew.sh/):

```bash
brew install python@3.10
```

### 2. Virtual Environment Setup

From the project root, create and activate a virtual environment:

```bash
/opt/homebrew/opt/python@3.10/bin/python3.10 -m venv venv
source venv/bin/activate
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

### 3. OpenMP Dependency for XGBoost on macOS

`pip install xgboost` on macOS dynamically links against `@rpath/libomp.dylib`. Because macOS does not include OpenMP by default, attempting to import XGBoost without OpenMP raises an `XGBoostError`.

Install the OpenMP library via Homebrew:

```bash
brew install libomp
```

*Alternative for isolated setups:* If Scikit-learn or PyTorch is already installed in your virtual environment, you can link their internal OpenMP runtime to the expected Homebrew path:

```bash
mkdir -p /opt/homebrew/opt/libomp/lib
cp $(python -c "import sklearn, pathlib; print(pathlib.Path(sklearn.__file__).parent / '.dylibs' / 'libomp.dylib')") /opt/homebrew/opt/libomp/lib/
```

### 4. Install Dependencies

Install all core and optional dependencies:

```bash
pip install -r requirements.txt
```

Verify that all model dependencies import correctly:

```bash
python -c "import stage1_pipeline.models as m; print('Models Available:', sum(1 for v in m.MODEL_REGISTRY.values() if v['available']), '/ 16')"
```

### 5. Run the Application

```bash
python app.py
```

The application will bind to `0.0.0.0:5000` (or the port defined in `.env`). Open **[http://localhost:5000](http://localhost:5000)** in your browser.

> [!NOTE]
> **macOS Port 5000 Conflict:** On macOS Monterey and later, port 5000 is used by the system AirPlay Receiver service. If you encounter `Address already in use`, either disable AirPlay Receiver in **System Settings > General > AirDrop & AirPlay**, or specify an alternative port:
> ```bash
> PORT=5001 python app.py
> ```

---

## Setup & Execution: Windows Native (Pending Validation)

> [!IMPORTANT]
> **Validation Status:** Windows native execution has **not yet been verified** in this test environment. The instructions below reflect standard Windows Python deployment practices but may require manual adjustments for compiler toolchains or path limits. For guaranteed stability on Windows, use the [Docker Quickstart](#setup--execution-docker-recommended).

### 1. Prerequisites

Install Python 3.10 or 3.11 from the official [Python for Windows](https://www.python.org/downloads/windows/) installer. Ensure the option **"Add python.exe to PATH"** is selected during installation.

Ensure the [Microsoft Visual C++ 2015–2022 Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) is installed.

### 2. Create and Activate Virtual Environment

Open PowerShell as an Administrator or standard user:

```powershell
cd C:\path\to\AI_vision_lab

python -m venv venv
```

If script execution is disabled on your system, enable it for the current process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Upgrade package installer and install requirements:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment

Copy the configuration template:

```powershell
Copy-Item .env.example .env
```

### 5. Start the Application

```powershell
python app.py
```

Open **[http://localhost:5000](http://localhost:5000)** in your browser. Verify the service status using PowerShell:

```powershell
Invoke-RestMethod -Uri http://localhost:5000/api/health
```

### Common Windows Troubleshooting
- **Path Length Limits:** Python may encounter errors when unzipping deeply nested dataset structures. Enable long paths in the Windows Registry or run PowerShell with elevated permissions.
- **OpenMP Runtime:** Windows wheels for XGBoost typically bundle `vcomp140.dll`. If DLL load errors occur, ensure the Visual C++ Redistributable package is installed.

---

## Configuration & Environment Variables

Application settings are managed through `config.py` and can be overridden using environment variables or a local `.env` file.

| Variable Name | Default Value | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Network host interface the server binds to (`0.0.0.0` allows container bridging). |
| `PORT` | `5000` | Port on which the web application listens. |
| `FLASK_ENV` | `production` | Environment mode (`production` or `development`). |
| `FLASK_DEBUG` | `false` | Enables Flask auto-reloading and debug traceback if set to `true`. |
| `STAGE1_SECRET_KEY` | `dev-secret-key-change-me` | Secret key used for session cookie signing. |
| `MAX_UPLOAD_MB` | `500` | Maximum allowable uploaded dataset archive size in Megabytes. |
| `DATA_DIR` | `<project_root>/webapp_data` | Root directory where uploaded files, extracted datasets, and JSON results are stored. |

> [!WARNING]
> **Production Security:** The default secret key (`dev-secret-key-change-me`) is insecure. You must generate a cryptographically strong key (e.g. `python -c "import secrets; print(secrets.token_hex(32))"`) and assign it to `STAGE1_SECRET_KEY` in production environments.

---

## Application Routes & Health Checks

The application registers the following HTTP routes in `app.py`:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Dashboard and experiment overview landing page. |
| `GET` | `/api/health` | Service health status check returning JSON status and active environment. |
| `GET` | `/upload` | File upload interface for dataset ZIP archives. |
| `POST` | `/start` | Handles file upload, archive extraction, characteristic inspection, and job initialization. |
| `GET` | `/preview/<job_id>` | Dataset preview showing image distribution, detected classes, and sample images. |
| `GET` | `/dataset-image/<job_id>/<path>` | Secure file-serving endpoint for previewing raw dataset images. |
| `GET` | `/preprocessing/<job_id>` | AI-guided preprocessing recommendations and configuration controls. |
| `POST` | `/preprocessing/<job_id>/save` | Saves chosen preprocessing settings and forwards to model selection. |
| `GET` | `/preview-transform/<job_id>/<path>` | Dynamically returns an augmented/normalized preview of an image. |
| `GET` | `/models/<job_id>` | Model architecture selection interface (Classical ML, DL, Hybrid, Ensemble). |
| `POST` | `/models/<job_id>/save` | Persists selected models and forwards to hyperparameter configuration. |
| `GET` | `/config/<job_id>` | Training parameter configuration (epochs, batch size, learning rate, split). |
| `POST` | `/train/<job_id>` | Initiates asynchronous background AutoML training for the job. |
| `POST` | `/start-training/<job_id>` | Backward-compatible route alias for `/train/<job_id>`. |
| `GET` | `/processing/<job_id>` | Live progress monitoring view displaying progress bars and metrics. |
| `GET` | `/api/progress/<job_id>` | Polling API endpoint returning JSON progress state, logs, and loss/accuracy metrics. |
| `POST` | `/api/cancel/<job_id>` | Signals early cancellation of the active training loop. |
| `GET` | `/results/<job_id>` | Evaluation results dashboard, confusion matrix, and Grad-CAM visualizations. |
| `GET` | `/compare/<job_id>` | Side-by-side leaderboard comparison across all evaluated models. |
| `GET` | `/report/<job_id>` | Full experiment report view formatted for printing or review. |
| `GET` | `/download/<job_id>/csv` | Downloads leaderboard metrics and hyperparameter records as a `.csv` file. |
| `GET` | `/download/<job_id>/notebook` | Downloads a self-contained, reproducible Jupyter Notebook (`.ipynb`). |
| `GET` | `/download/<job_id>/pdf` | Generates and downloads a formatted PDF report with embedded heatmaps. |

---

## Dataset Preparation & Upload Instructions

The dataset discovery engine in `stage1_pipeline/data.py` supports standard image classification folder formats. Datasets must be compressed into a single `.zip` file prior to upload.

### Supported Layout 1: Standard ImageFolder
Each subfolder represents a distinct class label:

```text
dataset.zip
└── dataset/
    ├── circle/
    │   ├── circle_001.png
    │   └── circle_002.png
    ├── square/
    │   ├── square_001.png
    │   └── square_002.png
    └── triangle/
        ├── triangle_001.png
        └── triangle_002.png
```

### Supported Layout 2: Pre-Partitioned Dataset
Recognizes predefined train/test/prediction structures (such as Intel Image Classification):

```text
dataset.zip
└── dataset/
    ├── seg_train/
    │   ├── buildings/
    │   └── forest/
    ├── seg_test/
    │   ├── buildings/
    │   └── forest/
    └── seg_pred/
        ├── preview_01.jpg
        └── preview_02.jpg
```
- `seg_train` / `train` / `training`: Labeled training samples.
- `seg_test` / `test` / `val` / `validation`: Preserved holdout test set used for final evaluation.
- `seg_pred` / `pred` / `prediction`: Treated as unlabelled prediction images, excluded from class calculations.

### Validation Rules & Constraints
- Supported image file formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.
- Non-image files and macOS system files (`.DS_Store`, `__MACOSX`, `._*`) are automatically ignored.
- The dataset must contain at least **two distinct class folders** with valid images.
- Exact duplicate images are identified via MD5 hash comparison and removed before partitioning to prevent data leakage. Conflicting duplicates (identical images under different class labels) are completely discarded.

The web app accepts ZIP uploads up to 2,048 MB by default. To change this limit,
set `MAX_UPLOAD_MB` to a positive integer. For backward compatibility,
`STAGE1_MAX_UPLOAD_MB` is also supported when `MAX_UPLOAD_MB` is not set.

If the application is behind a reverse proxy or hosting platform, configure
that layer's request-body limit to at least the same size; otherwise it may
reject the upload before Flask receives it.

---


## Model Training & Evaluation Workflow

```text
Upload ZIP Archive
       │
       ▼
Dataset Discovery & Validation (stage1_pipeline/data.py)
       │
       ▼
Characteristic Profiling & Recommendations (stage1_pipeline/preprocessing.py)
       │
       ▼
Model Architecture Selection (stage1_pipeline/models.py)
       │
       ▼
Hyperparameter Configuration (Epochs, Batch Size, Learning Rate, Metric)
       │
       ▼
Stratified Holdout Partitioning (stage1_pipeline/splits.py)
       │
       ▼
Asynchronous AutoML Search Loop (stage1_pipeline/search.py)
       │
       ├─► Classical ML Feature Extraction (HOG + Color Histograms + Spatial Grids)
       └─► Deep Neural Network Training (PyTorch DataLoader + Backprop)
       │
       ▼
Holdout Metric Evaluation & Confusion Matrix Computation
       │
       ▼
Grad-CAM Heatmap Generation for Convolutional Models (stage1_pipeline/xai.py)
       │
       ▼
Results Leaderboard, PDF Summary, CSV & Notebook Export
```

> [!NOTE]
> **Evaluation Integrity Note:** Reported validation accuracy and F1 scores are calculated strictly on holdout test partitions excluded during model training. High scores achieved on small synthetic datasets (e.g., `shapes_test.zip`) reflect model separation on distinct geometric shapes, but do not guarantee identical generalization on complex real-world imagery.

---

## Explainable AI (Grad-CAM & Grad-CAM++)

For convolutional neural networks (`small_cnn`, `deep_cnn`, `resnet18`, `mobilenet`, `vgg16`, `efficientnet`, and hybrid architectures), AI VisionLab provides Explainable AI visualizations via `stage1_pipeline/xai.py`:

- **Technique:** Computes gradients of the target class score with respect to feature activation maps of the final convolutional layer.
- **Methods Supported:**
  - **Grad-CAM:** Weights activation maps by average gradients across spatial dimensions.
  - **Grad-CAM++:** Uses second- and third-order positive partial derivatives for refined pixel-level attribution and handling multi-instance scenes.
- **Rendering:** Heatmaps are normalized and blended with the original input image using a jet color palette (blue = low activation, red = highest discriminative focus), returning a PNG byte stream rendered directly in the web UI.

---

## Reports & Export Formats

1. **CSV Export (`/download/<job_id>/csv`):**
   Tabular representation containing model identifier, family, train/test split ratio, accuracy, macro precision, macro recall, macro F1, ROC-AUC score, parameter count, training duration (seconds), and hyperparameter JSON.
2. **PDF Evaluation Summary (`/download/<job_id>/pdf`):**
   Multi-page A4 document generated via ReportLab detailing dataset summary statistics, candidate leaderboard table, best model parameters, confusion matrix, and embedded Grad-CAM visual heatmaps.
3. **Reproducible Jupyter Notebook (`/download/<job_id>/notebook`):**
   Generates a standalone `.ipynb` notebook embedding:
   - Complete pipeline source code snapshots stored during training.
   - Exact configuration dictionaries and dataset paths.
   - Code cells to reconstruct the pipeline package and re-run evaluations independently.
   - Captured execution output logs and evaluation metric dictionaries.

---

## Standalone CLI Pipeline

The core AutoML search can be executed independently of the web application:

```bash
python run_pipeline.py \
    --data sample_data/shapes \
    --out results.json \
    --config example_config.yaml \
    --metric f1_macro
```

### CLI Arguments
- `--data` (required): Path to the dataset directory containing class subfolders.
- `--out` (optional, default: `stage1_results.json`): Path where output evaluation metrics and winner configurations will be saved as JSON.
- `--config` (optional): Path to a YAML configuration file defining custom search spaces and split ratios (see `example_config.yaml`).
- `--metric` (optional, default: `f1_macro`): Optimization metric for candidate ranking (`f1_macro` or `accuracy`).

To generate the synthetic shapes test dataset prior to running the CLI:

```bash
python make_sample_data.py
```

---

## Data Persistence & Storage Architecture

AI VisionLab stores application state and output files in `webapp_data/` (or the directory configured via `DATA_DIR`):

```text
webapp_data/
├── uploads/
│   └── <job_id>/
│       ├── uploaded_archive.zip
│       └── dataset/
└── results/
    └── <job_id>.json
```

- **Persistence across Restarts:** When the application starts or reloads, `_job(job_id)` in `app.py` checks disk storage (`webapp_data/results/<job_id>.json`). Completed experiment runs remain accessible via their URLs even after server or container restarts.
- **Docker Volume Mounting:** In Docker Compose, `webapp_data/` is mounted to the named volume `ai_visionlab_data`. Application data persists across container stops, restarts, and rebuilds.

---

## Troubleshooting

### 1. Port Already in Use (`Errno 48 / EADDRINUSE`)
- **Cause:** Another process is bound to port 5000 (common on macOS Monterey+ due to AirPlay Receiver).
- **Fix:** Specify a different port when launching:
  ```bash
  # Docker
  PORT=5001 docker compose up -d

  # Native
  PORT=5001 python app.py
  ```

### 2. XGBoost `libomp.dylib` Load Failure on macOS
- **Cause:** macOS native installation lacks OpenMP runtime.
- **Fix:** Run `brew install libomp`, or copy the bundled OpenMP runtime from Scikit-Learn as described in the [macOS Native instructions](#3-openmp-dependency-for-xgboost-on-macos).

### 3. Docker Build Error (`failed to solve ... no such file or directory`)
- **Cause:** Docker Desktop snapshotter or storage driver cache corruption during large layer exports.
- **Fix:** Clean unused Docker builder caches:
  ```bash
  docker builder prune -f
  docker compose build --no-cache
  ```

### 4. Memory Exhaustion During Deep Learning Training
- **Cause:** Training heavy backbones (ResNet-18, VGG-16) with large batch sizes exceeds available system or container RAM.
- **Fix:** In the model configuration page, decrease the batch size (e.g. from 32 to 8 or 16) and target image resolution (e.g., 96x96). In Docker Desktop, allocate at least 4 GB to 8 GB of RAM in Settings > Resources.

---

## Security Considerations

- **Secret Key Configuration:** Always override `STAGE1_SECRET_KEY` in production deployments using a cryptographically random string.
- **Upload Restrictions:** Uploaded files are restricted to `.zip` archives with a maximum default size of 500 MB (`MAX_UPLOAD_MB`).
- **Archive Extraction Safety:** File extraction in `app.py` sanitizes member filenames using `os.path.normpath` and rejects path traversal sequences (`..`) to prevent Zip Slip vulnerabilities.
- **Unprivileged Container Execution:** The `Dockerfile` creates and runs as an unprivileged user (`appuser`, UID 10001) rather than `root`.
- **Sanitized Path Retrieval:** File serving endpoints (`/dataset-image/` and `/preview-transform/`) enforce strict realpath boundary checks (`target.startswith(root + os.sep)`) to prevent directory traversal outside the dataset sandbox.

---

## Testing & Platform Validation Status

| Platform / Environment | Validation Status | Verified Features |
|---|:---:|---|
| **macOS Native (Apple Silicon M-series)** | **Verified** | Server healthcheck, dataset upload, preview, preprocessing recommendations, model training (Classical ML, DL, Transfer Learning), Grad-CAM heatmaps, CSV/PDF/Notebook exports, and standalone CLI pipeline. |
| **macOS Native (Intel x86_64)** | **Partially Verified** | Architecture paths and dependency configurations match Apple Silicon; native runs depend on local Python and OpenMP installation. |
| **Docker (Debian Bookworm base)** | **Verified Specification** | Multi-platform build configuration (`linux/amd64` and `linux/arm64`), non-root execution, health check endpoint, and persistent volume storage. |
| **Windows Native (PowerShell / Command Prompt)** | **Pending Validation** | Configuration structure and PowerShell activation scripts documented; native Windows execution not verified in current environment. |

---

## Contributing

1. Fork the repository on GitHub.
2. Create a feature branch: `git checkout -b feature/model-optimization`.
3. Commit your changes: `git commit -m "Add optimized feature descriptor"`.
4. Push to your branch: `git push origin feature/model-optimization`.
5. Open a Pull Request detailing the changes, benchmarks, and platform verification steps.

---

## License

This project is licensed under the [MIT License](LICENSE).
