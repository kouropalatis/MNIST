# MNIST MLOps Project 🚀

![Python](https://img.shields.io/badge/python-3.12-blue) ![Hydra](https://img.shields.io/badge/config-hydra-orange) ![FastAPI](https://img.shields.io/badge/api-fastapi-green) ![Streamlit](https://img.shields.io/badge/frontend-streamlit-red) ![Docker](https://img.shields.io/badge/docker-enabled-blue)

A complete MLOps project demonstrating best practices for training, deploying, and serving a PyTorch model for Corrupted MNIST classification.

## 🌟 Features
- **Training**: Reproducible training pipelines using **Hydra** and **WandB**.
- **Backend**: High-performance REST API built with **FastAPI**.
- **Frontend**: Interactive **Streamlit** UI for model inference.
- **Cloud**: Automated Docker builds via **Google Cloud Build** and **Artifact Registry**.
- **Infrastructure**: Dependency management with **uv**.

---

## 🛠️ Installation

Prerequisites: [Python 3.12+](https://www.python.org/) and [uv](https://github.com/astral-sh/uv).

```bash
# Clone the repository
git clone <your-repo-url>
cd MNIST

# Install dependencies (fast!)
uv sync
```

---

## 💻 Usage

### 1. Train the Model 🧠
Train the CNN using Hydra configuration (`configs/config.yaml`).
```bash
uv run train
```
*Experiments are logged to Weights & Biases automatically.*

### 2. Start the Backend API 🔌
Run the FastAPI server for inference.
```bash
uv run uvicorn src.mnist.backend:app --reload
```
*Docs available at `http://localhost:8000/docs`.*

### 3. Start the Frontend UI 🖥️
Run the interactive web app to upload images and get predictions.
```bash
streamlit run src/mnist/frontend.py
```

---

## ☁️ Cloud Deployment

This project is configured for **Google Cloud Platform**.

1. **Build Docker Images**:
   Uses `cloudbuild.yaml` to build and push images to Artifact Registry (`mlops-repos`).
   ```bash
   gcloud builds submit .
   ```

2. **Artifacts**:
   - Backend Image: `.../mlops-repos/backend:latest`
   - Frontend Image: `.../mlops-repos/frontend:latest`

## 📂 Project Structure
```
├── configs/             # Hydra configurations
├── dockerfiles/         # Docker setup for API & Frontend
├── src/mnist/
│   ├── backend.py       # FastAPI application
│   ├── frontend.py      # Streamlit application
│   ├── train.py         # Training script
│   └── model.py         # PyTorch Model
├── .github/workflows/   # CI/CD (Linting & Tests)
└── pyproject.toml       # Dependencies (uv)
```