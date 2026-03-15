# Setup Guide: Byeias

This guide explains how to set up the Byeias project for local development and how to run it using Docker.

## Prerequisites

Make sure you have the following tools installed:
- **Python** (>= 3.13)
- **Node.js** (>= 20) & npm
- **Poetry**
- **Docker**

---

## 💻 Local Development Setup

To develop locally, you need to run both the Python backend and the React/Vite frontend in separate terminal windows.

### 1. Configure the API Key
Before starting the backend, edit the `configs/config.yaml` file and insert your Mistral API Key at `model.mistral.api_key`.

### 2. Start the Backend API
Open a terminal and install the Python dependencies using Poetry:

```bash
# Install dependencies
poetry install

# Navigate to the source directory
cd src

# Start the FastAPI backend
poetry run uvicorn byeias.backend.api:app --reload
```

### 3. Start the Frontend

```bash
# Navigate to the frontend directory
cd src/byeias/frontend

# Install Node dependencies (only required the first time)
npm install

# Start the development server
npm run dev
```

Opens at **http://localhost:8501**

### 4. Docker Deployment

If you want to run the entire application (Frontend + Backend) in a single container, you can easily deploy it using Docker.

#### 1. Build the Docker Image
Make sure your `configs/config.yaml` is configured with your Mistral API key before building, as it will be included in the image. In the root directory of the project, run:

```bash
docker build -t byeias-app .
```

#### 2. Run the Container
Start the container and map the application port to your local machine:

```bash
docker run -p 8000:8000 byeias-app
```
