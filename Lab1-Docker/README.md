# Lab 1: Docker - Go Web Application

This project is part of the **MLOps Spring 2026** course. It demonstrates how to containerize a simple Go web application using Docker, showcasing multi-stage builds and lightweight container images.

## Project Overview

The application is a simple static site server written in Go. It serves HTML/CSS/JS content from a `static/` directory.

### Tech Stack
- **Go (Golang)**: The backend server handling HTTP requests.
- **Docker**: Used for containerizing the application.
- **Alpine Linux**: A lightweight security-oriented Linux distribution used as the base image to keep the container size small.

## Getting Started

You can run this project either using Docker (recommended) or directly on your local machine if you have Go installed.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed.
- (Optional) [Go 1.25+](https://go.dev/dl/) installed for local development.

### Run with Docker (Recommended)

1.  **Build and Run**:
    This command builds the image and starts the container in the background.
    ```bash
    docker compose up -d --build
    ```

2.  **Access the Site**:
    Open your browser and navigate to:
    [http://localhost:8080](http://localhost:8080)

3.  **View Logs**:
    To see the server logs:
    ```bash
    docker compose logs -f
    ```

4.  **Stop**:
    To stop and remove the containers:
    ```bash
    docker compose down
    ```

### Run Locally (development)

If you want to run the code without Docker:

1.  **Run**:
    ```bash
    go run main.go
    ```

2.  **Access**:
    Open [http://localhost:8080](http://localhost:8080).

## Project Structure

- `main.go`: The Go entry point that spins up the HTTP server.
- `static/`: Contains the frontend assets (HTML, CSS, JS).
- `Dockerfile`: Defines the multi-stage build process:
    - **Stage 1 (Builder)**: Compiles the Go binary.
    - **Stage 2 (Final)**: Copies the binary to a minimal Alpine image.
- `docker-compose.yml`: Simplified orchestration for running the container.
