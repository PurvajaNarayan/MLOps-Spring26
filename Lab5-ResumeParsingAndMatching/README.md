# Lab 5 — Resume Parsing & Matching on Google Cloud Run

## Project Overview

This project deploys a **Resume Parser** microservice to **Google Cloud Run**. Users can upload a PDF resume through a browser-based frontend, and the service extracts structured data — contact information, skills, work experience, and education — using rule-based NLP with `pdfplumber`.

**Live URL:** `https://resume-parser-801527443844.us-central1.run.app`

---

## What This Project Does

| Feature | Description |
|---|---|
| **PDF Upload** | Drag-and-drop or click-to-browse file upload (10 MB max) |
| **Resume Parsing** | Extracts name, email, phone, skills, experience, and education from PDF resumes |
| **Structured Output** | Returns parsed data as JSON via REST API (`POST /parse`) |
| **Frontend UI** | Interactive single-page app with real-time result rendering |
| **Health Check** | `GET /health` endpoint for container readiness probes |
| **Cloud Run Deployment** | Fully containerized and deployed as a serverless service |

---

## Enhancements Beyond the Basic Cloud Runner Lab

The base Cloud Runner Lab deploys a minimal Flask "Hello, World!" app. This project significantly extends that foundation:

| Aspect | Basic Cloud Runner Lab | This Project |
|---|---|---|
| **Framework** | Flask | FastAPI (async, auto-generated OpenAPI docs) |
| **Python Version** | 3.8 | 3.11 |
| **Application Logic** | Single `"Hello, World!"` string response | Full PDF parsing pipeline with regex-based section extraction, contact parsing, skills extraction, and experience/education structuring |
| **API Endpoints** | 1 (`GET /`) | 3 (`GET /`, `GET /health`, `POST /parse`) |
| **Frontend** | None | Drag-and-drop UI with dark glassmorphism theme, loading states, and structured result cards |
| **File Handling** | None | Multipart file upload with validation (type check, size limit) |
| **Data Models** | None | Pydantic models (`ParsedResume`, `ExperienceEntry`, `EducationEntry`) with automatic request/response validation |
| **Dependencies** | Flask only | FastAPI, uvicorn, pdfplumber, python-multipart |
| **Dockerfile** | Basic single-stage | Optimized with `--no-cache-dir`, separated COPY layers for better caching, multi-platform build (`linux/amd64`) |
| **API Documentation** | None | Auto-generated Swagger UI at `/docs` (built into FastAPI) |
| **Error Handling** | None | HTTP 400/422 responses with descriptive error messages |

---

## Project Structure

```
resume-match-api/services/resume-parser/
├── Dockerfile                  # Container definition (Python 3.11-slim)
├── requirements.txt            # Python dependencies
├── app/
│   ├── main.py                 # FastAPI app with routes and static file serving
│   ├── models.py               # Pydantic data models for parsed resume data
│   └── parser.py               # PDF text extraction and section parsing logic
├── static/
│   └── index.html              # Frontend UI (drag-and-drop upload + results display)
├── sample_resumes/             # Directory for test PDF resumes
└── tests/                      # Test directory
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend UI |
| `GET` | `/health` | Returns `{"status": "healthy"}` |
| `GET` | `/docs` | Auto-generated Swagger/OpenAPI documentation |
| `POST` | `/parse` | Accepts a PDF file (`multipart/form-data`), returns structured resume data |

### Example Response (`POST /parse`)

```json
{
  "name": "Jane Doe",
  "email": "jane.doe@email.com",
  "phone": "(555) 123-4567",
  "skills": ["Python", "Machine Learning", "Docker", "SQL"],
  "experience": [
    {
      "company": "Acme Corp",
      "title": "Software Engineer",
      "dates": "Jan 2022 – Present",
      "bullets": ["Built data pipelines", "Led team of 3 engineers"]
    }
  ],
  "education": [
    {
      "institution": "MIT",
      "degree": "B.S. Computer Science",
      "dates": "2018 – 2022",
      "details": ["GPA: 3.8"]
    }
  ],
  "raw_text": "..."
}
```

---

## Deployment Steps

### Prerequisites
- Google Cloud account with billing enabled
- Docker Desktop installed
- `gcloud` CLI installed (`brew install --cask google-cloud-sdk`)

### 1. Set Up GCP Project
```bash
gcloud init                     # Create/select project
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
gcloud auth configure-docker
```

### 2. Build & Push (for Apple Silicon Macs)
```bash
cd resume-match-api/services/resume-parser
docker build --platform linux/amd64 -t gcr.io/YOUR_PROJECT_ID/resume-parser .
docker push gcr.io/YOUR_PROJECT_ID/resume-parser
```

### 3. Deploy to Cloud Run
```bash
gcloud run deploy resume-parser \
  --image gcr.io/YOUR_PROJECT_ID/resume-parser \
  --platform managed \
  --region us-central1 \
  --port 8080 \
  --allow-unauthenticated
```

### 4. Access the Service
Cloud Run provides a public URL upon deployment. Open it in a browser to use the frontend, or call the API directly:
```bash
curl -X POST https://YOUR_SERVICE_URL/parse \
  -F "resume=@my_resume.pdf"
```

---

## Running Locally

```bash
cd resume-match-api/services/resume-parser
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```
Then open `http://localhost:8080` in your browser.

---

## Monitoring & Scaling

- **Metrics**: View request count, latency, and memory usage in the [Cloud Run Console](https://console.cloud.google.com/run)
- **Auto-scaling**: Cloud Run automatically scales from 0 to N instances based on incoming traffic
- **Configuration**: Adjust min/max instances, memory, and CPU via the Cloud Run console or `gcloud run services update`
