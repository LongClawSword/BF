# Bill Data Extraction AI

## Overview
This project is a high-accuracy solution for the Bajaj Finserv Health Datathon. It provides an API to extract line items, quantities, rates, and amounts from complex health insurance bills.

## Key Differentiators
-   **Hybrid Architecture**: Combines the precision of **Google Gemini 1.5 Flash** (Multimodal LLM) with the robustness of **PaddleOCR**.
-   **Intelligent Fallback**: Automatically switches to local OCR if the LLM service is unavailable or not configured.
-   **Production Ready**: Dockerized, Gunicorn-backed, and fully configurable via environment variables.
-   **Advanced Preprocessing**: Uses CLAHE and denoising to handle poor quality scans.

## Tech Stack
-   **Framework**: FastAPI
-   **LLM**: Google Gemini 1.5 Flash
-   **OCR**: PaddleOCR (v2.7+)
-   **PDF Engine**: PyMuPDF (fitz)
-   **Deployment**: Docker, Gunicorn

## Setup & Installation

### Prerequisites
-   Docker installed
-   Google Gemini API Key (Optional, for higher accuracy)

### Running with Docker
1.  **Build the image**:
    ```bash
    docker build -t bill-extractor .
    ```

2.  **Run the container**:
    ```bash
    # Create a .env file with your GEMINI_API_KEY first
    docker run -p 8000:8000 --env-file .env bill-extractor
    ```

### Running Locally
1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set API Key**:
    ```bash
    export GEMINI_API_KEY="your_key_here"
    ```

3.  **Start Server**:
    ```bash
    gunicorn app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
    ```

## API Usage
**Endpoint**: `POST /extract-bill-data`

**Request**:
```json
{
  "document": "https://example.com/path/to/bill.pdf"
}
```

**Response**:
```json
{
  "is_success": true,
  "data": {
    "pagewise_line_items": [
      {
        "page_no": "1",
        "page_type": "Bill Detail",
        "bill_items": [
          {
            "item_name": "Consultation",
            "item_amount": 500.0,
            "item_rate": 500.0,
            "item_quantity": 1.0
          }
        ]
      }
    ],
    "total_item_count": 1
  }
}
```
