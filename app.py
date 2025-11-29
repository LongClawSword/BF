import httpx
import fitz
import io
from pdf2image import convert_from_bytes
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
import os
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import utils

import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Datathon Bill Extractor")

class ExtractRequest(BaseModel):
    document: str

@app.post("/extract-bill-data")
async def extract_bill_data(request: ExtractRequest):
    job_meta = {
        "is_success": False,
        "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
        "data": {"pagewise_line_items": [], "total_item_count": 0}
    }
    try:
        url = request.document
        content = utils.download_file(url)
        
        # Determine content type from URL or magic bytes, but for now assume PDF or Image based on extension or try both
        # The prompt examples are PDF and PNG.
        # Simple heuristic:
        is_pdf = url.lower().endswith(".pdf") or content.startswith(b"%PDF")
        
        pages_images = []
        if is_pdf:
            pages_images = utils.images_from_pdf_bytes(content, dpi=300)
        else:
            try:
                pil = Image.open(io.BytesIO(content)).convert("RGB")
                pages_images = [pil]
            except Exception:
                # Fallback if it's not an image we can open
                raise HTTPException(status_code=400, detail="Unsupported file format")
        
        page_results = []
        page_no = 1
        
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        
        for pil_img in pages_images:
            if gemini_api_key:
                # Gemini Path
                gemini_result = utils.extract_with_gemini(pil_img, gemini_api_key)
                if gemini_result:
                    # Map Gemini result to internal structure
                    # We need to wrap it to match what dedupe expects or just use it directly
                    # dedupe expects: {"page_no": ..., "page_type": ..., "raw_rows": ...}
                    # But since Gemini gives us the final structure, we can skip the complex dedupe logic 
                    # or adapt it.
                    # Let's adapt it to return "bill_items" directly in the page result
                    page_results.append({
                        "page_no": str(page_no),
                        "page_type": gemini_result.get("page_type", "Unknown"),
                        "bill_items": gemini_result.get("bill_items", []),
                        "is_gemini": True
                    })
                else:
                    # Fallback if Gemini fails
                    preproc_img, preproc_meta = utils.preprocess_image_stub(pil_img)
                    table_img = utils.simple_segment_table_region(preproc_img)
                    ocr_tokens = utils.ocr_stub(table_img)
                    grouped = utils.cluster_tokens_y(ocr_tokens)
                    rows = utils.extract_rows_from_ocr_stub(grouped)
                    page_results.append(
                        {
                            "page_no": str(page_no),
                            "raw_rows": rows,
                            "preproc_meta": preproc_meta,
                            "ocr_meta": {"token_count": len(ocr_tokens)},
                            "is_gemini": False
                        }
                    )
            else:
                # OCR Path
                preproc_img, preproc_meta = utils.preprocess_image_stub(pil_img)
                table_img = utils.simple_segment_table_region(preproc_img)
                ocr_tokens = utils.ocr_stub(table_img)
                grouped = utils.cluster_tokens_y(ocr_tokens)
                rows = utils.extract_rows_from_ocr_stub(grouped)
                page_results.append(
                    {
                        "page_no": str(page_no),
                        "raw_rows": rows,
                        "preproc_meta": preproc_meta,
                        "ocr_meta": {"token_count": len(ocr_tokens)},
                        "is_gemini": False
                    }
                )
            page_no += 1

        deduped, computed_total, total_item_count = utils.dedupe_and_compute_totals_stub(page_results)
        
        pagewise_line_items = []
        for p in deduped:
            page_items = []
            for it in p.get("bill_items", []):
                page_items.append(
                    {
                        "item_name": it.get("item_name"),
                        "item_amount": float(it.get("item_amount")) if it.get("item_amount") is not None else 0.0,
                        "item_rate": float(it.get("item_rate")) if it.get("item_rate") is not None else 0.0,
                        "item_quantity": float(it.get("item_quantity")) if it.get("item_quantity") is not None else 0.0
                    }
                )
            pagewise_line_items.append(
                {
                    "page_no": str(p.get("page_no")),
                    "page_type": p.get("page_type"),
                    "bill_items": page_items,
                }
            )

        job_meta["is_success"] = True
        job_meta["data"]["pagewise_line_items"] = pagewise_line_items
        job_meta["data"]["total_item_count"] = total_item_count

        return JSONResponse(status_code=200, content=job_meta)
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"is_success": False, "error": str(he.detail)})
    except Exception as e:
        # Log the error for debugging
        logger.error(f"Error processing request: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"is_success": False, "error": str(e)})