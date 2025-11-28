import httpx
import fitz
import io
from pdf2image import convert_from_bytes
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from utils import (
    compute_md5,
    extract_text_from_pdf_bytes,
    images_from_pdf_bytes,
    preprocess_image_stub,
    ocr_stub,
    extract_rows_from_ocr_stub,
    dedupe_and_compute_totals_stub,
    simple_segment_table_region,
    cluster_tokens_y,
)

app = FastAPI(title="Datathon Bill Extractor")

class ExtractRequest(BaseModel):
    document: HttpUrl

@app.post("/extract-bill-data")
async def extract_bill_data(file: UploadFile = File(...)):
    job_meta = {
        "is_success": False,
        "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
        "data": {"pagewise_line_items": [], "total_item_count": 0},
        "warnings": [],
    }
    try:
        content = await file.read()
        content_type = file.content_type
        md5 = compute_md5(content)
        if not content:
            raise HTTPException(status_code=400, detail="Unable to download document")
        
        pages_images = []
        embedded_text = None
        if content_type == "application/pdf":
            embedded_text = extract_text_from_pdf_bytes(content)
            pages_images = images_from_pdf_bytes(content, dpi=300)
        else:
            pil = Image.open(io.BytesIO(content)).convert("RGB")
            pages_images = [pil]
        
        page_results = []
        page_no = 1
        for pil_img in pages_images:
            preproc_img, preproc_meta = preprocess_image_stub(pil_img)
            table_img = simple_segment_table_region(preproc_img)
            ocr_tokens = ocr_stub(table_img)
            grouped = cluster_tokens_y(ocr_tokens)
            rows = extract_rows_from_ocr_stub(grouped)
            page_results.append(
                {
                    "page_no": str(page_no),
                    "page_type": "Unknown",
                    "raw_rows": rows,
                    "preproc_meta": preproc_meta,
                    "ocr_meta": {"token_count": len(ocr_tokens)},
                }
            )
            page_no += 1

        deduped, computed_total, total_item_count = dedupe_and_compute_totals_stub(page_results)
        pagewise_line_items = []
        for p in deduped:
            page_items = []
            for it in p.get("bill_items", []):
                page_items.append(
                    {
                        "item_name": it.get("item_name"),
                        "item_amount": it.get("item_amount") if it.get("item_amount") is not None else None,
                        "item_rate": float(it.get("item_rate")) if it.get("item_rate") is not None else None,
                        "item_quantity": float(it.get("item_quantity")) if it.get("item_quantity") is not None else None
                    }
                )
            pagewise_line_items.append(
                {
                    "page_no": p.get("page_no"),
                    "page_type": p.get("page_type"),
                    "line_items": page_items,
                }
            )

        job_meta["is_success"] = True
        job_meta["data"]["pagewise_line_items"] = pagewise_line_items
        job_meta["data"]["total_item_count"] = total_item_count
        job_meta["computed_total"] = computed_total
        job_meta["md5"] = md5

        return JSONResponse(status_code=200, content=job_meta)
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"is_success": False, "error": str(he.detail)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"is_success": False, "error": str(e)})