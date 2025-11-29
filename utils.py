import httpx
import hashlib
import fitz
import cv2
import re
import numpy as np
from typing import Any, Tuple, List, Dict
from PIL import Image
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR
from difflib import SequenceMatcher

import google.generativeai as genai
import os
import json

paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')

def extract_with_gemini(pil_img: Image.Image, api_key: str) -> Dict[str, Any]:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = """
    You are an expert data extraction agent. Extract the line items from this bill image.
    Return the output in the following JSON format ONLY. Do not include markdown formatting like ```json ... ```.
    
    {
        "page_type": "Bill Detail | Final Bill | Pharmacy",
        "bill_items": [
            {
                "item_name": "string", // Exactly as mentioned in the bill
                "item_amount": "float", // Net Amount of the item post discounts
                "item_rate": "float", // Exactly as mentioned in the bill
                "item_quantity": "float" // Exactly as mentioned in the bill
            }
        ]
    }
    
    Rules:
    1. Extract all line items accurately.
    2. Do not double count.
    3. If a field is missing, use 0.0 for numbers and "" for strings.
    4. "page_type" should be one of the specified values based on content.
    5. Exclude "Sub-total" and "Final Total" rows from "bill_items".
    """
    
    try:
        response = model.generate_content([prompt, pil_img])
        text = response.text
        # Clean up markdown if present
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text)
    except Exception as e:
        print(f"Gemini extraction failed: {e}")
        return None

def is_summary_row(text: str):
    t = text.lower()
    return any(k in t for k in ["total", "subtotal", "grand total", "bill amount", "net amount"])

def fuzzy_ratio(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def dedupe_minimal(items, threshold=0.85):
    final = []
    for it in items:
        is_dup = False
        for f in final:
            if fuzzy_ratio(it.get("item_name"), f.get("item_name")) > threshold:
                is_dup = True
                break
        if not is_dup:
            final.append(it)
    return final

def is_valid_numeric(item):
    q = item.get("item_quantity")
    r = item.get("item_rate")
    a = item.get("item_amount")
    if q is None or r is None or a is None:
        return True

    expected = q * r
    if expected == 0:
        return True

    diff = abs(expected - a) / expected
    return diff < 0.10     

def compute_md5(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()
    
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            texts.append(text)
    return "\n".join(texts)

def images_from_pdf_bytes(pdf_bytes: bytes, dpi: int = 300) -> List[Image.Image]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        # Determine mode based on number of components (n)
        # 1: Gray, 2: Gray+Alpha, 3: RGB, 4: RGB+Alpha
        if pix.n < 3:
            mode = "L"
        else:
            mode = "RGB"
        
        # If alpha is present, we might need to handle it, but for now let's try basic mapping
        # fitz pixmap samples are bytes.
        # If we have 4 channels, it's likely RGBA.
        if pix.n == 4:
            mode = "RGBA"
        
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        if mode != "RGB":
            img = img.convert("RGB")
            
        images.append(img)
    return images

def preprocess_image_stub(pil_img: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    h, w = img.shape
    if h > 1280:
        scale = 1280 / h
        img = cv2.resize(img, (int(w * scale), 1280), interpolation=cv2.INTER_AREA)
    
    img = cv2.fastNlMeansDenoising(img, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img = clahe.apply(img)

    pil_out = Image.fromarray(img)
    return pil_out, {"preproc": "resize+denoise+clahe"}

def simple_segment_table_region(pil_img):
    img = np.array(pil_img)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    projection = np.sum(img < 240, axis=1)
    threshold = np.max(projection) * 0.2
    
    mask = projection > threshold
    idx = np.where(mask)[0]

    if len(idx) == 0:
        return pil_img
    
    top, bottom = int(idx[0]), int(idx[-1])
    table = pil_img.crop((0, top, pil_img.width, bottom))
    return table

def ocr_stub(pil_img: Image.Image) -> List[Dict[str, Any]]:
    result = paddle_ocr.ocr(np.array(pil_img), cls=True)
    tokens = []

    for line in result:
        for (bbox, (text, conf)) in line:
            tokens.append({
                "text": text,
                "bbox": bbox,
                "conf": conf
            })
        
    return tokens

def cluster_tokens_y(tokens, threshold=20):
    rows = []
    tokens = sorted(tokens, key=lambda t: sum(p[1] for p in t["bbox"]) / 4)

    current = [tokens[0]]
    prev_y = sum(p[1] for p in tokens[0]["bbox"]) / 4

    for t in tokens[1:]:
        y = sum(p[1] for p in t["bbox"]) / 4
        if abs(y - prev_y) < threshold:
            current.append(t)
        else:
            rows.append(current)
            current = [t]
        prev_y = y
    
    rows.append(current)
    return rows

def extract_rows_from_ocr_stub(grouped_rows: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    money_re = re.compile(r"([0-9]{1,3}(?:[,0-9]{0,3})*(?:\.[0-9]+)?)")
    
    for row_tokens in grouped_rows:
        # Sort tokens by X coordinate (left to right)
        row_tokens = sorted(row_tokens, key=lambda t: t["bbox"][0][0])
        
        # Join text
        full_text = " ".join([t["text"] for t in row_tokens])
        
        # Simple extraction logic
        # 1. Find all money-like patterns
        money_matches = money_re.findall(full_text.replace(",", ""))
        
        amount = None
        quantity = None
        rate = None
        
        # Heuristic: The last number is often the amount
        if money_matches:
            try:
                amount = float(money_matches[-1])
            except ValueError:
                pass
                
        # Heuristic: Look for Quantity and Rate
        # This is hard without column info, but let's try some patterns
        # e.g. "Item Name 2 x 100.00 200.00"
        # or "Item Name 2 100.00 200.00"
        
        # If we have at least 3 numbers, maybe Qty, Rate, Amount?
        # Or if we have "x"
        parts = full_text.split()
        if "x" in parts:
            try:
                idx = parts.index("x")
                if idx > 0 and idx < len(parts) - 1:
                    q_str = parts[idx - 1]
                    r_str = parts[idx + 1]
                    # Clean strings
                    q_str = re.sub(r"[^0-9.]", "", q_str)
                    r_str = re.sub(r"[^0-9.]", "", r_str)
                    if q_str: quantity = float(q_str)
                    if r_str: rate = float(r_str)
            except Exception:
                pass
        
        # If we didn't find via 'x', try to infer from numbers if we have amount
        if amount is not None and len(money_matches) >= 2:
            # Maybe the second to last is rate?
            try:
                candidate_rate = float(money_matches[-2])
                if candidate_rate > 0:
                    # check if amount / rate is an integer-ish quantity
                    calc_qty = amount / candidate_rate
                    if abs(calc_qty - round(calc_qty)) < 0.01:
                        quantity = round(calc_qty)
                        rate = candidate_rate
            except Exception:
                pass

        # Item name is the text minus the numbers at the end? 
        # Or just the full text for now, maybe stripping the amount
        item_name = full_text
        
        # Calculate bbox of the row (min x, min y, max x, max y)
        # But we need 4 points. Let's just take the union of bboxes.
        # Simplified: just pass the first token's bbox or None
        bbox = row_tokens[0]["bbox"] if row_tokens else None
        conf = sum(t["conf"] for t in row_tokens) / len(row_tokens) if row_tokens else 0.0

        rows.append(
            {
                "raw_text": full_text,
                "item_name": item_name,
                "item_quantity": quantity,
                "item_rate": rate,
                "item_amount": amount,
                "bbox": bbox,
                "conf": conf,
            }
        )
    return rows

def download_file(url: str) -> bytes:
    with httpx.Client() as client:
        resp = client.get(url, timeout=30.0)
        resp.raise_for_status()
        return resp.content

def classify_page_type(text: str) -> str:
    t = text.lower()
    if "pharmacy" in t or "medicine" in t or "tablet" in t:
        return "Pharmacy"
    elif "final bill" in t or "summary" in t or "abstract" in t:
        return "Final Bill"
    elif "detail" in t or "particulars" in t:
        return "Bill Detail"
    return "Bill Detail" # Default fallback

def dedupe_and_compute_totals_stub(page_results: List[Dict[str, Any]]):
    deduped_pages = []
    total = 0.0
    item_count = 0
    
    for p in page_results:
        # If Gemini already extracted items, use them directly
        if p.get("is_gemini") and "bill_items" in p:
            deduped_pages.append({
                "page_no": p.get("page_no"),
                "page_type": p.get("page_type"),
                "bill_items": p.get("bill_items")
            })
            # Add to totals
            for item in p.get("bill_items", []):
                try:
                    total += float(item.get("item_amount", 0.0))
                    item_count += 1
                except (ValueError, TypeError):
                    pass
            continue

        # Determine page type based on all text in the page
        # We might need the full text for this, but we only have rows here.
        # Let's reconstruct or check raw_rows.
        full_text = " ".join([r.get("raw_text", "") for r in p.get("raw_rows", [])])
        page_type = classify_page_type(full_text)
        
        bill_items = []
        for r in p.get("raw_rows", []):
            # Filter out obvious headers/footers or summary lines
            if is_summary_row(r.get("raw_text", "")):
                continue
            
            # Filter out JSON-like rows (artifacts from training data having expected output)
            raw_text = r.get("raw_text", "")
            t_lower = raw_text.lower()
            if ('"item' in t_lower or '"page' in t_lower or 
                'input_tokens' in t_lower or 'output_tokens' in t_lower or 
                'is_success' in t_lower or 'total_item_count' in t_lower or
                '{' in raw_text or '}' in raw_text or 
                ('":' in raw_text and '"' in raw_text)):
                continue

            amt = r.get("item_amount")
            if amt is None:
                continue
                
            bill_items.append(
                {
                    "item_name": r.get("item_name"),
                    "item_quantity": r.get("item_quantity"),
                    "item_rate": r.get("item_rate"),
                    "item_amount": r.get("item_amount"),
                    "bbox": r.get("bbox"),
                    "conf": r.get("conf"),
                }
            )
            total += float(amt)
            item_count += 1
            
        deduped_pages.append({
            "page_no": p.get("page_no"),
            "page_type": page_type,
            "bill_items": bill_items
        })
        
    return deduped_pages, total, item_count
