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

paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')

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

def images_from_pdf_bytes(pdf_bytes: bytes, dpi: int = 300) ->List[Image.Image]:
    pil_pages = convert_from_bytes(pdf_bytes, dpi=dpi)
    return [p.convert("RGB") for p in pil_pages]

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
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    
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

def extract_rows_from_ocr_stub(ocr_tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    money_re = re.compile(r"([0-9]{1,3}(?:[,0-9]{0,3})*(?:\.[0-9]+)?)")
    for t in ocr_tokens:
        text = t["text"]
        parts = t["text"]
        parts = text.split()
        money_matches = money_re.findall(text.replace(",", ""))
        amount = None
        quantity = None
        rate = None
        if money_matches:
            amount = float(money_matches[-1])
        if "x" in parts:
            try:
                idx = parts.index("x")
                q = parts[idx - 1]
                r = parts[idx + 1]
                quantity = float(q)
                rate = float(r)
            except Exception:
                pass
        item_name = text
        rows.append(
            {
                "raw_text": text,
                "item_name": item_name,
                "item_quantity": quantity,
                "item_rate": rate,
                "item_amount": amount,
                "bbox": t.get("bbox"),
                "conf": t.get("conf", 0.0),
            }
        )
    return rows

def dedupe_and_compute_totals_stub(page_results: List[Dict[str, Any]]):

    def is_summary(text: str):
        t = text.lower()
        return any(k in t for k in ["total", "subtotal", "net payable", "grand total", "amount due"])
    
    deduped_pages = []
    total = 0.0
    item_count = 0
    for p in page_results:
        bill_items = []
        for r in p.get("raw_rows", []):
            if r.get("item_name") and is_summary(r.get("raw_text", "")):
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
        deduped_pages.append({"page_no": p.get("page_no"), "page_type": p.get("page_type", "Unknown"), "bill_items": bill_items})
    return deduped_pages, total, item_count
