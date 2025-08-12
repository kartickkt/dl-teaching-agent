import fitz  # PyMuPDF
import os
import json

def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    all_pages = []

    for i, page in enumerate(doc):
        text = page.get_text("blocks")  # layout-aware
        blocks = sorted(text, key=lambda b: (b[1], b[0]))  # top-to-bottom
        page_text = [b[4].strip() for b in blocks if b[4].strip()]
        all_pages.append({
            "page": i + 1,
            "content": "\n".join(page_text)
        })

    return all_pages

def save_to_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_pdf = os.path.join("data", "d2l.pdf")
    output_json = os.path.join("outputs", "d2l_extracted.json")

    extracted = extract_pdf_text(input_pdf)
    save_to_json(extracted, output_json)
    print(f"✅ Extraction complete. Saved to {output_json}")
