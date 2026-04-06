from pdf2image import pdfinfo_from_path, convert_from_path
import pytesseract
from tqdm import tqdm

pdf_path = "/home/rrao/projects/agents/agentic-reasoning-framework/examples/BAnzAT04072022B1300.pdf"

info = pdfinfo_from_path(pdf_path)
n_pages = info["Pages"]

all_text = []

for page_num in tqdm(range(1, n_pages + 1), desc="Processing pages"):
    page = convert_from_path(
        pdf_path,
        dpi=300,
        first_page=page_num,
        last_page=page_num,
    )[0]

    text = pytesseract.image_to_string(page, lang="deu")
    all_text.append(f"\n--- Page {page_num} ---\n{text}")

result = "\n".join(all_text)

with open("output_deu.txt", "w", encoding="utf-8") as f:
    f.write(result)

print("Done.")