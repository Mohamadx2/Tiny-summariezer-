import argparse
import sys
from pathlib import Path

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
MAX_PAGES = 30
CHUNK_TOKENS = 850


def extract_pdf_text(pdf_path: Path, max_pages: int) -> str:
    import pymupdf

    with pymupdf.open(pdf_path) as document:
        if document.page_count == 0:
            raise ValueError("The PDF has no pages.")
        if document.page_count > max_pages:
            raise ValueError(
                f"This PDF has {document.page_count} pages; the limit is {max_pages}."
            )

        pages = [page.get_text().strip() for page in document]

    text = "\n\n".join(page for page in pages if page)
    if not text:
        raise ValueError("No selectable text found. Scanned PDFs need OCR first.")
    return text


def split_into_chunks(text: str, tokenizer, chunk_tokens: int = CHUNK_TOKENS) -> list[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return [
        tokenizer.decode(token_ids[start : start + chunk_tokens], skip_special_tokens=True)
        for start in range(0, len(token_ids), chunk_tokens)
    ]


def summarize_text(text: str, model_name: str = MODEL_NAME) -> str:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    current_text = text
    while True:
        chunks = split_into_chunks(current_text, tokenizer)
        summaries = []
        for chunk in chunks:
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(device)
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=160,
                    min_new_tokens=30,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True,
                )
            summaries.append(
                tokenizer.decode(output[0], skip_special_tokens=True).strip()
            )

        current_text = "\n\n".join(summaries)
        if len(chunks) == 1:
            return current_text


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Summarize a text-based PDF of up to {MAX_PAGES} pages using a free local AI model."
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument(
        "-o", "--output", type=Path, help="Output text file (default: <pdf-name>_summary.txt)"
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        parser.error(f"PDF file not found: {args.pdf}")

    output_path = args.output or args.pdf.with_name(f"{args.pdf.stem}_summary.txt")
    try:
        text = extract_pdf_text(args.pdf, MAX_PAGES)
        print(f"Summarizing {args.pdf.name} with {MODEL_NAME}...", file=sys.stderr)
        summary = summarize_text(text)
        output_path.write_text(summary + "\n", encoding="utf-8")
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        if isinstance(error, ImportError):
            print("Install the required packages with: python -m pip install -r requirements.txt", file=sys.stderr)
            return 1
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Summary saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())