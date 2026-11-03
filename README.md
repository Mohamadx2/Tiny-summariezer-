# Tiny Summarizer

Summarize text-based PDF files locally with a free Hugging Face model. No API key
or paid service is needed. The first run downloads the model (about 1.2 GB); later
runs use the cached copy. CPU works, though a GPU is faster.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Usage

The app accepts PDFs up to 30 pages and saves the summary beside the source PDF:

```bash
python summarizer.py path/to/document.pdf
```

Choose a different output path with `-o`:

```bash
python summarizer.py path/to/document.pdf -o summary.txt
```

The model is `sshleifer/distilbart-cnn-12-6`, downloaded from Hugging Face and
run on your machine. PDFs containing only scanned images are not supported; run
OCR on them first to create selectable text.
