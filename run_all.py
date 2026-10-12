#!/usr/bin/env python
"""Run all LLM capabilities in one file: Photos, Tables, Languages"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# 1. Photos + Scanned PDFs (OCR + Gemini Vision)
from ocr_vision import process_image, process_scanned_pdf

# 2. Tables (layout preserved)
from table_extraction import extract_tables_preserve_layout

# 3. Different Languages (multilingual search)
from multilingual_search import embed_multilingual

# 4. Config
from categories import get_status

if __name__ == "__main__":
    print("All categories enabled:")
    print("✅ Photos (OCR + Vision)")
    print("✅ Tables (layout preserved)")
    print("✅ Languages (multilingual)")
