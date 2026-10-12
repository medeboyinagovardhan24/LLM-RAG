# LLM Project - Category Configuration

This project organizes different document/evidence types with their processing capabilities.

## Categories Overview

### 1. Direct Image Uploads (.jpg/.png)
**Status:** ❌ No (Only .pdf files are permitted)

### 2. Scanned PDFs / Photos in PDFs  
**Status:** ❌ No (Ignored by pypdf; requires OCR or Gemini Vision)

### 3. Tables
**Status:** ⚠️ Partial (Reads text from digital tables, but layout/columns may get flattened)

### 4. Different Languages (LLM)
**Status:** ✅ Yes (Gemini 2.0 Flash is natively multilingual)

### 5. Different Languages (Search)
**Status:** ⚠️ Partial (ChromaDB uses all-MiniLM-L6-v2, best on English)

## Configuration Notes

- Categories with ✅ Yes status can be used as-is
- Categories with ⚠️ Partial status have limitations that need to be considered
- Categories with ❌ No status are not supported and need alternative approaches

## Next Steps

Consider implementing alternative solutions for unsupported categories:
- For images: Convert to PDF format
- For scanned PDFs: Add OCR processing
- For partial support: Document the limitations clearly