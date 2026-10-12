categories = {
    "direct_image_uploads": "yes",
    "scanned_pdfs_photos": "yes",
    "tables": "yes",
    "different_languages_llm": "yes",
    "different_languages_search": "yes"
}

def get_status(name):
    return categories.get(name, "unknown")
