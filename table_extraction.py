import tabula, pandas as pd

def extract_tables_preserve_layout(pdf_path):
    """Extract tables with column/row layout preserved"""
    dfs = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True, lattice=True)
    structured = []
    for df in dfs:
        structured.append(df.to_dict(orient="records"))
    return structured
