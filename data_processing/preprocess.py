# This script is for preprocessing financial documents, specifically Apple 10-K filings.
# using PyMuPDF and pdfplumber for text extraction and cleaning.
# using regex for text cleaning and formatting.
# using fitz for PDF handling and pdfplumber for table extraction.

import fitz  # PyMuPDF
import re
import os
import pdfplumber

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text (non-tabular) from a PDF file."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text")
    return full_text

def extract_tables_from_pdf(pdf_path: str) -> str:
    """Extract tables from PDF and convert them into readable text."""
    table_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            tables = page.extract_tables()
            for table in tables:
                # Flatten table rows into readable lines
                for row in table:
                    clean_row = [cell.strip() if cell else "" for cell in row]
                    line = " | ".join(clean_row)
                    table_text += line + "\n"
                table_text += "\n"
    return table_text

def clean_text(text: str) -> str:
    """Remove headers, footers, page numbers, but PRESERVE FINANCIAL DATA"""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if re.match(r'^\s*Page \d+\s*$', line):
            continue
        if len(line) > 1:
            cleaned_lines.append(line)
    
    cleaned_text = ' '.join(cleaned_lines)
    
    # regex to handle spaces around dollar amounts
    # It looks for a dollar sign, followed by optional spaces, then the number
    cleaned_text = re.sub(r'(\$\s*[\d,]+\s*(?:\.\s*\d+)?\s*(?:[Bb]illion|[Mm]illion|[Tt]housand)?)', r' \1 ', cleaned_text)
    
    # Also handle percentages
    cleaned_text = re.sub(r'(\d+(?:\.\d+)?\s*%)', r' \1 ', cleaned_text)
    
    # Now collapse other multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text

def save_text(text: str, output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

if __name__ == "__main__":
    # Process both years
    for year in [2022, 2023]:
        pdf_path = f"data/raw/apple_10k_{year}.pdf"
        output_path = f"data/processed/cleaned_apple_10k_{year}.txt"

        # Extract narrative text
        narrative_text = extract_text_from_pdf(pdf_path)
        narrative_cleaned = clean_text(narrative_text)

        # Extract tabular data
        tables_text = extract_tables_from_pdf(pdf_path)

        # Merge everything
        final_text = narrative_cleaned + "\n\n" + "=== EXTRACTED TABLES ===\n" + tables_text

        # Save
        save_text(final_text, output_path)
        print(f"Cleaned text for {year} saved to {output_path}")