# This script segments financial documents into structured sections based on common headings.

# segmenter.py
import re
import os
from typing import Dict
from unidecode import unidecode
import pandas as pd

def load_cleaned_text(year: int) -> str:
    """Load cleaned text for a specific year"""
    file_path = f"data/processed/cleaned_apple_10k_{year}.txt"
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def segment_text(text: str, year: int) -> Dict[str, str]:
    """Segment text into financial sections"""
    
    sections = {
        "income_statement": r"(consolidated\s+)?(statements|statement)\s+of\s+(operations|income)",
        "balance_sheet": r"(consolidated\s+)?balance\s+sheets?",
        "cash_flow": r"(consolidated\s+)?statements?\s+of\s+cash\s+flows?",
        "equity": r"(consolidated\s+)?statements?\s+of\s+(stockholders'|shareholders')?\s*equity",
        "financial_highlights": r"(selected\s+financial\s+data|financial\s+highlights|summary\s+of\s+operations)",
        "notes": r"notes\s+to\s+(consolidated\s+)?financial\s+statements?",
        "management_discussion": r"management(?:'s)?\s+discussion\s+and\s+analysis",
        "risk_factors": r"risk\s+factors",
        "business_overview": r"item\s*1\.\s*business"
    }

    section_matches = {}
    for key, pattern in sections.items():
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            section_matches[f"{key}_{year}"] = matches[0].start()

    sorted_sections = sorted(section_matches.items(), key=lambda x: x[1])
    
    segmented = {}
    for i in range(len(sorted_sections)):
        name, start = sorted_sections[i]
        end = sorted_sections[i+1][1] if i+1 < len(sorted_sections) else len(text)
        segment_text = text[start:end]
        segment_text = unidecode(segment_text)
        segmented[name] = segment_text.strip()

    return segmented

def save_segments(segments: Dict[str, str]):
    """Save segments to the processed directory"""
    segments_dir = "data/processed/segments"
    os.makedirs(segments_dir, exist_ok=True)
    
    for name, content in segments.items():
        filename = os.path.join(segments_dir, f"{name}.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

def process_all_years(years=[2022, 2023]):
    """Process segmentation for all years"""
    all_segments = {}
    
    for year in years:
        print(f"Segmenting {year}...")
        text = load_cleaned_text(year)
        segments = segment_text(text, year)
        all_segments.update(segments)
        print(f"  Found {len(segments)} sections")
    
    save_segments(all_segments)
    
    # Create segmentation report
    segment_info = []
    for name, content in all_segments.items():
        segment_info.append({
            'segment_name': name,
            'year': name.split('_')[-1],
            'section_type': '_'.join(name.split('_')[:-1]),
            'length_chars': len(content),
            'length_words': len(content.split())
        })
    
    df = pd.DataFrame(segment_info)
    df.to_csv("data/processed/segmentation_report.csv", index=False)
    
    print(f"\nSegmentation complete! Total segments: {len(all_segments)}")
    return all_segments

if __name__ == "__main__":
    segments = process_all_years()
    print("Segment files saved in: data/processed/segments/")







    

#-------------------------------old version-------------------------------

# import re
# import os
# from typing import Dict, List
# from unidecode import unidecode
# import pandas as pd
# from bs4 import BeautifulSoup

# def load_text(file_path: str) -> str:
#     """Loads text from a file, ensuring UTF-8 encoding."""
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return f.read()

# def segment_text_by_headings(text: str) -> Dict[str, str]:
#     """Segments text into sections based on common headings."""
#     # Define common financial section headings as a list of tuples (name, pattern)
#     section_patterns = [
#         ("item1", r"item\s*1\.\s*business"),
#         ("item1a", r"item\s*1a\.\s*risk\s*factors"),
#         ("item7", r"item\s*7\.\s*management(?:’s)?\s*discussion\s*and\s*analysis"),
#         ("item8", r"item\s*8\.\s*financial\s*statements\s*and\s*supplementary\s*data"),
#         ("notes_start", r"notes\s+to\s+(consolidated\s+)?financial\s+statements?")
#     ]

#     section_matches = {}
#     for name, pattern in section_patterns:
#         matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
#         if matches:
#             section_matches[name] = matches[0].start()

#     sorted_sections = sorted(section_matches.items(), key=lambda x: x[1])
    
#     segmented = {}
#     for i in range(len(sorted_sections)):
#         name, start = sorted_sections[i]
#         end = sorted_sections[i+1][1] if i+1 < len(sorted_sections) else len(text)
        
#         # Save the full section content
#         section_content = text[start:end]
#         segmented[name] = unidecode(section_content).strip()

#     return segmented

# def extract_tables_from_html(html_path: str) -> List[Dict[str, str]]:
#     """
#     Extracts tables from an HTML file and converts them to descriptive text.
#     Assumes the source file is an HTML version of the 10-K.
#     """
#     if not os.path.exists(html_path):
#         return []

#     # Try different encodings until one works
#     try:
#         with open(html_path, 'r', encoding='utf-8') as f:
#             soup = BeautifulSoup(f, 'html.parser')
#     except UnicodeDecodeError:
#         try:
#             with open(html_path, 'r', encoding='cp1252') as f:
#                 soup = BeautifulSoup(f, 'html.parser')
#         except UnicodeDecodeError:
#             print("Error: Could not decode the HTML file with 'utf-8' or 'cp1252'. Please check the file's encoding.")
#             return []

#     extracted_tables = []
#     # Common table headings to search for
#     table_titles = [
#         "Consolidated Statements of Operations",
#         "Consolidated Balance Sheets",
#         "Consolidated Statements of Cash Flows"
#     ]

#     for title in table_titles:
#         title_tag = soup.find(string=lambda text: title in text)
#         if title_tag:
#             # Find the parent table and extract its content
#             table = title_tag.find_parent("table")
#             if table:
#                 try:
#                     df_list = pd.read_html(table.prettify())
#                     if df_list:
#                         df = df_list[0].T.reset_index(drop=True).T
#                         df.columns = df.iloc[0]
#                         df = df[1:]
                        
#                         table_text = f"The following is a {title} table:\n"
#                         table_text += df.to_string(index=False, header=True)
                        
#                         extracted_tables.append({
#                             "section": title.replace(" ", "_").lower(),
#                             "text": unidecode(table_text)
#                         })
#                 except ValueError as e:
#                     print(f"Warning: Could not parse table for '{title}'. Error: {e}")
#                     continue

#     return extracted_tables

# if __name__ == "__main__":
#     # Assumes you have saved the raw HTML version of the 10-K
#     input_html_path = "data/raw/aapl-20230930.htm"
#     # Assumes you have a plain text version for regular segmentation
#     input_text_path = "data/processed/cleaned_apple_10k_2023.txt"
#     output_dir = "data/processed/segments"

#     # Step 1: Extract and convert tables from HTML
#     tables_data = extract_tables_from_html(input_html_path)

#     # Step 2: Segment the plain text file
#     if os.path.exists(input_text_path):
#         text = load_text(input_text_path)
#         segments = segment_text_by_headings(text)
        
#         # Add a flag to indicate if text-based segmentation was performed
#         segments['source_type'] = 'text'

#         # Save the segmented text to files
#         os.makedirs(output_dir, exist_ok=True)
#         for name, content in segments.items():
#             filename = os.path.join(output_dir, f"{name}.txt")
#             with open(filename, 'w', encoding='utf-8') as f:
#                 f.write(content)
#     else:
#         print(f"Warning: {input_text_path} not found. Skipping text segmentation.")
#         segments = {}

#     print(f"Segmented sections saved to {output_dir}")
#     print(f"Extracted {len(tables_data)} tables from HTML. This data will be passed to the chunking script.")

#     # The `tables_data` object will now be used by the chunking script.





