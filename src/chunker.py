# This script is for chunking the text segments into smaller, manageable pieces
# using a sliding window approach with overlap to maintain context.


import os
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from tqdm import tqdm

# Ensure 'punkt' is downloaded for word_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading 'punkt' NLTK data...")
    nltk.download('punkt', quiet=True)


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list:
    """
    Splits text into chunks using a sliding window approach.
    Each chunk will overlap with the next one.
    """
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than the chunk size.")

    tokens = word_tokenize(text)
    chunks = []
    # the loop to create a sliding window
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(' '.join(chunk))
    return chunks

def process_segments_directory() -> pd.DataFrame:
    """Process all segment files from the segments directory"""
    input_dir = "data/processed/segments"
    all_chunks = []

    for filename in tqdm(os.listdir(input_dir), desc="Chunking segments"):
        if not filename.endswith(".txt"):
            continue

        section_name = filename.replace(".txt", "")
        file_path = os.path.join(input_dir, filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Extract year from filename
        year = section_name.split('_')[-1]
        
        # If it's a financial section → larger chunks
        financial_sections = ["income_statement", "balance_sheet", "cash_flow", "equity", "financial_highlights"]
        is_financial = any(section in section_name for section in financial_sections)
        
        if is_financial:
            # Using sliding window with 400-token chunks and 50-token overlap
            chunks = chunk_text(text, chunk_size=400, overlap=50)
        else:
            # Using sliding window with 150-token chunks and 30-token overlap
            chunks = chunk_text(text, chunk_size=150, overlap=30)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "section": section_name,
                "chunk_id": f"{section_name}_{i}",
                "text": chunk,
                "year": year,
                "is_financial": 1 if is_financial else 0,
                "word_count": len(word_tokenize(chunk))
            })

    return pd.DataFrame(all_chunks)

if __name__ == "__main__":
    input_dir = "data/processed/segments"
    output_path = "data/processed/chunks.csv"

    # Verify segments directory exists
    if not os.path.exists(input_dir):
        print("Segments directory not found. Run segmenter.py first.")
    else:
        df_chunks = process_segments_directory()
        df_chunks.to_csv(output_path, index=False)
        print(f"Chunked data saved to {output_path}")
        print(f"Total chunks: {len(df_chunks)}")
        print(f"Financial chunks: {df_chunks['is_financial'].sum()}")



#older versions

# import os
# import nltk
# from nltk.tokenize import word_tokenize
# import pandas as pd
# from tqdm import tqdm

# # nltk.download('punkt')

# def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list:
#     tokens = word_tokenize(text)
#     chunks = []
#     for i in range(0, len(tokens), chunk_size - overlap):
#         chunk = tokens[i:i + chunk_size]
#         chunks.append(' '.join(chunk))
#     return chunks

# def process_segments_directory() -> pd.DataFrame:
#     """Process all segment files from the segments directory"""
#     input_dir = "data/processed/segments"
#     all_chunks = []

#     for filename in tqdm(os.listdir(input_dir), desc="Chunking segments"):
#         if not filename.endswith(".txt"):
#             continue

#         section_name = filename.replace(".txt", "")
#         file_path = os.path.join(input_dir, filename)

#         with open(file_path, 'r', encoding='utf-8') as f:
#             text = f.read()

#         # Extract year from filename
#         year = section_name.split('_')[-1]
        
#         # If it's a financial section → larger chunks
#         financial_sections = ["income_statement", "balance_sheet", "cash_flow", "equity", "financial_highlights"]
#         is_financial = any(section in section_name for section in financial_sections)
        
#         if is_financial:
#             chunks = chunk_text(text, chunk_size=400, overlap=50)
#         else:
#             chunks = chunk_text(text, chunk_size=150, overlap=30)

#         for i, chunk in enumerate(chunks):
#             all_chunks.append({
#                 "section": section_name,
#                 "chunk_id": f"{section_name}_{i}",
#                 "text": chunk,
#                 "year": year,
#                 "is_financial": 1 if is_financial else 0,
#                 "word_count": len(word_tokenize(chunk))
#             })

#     return pd.DataFrame(all_chunks)

# if __name__ == "__main__":
#     input_dir = "data/processed/segments"
#     output_path = "data/processed/chunks.csv"

#     # Verify segments directory exists
#     if not os.path.exists(input_dir):
#         print("Segments directory not found. Run segmenter.py first.")
#     else:
#         df_chunks = process_segments_directory()
#         df_chunks.to_csv(output_path, index=False)
#         print(f"Chunked data saved to {output_path}")
#         print(f"Total chunks: {len(df_chunks)}")
#         print(f"Financial chunks: {df_chunks['is_financial'].sum()}")



# import os
# import nltk
# from nltk.tokenize import word_tokenize
# import pandas as pd
# from tqdm import tqdm
# from segmenter import extract_tables_from_html, load_text, segment_text_by_headings

# # nltk.download('punkt')

# def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list:
#     """Chunks a block of text based on word count with overlap."""
#     tokens = word_tokenize(text)
#     chunks = []
#     if len(tokens) <= chunk_size:
#         chunks.append(' '.join(tokens))
#     else:
#         for i in range(0, len(tokens), chunk_size - overlap):
#             chunk = tokens[i:i + chunk_size]
#             chunks.append(' '.join(chunk))
#     return chunks

# def process_data_to_chunks(input_text_path: str, input_html_path: str) -> pd.DataFrame:
#     """Processes both text segments and HTML tables into a DataFrame of chunks."""
#     all_chunks = []

#     # Step 1: Process the text segments
#     if os.path.exists(input_text_path):
#         text = load_text(input_text_path)
#         segments = segment_text_by_headings(text)

#         for section_name, content in tqdm(segments.items(), desc="Chunking text segments"):
#             if section_name == "source_type":
#                 continue
            
#             chunks = chunk_text(content)
#             for i, chunk in enumerate(chunks):
#                 all_chunks.append({
#                     "section": section_name,
#                     "chunk_id": f"{section_name}_{i}",
#                     "text": chunk
#                 })

#     # Step 2: Process the HTML tables and add them as chunks
#     tables_data = extract_tables_from_html(input_html_path)
#     for table_info in tqdm(tables_data, desc="Processing tables"):
#         # The table is saved as a single, descriptive chunk
#         all_chunks.append({
#             "section": table_info["section"],
#             "chunk_id": f"{table_info['section']}_table",
#             "text": table_info["text"]
#         })

#     return pd.DataFrame(all_chunks)

# if __name__ == "__main__":
#     input_text_path = "data/processed/cleaned_apple_10k_2023.txt"
#     input_html_path = "data/raw/aapl-20230930.htm"
#     output_path = "data/processed/chunks.csv"

#     # Make sure you have the necessary data files
#     if not os.path.exists(input_text_path) or not os.path.exists(input_html_path):
#         print("Required input files not found. Please ensure both a cleaned text file and the raw HTML file of the 10-K are in the specified directories.")
#     else:
#         df_chunks = process_data_to_chunks(input_text_path, input_html_path)
#         df_chunks.to_csv(output_path, index=False)
#         print(f"Chunked data saved to {output_path}")


# 

