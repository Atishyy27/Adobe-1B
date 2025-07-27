import fitz  # PyMuPDF
import json
import os
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import operator

# --- CONFIGURATION ---
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"
MODEL_DIR = "/app/models/all-MiniLM-L6-v2" # Offline model path

# --- 1. PDF PARSING AND CHUNKING ---

def get_text_chunks(doc_path: str):
    """
    Parses a PDF document, identifies structural headers (H1, H2),
    and extracts text content for each section (chunk).
    This uses font size as a primary heuristic for identifying headers.
    """
    chunks = []
    try:
        doc = fitz.open(doc_path)
        doc_name = os.path.basename(doc_path)
        
        # Heuristic: Identify the most common font sizes to distinguish body from headers
        font_counts = {}
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            size = round(span["size"])
                            font_counts[size] = font_counts.get(size, 0) + 1
        
        if not font_counts: return []
        sorted_fonts = sorted(font_counts.items(), key=operator.itemgetter(1), reverse=True)
        body_font_size = sorted_fonts[0][0] if sorted_fonts else 10

        # Heuristic: Define header sizes relative to the body text
        h1_size = body_font_size * 1.5
        h2_size = body_font_size * 1.25

        current_h1 = None
        current_h2 = None
        current_text = ""
        current_page = 1

        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            size = round(span["size"])

                            is_h1 = size >= h1_size
                            is_h2 = size >= h2_size and size < h1_size

                            if is_h1 or is_h2:
                                # If we encounter a new header, save the previous chunk
                                if current_text.strip():
                                    title = current_h2 if current_h2 else current_h1
                                    chunks.append({
                                        "doc_name": doc_name,
                                        "page": current_page,
                                        "section_title": title if title else "Introduction",
                                        "content": re.sub(r'\s+', ' ', current_text).strip()
                                    })
                                current_text = "" # Reset text for the new section
                            
                            # Update current headers
                            if is_h1:
                                current_h1 = text
                                current_h2 = None # Reset H2 when a new H1 is found
                                current_page = page_num
                            elif is_h2:
                                current_h2 = text
                                current_page = page_num

                            # Append text to the current chunk
                            if not is_h1 and not is_h2:
                                current_text += text + " "
        
        # Add the last collected chunk
        if current_text.strip():
            title = current_h2 if current_h2 else current_h1
            chunks.append({
                "doc_name": doc_name,
                "page": current_page,
                "section_title": title if title else "Conclusion",
                "content": re.sub(r'\s+', ' ', current_text).strip()
            })

    except Exception as e:
        print(f"Error processing {doc_path}: {e}")
        
    return chunks

# --- 2. SEMANTIC ANALYSIS ---

def get_query_from_prompt(prompt_file: str):
    """Reads persona and JTBD from a JSON file and combines them into a query."""
    with open(prompt_file, 'r') as f:
        prompt_data = json.load(f)
    persona = prompt_data["persona"]
    job_to_be_done = prompt_data["job_to_be_done"]
    query = f"As a {persona}, I need to {job_to_be_done}"
    return query, prompt_data

def rank_chunks_by_relevance(query: str, chunks: list, model):
    """Encodes query and chunks, then ranks chunks based on cosine similarity."""
    query_embedding = model.encode([query])
    chunk_contents = [chunk["content"] for chunk in chunks]
    chunk_embeddings = model.encode(chunk_contents)
    
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    
    ranked_chunks = sorted(zip(chunks, similarities), key=lambda x: x[1], reverse=True)
    return ranked_chunks

# --- 3. SUB-SECTION ANALYSIS ---

def get_refined_text(section_content: str, query_embedding, model, num_sentences=3):
    """Performs extractive summarization to find the most relevant sentences."""
    # Simple sentence splitting
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', section_content)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return section_content[:500] # Fallback for short content

    sentence_embeddings = model.encode(sentences)
    similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]

    ranked_sentences = sorted(zip(sentences, similarities), key=lambda x: x[1], reverse=True)
    
    # Get top N sentences and join them
    top_sentences = [s[0] for s in ranked_sentences[:num_sentences]]
    return " ".join(top_sentences)

# --- MAIN ORCHESTRATION ---

if __name__ == "__main__":
    # Load the pre-downloaded sentence transformer model
    print("Loading semantic model...")
    model = SentenceTransformer(MODEL_DIR)
    
    # Find input files
    # Find input files by searching all subdirectories
    pdf_files = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    prompt_file = os.path.join(INPUT_DIR, "prompt.json")

    if not pdf_files or not os.path.exists(prompt_file):
        print("Error: Ensure PDFs and 'prompt.json' are in the /app/input directory.")
        exit(1)

    # 1. Formulate Query
    query, prompt_data = get_query_from_prompt(prompt_file)
    print(f"Query formulated: {query}")

    # 2. Parse all PDFs and create a single list of chunks
    print("Parsing PDFs...")
    all_chunks = []
    for pdf_path in pdf_files:
        all_chunks.extend(get_text_chunks(pdf_path))
    
    if not all_chunks:
        print("Could not extract any text chunks from the documents.")
        exit(1)

    # 3. Rank all chunks from all documents
    print("Ranking sections by relevance...")
    ranked_results = rank_chunks_by_relevance(query, all_chunks, model)
    query_embedding = model.encode([query]) # Re-use for sub-section analysis

    # 4. Prepare JSON output
    output_data = {
        "metadata": {
            "input_documents": [os.path.basename(f) for f in pdf_files],
            "persona": prompt_data["persona"],
            "job_to_be_done": prompt_data["job_to_be_done"],
            "processing_timestamp": datetime.utcnow().isoformat() + "Z"
        },
        "extracted_sections": [],
        "sub_section_analysis": []
    }

    # Process top N results for the final output (e.g., top 10)
    print("Generating final analysis...")
    for i, (chunk, score) in enumerate(ranked_results[:10]):
        rank = i + 1
        
        # Add to extracted_sections
        output_data["extracted_sections"].append({
            "document": chunk["doc_name"],
            "page_number": chunk["page"],
            "section_title": chunk["section_title"],
            "importance_rank": rank
        })

        # Generate refined text for sub-section analysis
        refined_text = get_refined_text(chunk["content"], query_embedding, model)
        output_data["sub_section_analysis"].append({
            "document": chunk["doc_name"],
            "page_number": chunk["page"],
            "refined_text": refined_text
        })
    
    # 5. Write output to file
    output_filename = f"{os.path.splitext(os.path.basename(pdf_files[0]))[0]}_analysis.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Processing complete. Output written to {output_path}")