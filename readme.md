# Persona-Driven Document Intelligence - Adobe Hackathon 2025 (Round 1B)

## 🚀 Project Overview
This project is our solution for Round 1B of the Adobe India Hackathon 2025: "Connecting the Dots." Moving beyond the structural parsing of Round 1A, this challenge required us to build an intelligent system that understands context and relevance.

Our solution functions as an **intelligent document analyst**. It ingests a collection of PDF documents and, based on a specific user `Persona` and their `Job-to-be-Done`, it semantically searches, identifies, and ranks the most relevant sections. The system is built to be fast, efficient, and fully offline, running entirely within a Docker container.

## ✨ Core Features
* **Persona-Based Semantic Analysis:** The core of our engine. It understands natural language prompts to find what truly matters to a specific user.
* **High-Relevance Ranking:** Uses a powerful sentence-transformer model to rank document sections by semantic similarity, ensuring the most important information surfaces first.
* **Multi-Document Processing:** Seamlessly processes and analyzes a collection of 3-10 PDFs at once, connecting ideas across an entire library.
* **Targeted Extractive Summarization:** For top-ranked sections, the system generates concise, "refined text" summaries that get straight to the point.
* **Fully Dockerized & Offline:** The entire application is containerized for portability and meets all offline execution constraints of the hackathon.

## 🧠 Our Approach: The Semantic Analysis Pipeline
Our solution is built on a robust four-stage pipeline designed to deconstruct the user's needs and map them to the most relevant content in the document library.

#### Stage 1: PDF Parsing & Content Chunking
The foundation of the system. It intelligently parses each PDF to extract not just text, but meaningful sections.
* **How it works:** Using `PyMuPDF`, the engine analyzes the typographical structure of each document (font sizes, weights) to identify headings. It then extracts the text content associated with each heading, creating a "chunk" that represents a logical section.

#### Stage 2: Semantic Embedding
This is where the machine "understands" the language.
* **How it works:** We use a pre-trained `all-MiniLM-L6-v2` sentence-transformer model. This powerful yet lightweight model converts both the user's query (Persona + Job) and every content chunk into high-dimensional vectors (embeddings).

#### Stage 3: Relevance Ranking
With the text converted to vectors, we can mathematically find the best matches.
* **How it works:** The system calculates the **cosine similarity** between the user's query vector and every chunk vector. The higher the similarity score, the more semantically relevant the chunk is to the user's request. This score is used to create the final `importance_rank`.

#### Stage 4: Targeted Summarization
To provide quick insights, the final stage generates a summary of the most important sections.
* **How it works:** Instead of a generic summary, our engine performs targeted extraction. For a top-ranked chunk, it re-uses the semantic model to find the top 3-5 sentences within that chunk that are most similar to the user's original query, producing a highly relevant `refined_text` output.

## 💻 Technology Stack
* **Core Logic:** Python 3.9
* **PDF Parsing:** PyMuPDF (fitz)
* **Semantic Analysis:** `sentence-transformers` (`all-MiniLM-L6-v2`), `scikit-learn`, `numpy`
* **ML Framework:** PyTorch (CPU version)
* **Containerization:** Docker

## 📂 Project Structure
```
.
├── models/             # Stores the pre-trained model files (for offline use)
├── input/              # Input PDFs and prompt.json are placed here
├── output/             # Generated JSON analysis is saved here
├── main.py             # Main script orchestrating the entire pipeline
├── models.py           # Script to download the model for offline use
├── Dockerfile          # Instructions to build the Docker container
└── requirements.txt    # Python dependencies
```

## ⚙️ Setup and Execution

### Prerequisites
* Docker Desktop installed and running.
* Git for version control.

### 1. Model Setup (One-Time Manual Step)
Our system uses a pre-trained model which must be downloaded before building the Docker image.

Run the `models.py` script from your terminal:
```bash
python models.py
```
This will download the `all-MiniLM-L6-v2` model and save it in the `models` folder.

### 2. Prepare the Input
1.  Place all your PDF documents inside the `input` folder (or its subdirectories).
2.  Create a `prompt.json` file in the `input` folder with the following structure:
    ```json
    {
      "persona": "Your Persona Here",
      "job_to_be_done": "The task you want to accomplish here."
    }
    ```

### 3. Build the Docker Image
This command packages the application and all its dependencies.
```bash
docker build --platform linux/amd64 -t adobe-1b-solution .
```

### 4. Run the Application
This command will start the container. It automatically processes all PDFs in the `input` folder and saves the JSON result in the `output` folder.

**For Windows (PowerShell):**
```powershell
docker run --rm -v ${PWD}/input:/app/input -v ${PWD}/output:/app/output adobe-1b-solution
```

**For macOS/Linux:**
```bash
docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" adobe-1b-solution
```

## 🧑‍💻 Team Members
* Atishay Jain
* Dilpreet Gill
* Saloni Jain