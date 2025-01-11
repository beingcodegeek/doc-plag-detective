import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from pptx import Presentation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import re
from nltk.corpus import stopwords
import nltk
import pandas as pd
import zipfile
import os
from sentence_transformers import SentenceTransformer
import hashlib
import numpy as np

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load the pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_text(text):
    """Preprocess text: lowercase, remove punctuation, remove stopwords."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[\W_]+", " ", text)
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        try:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        except Exception as e:
            st.error(f"Error extracting text from page: {e}")
    return text.strip()

def extract_text_from_docx(file):
    doc = Document(file)
    text = " ".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_pptx(file):
    presentation = Presentation(file)
    text = " ".join([shape.text for slide in presentation.slides for shape in slide.shapes if hasattr(shape, "text")])
    return text

def process_file(file):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    elif file.name.endswith(".pptx"):
        return extract_text_from_pptx(file)
    else:
        st.error("Unsupported file format. Please upload a PDF, DOCX, or PPTX file.")
        return None

def calculate_similarity(text1, text2):
    """Calculate semantic similarity using Sentence-BERT."""
    if not text1 or not text2:
        return 0.0
    embeddings = model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def generate_fingerprint(text, k=5):
    """Generate a fingerprint for the text using shingles (k-grams)."""
    shingles = [text[i:i + k] for i in range(len(text) - k + 1)]
    hashed_shingles = [hashlib.md5(shingle.encode('utf-8')).hexdigest() for shingle in shingles]
    return set(hashed_shingles)

def check_citations(text):
    """Basic citation pattern matching."""
    citation_pattern = r"\b(?:[A-Za-z]+(?:,?\s+[A-Za-z]+)*\s+\d{4})\b"
    return re.findall(citation_pattern, text)

def process_zip_file(zip_file):
    """Extract files from a ZIP archive and process them."""
    extracted_files = []
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall("extracted_files")
        for root, _, files in os.walk("extracted_files"):
            for file in files:
                extracted_files.append(os.path.join(root, file))
    return extracted_files

st.title("Assignment Plagiarism Checker")

uploaded_files = st.file_uploader("Upload student assignments (PDF, DOCX, PPTX, or ZIP)", type=["pdf", "docx", "pptx", "zip"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing files..."):
        assignments = {}
        for file in uploaded_files:
            if file.name.endswith(".zip"):
                extracted_files = process_zip_file(file)
                for extracted_file in extracted_files:
                    with open(extracted_file, "rb") as ef:
                        file_name = os.path.basename(extracted_file)
                        text = process_file(ef)
                        if text:
                            preprocessed_text = preprocess_text(text)
                            if preprocessed_text:
                                assignments[file_name] = preprocessed_text
                            else:
                                st.warning(f"File '{file_name}' contains no usable text after preprocessing.")
                        else:
                            st.warning(f"File '{file_name}' could not be processed.")
            else:
                text = process_file(file)
                if text:
                    preprocessed_text = preprocess_text(text)
                    if preprocessed_text:
                        assignments[file.name] = preprocessed_text
                    else:
                        st.warning(f"File '{file.name}' contains no usable text after preprocessing.")
                else:
                    st.warning(f"File '{file.name}' could not be processed.")

        if not assignments:
            st.error("No valid text was extracted from the uploaded files.")
        else:
            similarity_results = []
            assignment_names = list(assignments.keys())
            for i in range(len(assignment_names)):
                for j in range(i + 1, len(assignment_names)):
                    name1 = assignment_names[i]
                    name2 = assignment_names[j]

                    # Calculate semantic similarity using Sentence-BERT
                    similarity_score = calculate_similarity(assignments[name1], assignments[name2])

                    # Generate fingerprints for both assignments
                    fingerprint1 = generate_fingerprint(assignments[name1])
                    fingerprint2 = generate_fingerprint(assignments[name2])
                    fingerprint_similarity = len(fingerprint1.intersection(fingerprint2)) / len(fingerprint1.union(fingerprint2))

                    # Check for citations in both assignments
                    citations1 = check_citations(assignments[name1])
                    citations2 = check_citations(assignments[name2])
                    citation_similarity = 1 if len(citations1) == len(citations2) else 0

                    # Combine similarities
                    combined_similarity = 0.5 * similarity_score + 0.25 * fingerprint_similarity + 0.25 * citation_similarity

                    similarity_results.append({
                        "Assignment 1": name1,
                        "Assignment 2": name2,
                        "Semantic Similarity": similarity_score * 100,
                        "Fingerprint Similarity": fingerprint_similarity * 100,
                        "Citation Similarity": citation_similarity * 100,
                        "Combined Similarity": combined_similarity * 100
                    })

            results_df = pd.DataFrame(similarity_results)
            if not results_df.empty:
                st.subheader("Plagiarism Report")
                st.write(results_df)

                high_similarity_threshold = st.slider("Set similarity threshold", min_value=50, max_value=100, value=80, step=5)
                high_similarity_pairs = results_df[results_df["Combined Similarity"] >= high_similarity_threshold]

                if not high_similarity_pairs.empty:
                    st.warning("High similarity detected in the following assignments:")
                    st.write(high_similarity_pairs)
                else:
                    st.success("No significantly similar assignments detected.")
            else:
                st.error("No similarity scores were calculated. Please check your input files.")
