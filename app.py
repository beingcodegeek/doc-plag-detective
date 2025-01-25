import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from pptx import Presentation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd
import zipfile
import os

def preprocess_text(text):
    """Preprocess text: lowercase, remove punctuation, and extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[\W_]+", " ", text)
    return " ".join(text.split())

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text.strip()

def extract_text_from_docx(file):
    doc = Document(file)
    return " ".join([para.text for para in doc.paragraphs])

def extract_text_from_pptx(file):
    presentation = Presentation(file)
    return " ".join([shape.text for slide in presentation.slides for shape in slide.shapes if hasattr(shape, "text")])

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
                            assignments[file_name] = preprocess_text(text)
            else:
                text = process_file(file)
                if text:
                    assignments[file.name] = preprocess_text(text)

        if not assignments:
            st.error("No valid text was extracted from the uploaded files.")
        else:
            assignment_names = list(assignments.keys())
            assignment_texts = list(assignments.values())

            # Compute pairwise similarities
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(assignment_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)

            similarity_results = []
            for i in range(len(assignment_names)):
                for j in range(i + 1, len(assignment_names)):
                    similarity_results.append({
                        "Assignment 1": assignment_names[i],
                        "Assignment 2": assignment_names[j],
                        "Similarity (%)": similarity_matrix[i, j] * 100
                    })

            results_df = pd.DataFrame(similarity_results)
            if not results_df.empty:
                st.subheader("Plagiarism Report")
                st.write(results_df)

                high_similarity_threshold = st.slider("Set similarity threshold", min_value=50, max_value=100, value=80, step=5)
                high_similarity_pairs = results_df[results_df["Similarity (%)"] >= high_similarity_threshold]

                if not high_similarity_pairs.empty:
                    st.warning("High similarity detected in the following assignments:")
                    st.write(high_similarity_pairs)
                else:
                    st.success("No significantly similar assignments detected.")
            else:
                st.error("No similarity scores were calculated. Please check your input files.")
