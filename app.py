import os
import re
import zipfile
import hashlib
from difflib import SequenceMatcher
from flask import Flask, request, jsonify
import fitz  # PyMuPDF for PDF processing
from docx import Document
from pptx import Presentation

app = Flask(__name__)

def preprocess_text(text):
    """Lowercase and remove punctuation."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[\W_]+", " ", text)
    return text

def extract_text_from_pdf(file):
    """Extract text from PDF using PyMuPDF."""
    text = ""
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf_document:
        text += page.get_text()
    return text.strip()

def extract_text_from_docx(file):
    """Extract text from DOCX files."""
    doc = Document(file)
    return " ".join([para.text for para in doc.paragraphs])

def extract_text_from_pptx(file):
    """Extract text from PPTX files."""
    presentation = Presentation(file)
    return " ".join([shape.text for slide in presentation.slides for shape in slide.shapes if hasattr(shape, "text")])

def process_file(file, filename):
    """Process supported file types."""
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(file)
    elif filename.endswith(".pptx"):
        return extract_text_from_pptx(file)
    return None

def generate_fingerprint(text, k=5):
    """Generate a fingerprint for the text using k-grams."""
    shingles = [text[i:i + k] for i in range(len(text) - k + 1)]
    hashed_shingles = [hashlib.md5(shingle.encode("utf-8")).hexdigest() for shingle in shingles]
    return set(hashed_shingles)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    
    uploaded_files = request.files.getlist('files[]')
    assignments = {}

    for file in uploaded_files:
        filename = file.filename
        if filename.endswith(".zip"):
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall("extracted_files")
                for root, _, files in os.walk("extracted_files"):
                    for extracted_file in files:
                        with open(os.path.join(root, extracted_file), "rb") as ef:
                            text = process_file(ef, extracted_file)
                            if text:
                                assignments[extracted_file] = preprocess_text(text)
        else:
            text = process_file(file, filename)
            if text:
                assignments[filename] = preprocess_text(text)
    
    # Check for similarities
    results = []
    assignment_names = list(assignments.keys())
    for i in range(len(assignment_names)):
        for j in range(i + 1, len(assignment_names)):
            name1 = assignment_names[i]
            name2 = assignment_names[j]
            
            fingerprint1 = generate_fingerprint(assignments[name1])
            fingerprint2 = generate_fingerprint(assignments[name2])
            similarity = len(fingerprint1.intersection(fingerprint2)) / len(fingerprint1.union(fingerprint2)) * 100
            
            results.append({"Assignment 1": name1, "Assignment 2": name2, "Similarity (%)": round(similarity, 2)})

    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(debug=True)
