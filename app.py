import os
import re
import zipfile
import hashlib
from difflib import SequenceMatcher
from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF for PDF processing
from docx import Document
from pptx import Presentation
import requests

app = Flask(__name__)
CORS(app)
TEMP_DIR = "temp_files"
ZIP_FILE_PATH = "temp_files.zip"
os.makedirs(TEMP_DIR, exist_ok=True)

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
    file_details = request.json.get('fileDetails', [])
    # print(file_details)
    if 'fileDetails' not in request.json:
            return jsonify({"success": False, "message": "Invalid request data"}), 400
    file_paths = []
    files={}
    for file in file_details:
            student_id = file.get('studentId')
            file_url = file.get('fileUrl')

            if not file_url or not student_id:
                continue

            response = requests.get(file_url)
            if response.status_code == 200:
                file_name = f"{os.path.basename(file_url)}"
                file_path = os.path.join(TEMP_DIR, file_name)
                files[file_name] = student_id
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                file_paths.append(file_path)

            else:
                print(f"Failed to download file from: {file_url}")
  
    uploaded_files = []
    assignments = {}
    for file in file_paths:
        uploaded_files.append(file)

    for file in uploaded_files:
        with open(file, "rb") as f:
            text = process_file(f, file)
            if text:
                assignments[os.path.basename(file)] = preprocess_text(text)
            else:
                text = process_file(file, file)
    
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
    app.run(port=8081)
