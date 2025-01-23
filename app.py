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
import numpy as np
import zipfile
import os
from sentence_transformers import SentenceTransformer
import hashlib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from urllib.parse import unquote

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

def process_file(file, filename):
    print('file hoon mai:', file, filename)
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(file)
    elif filename.endswith(".pptx"):
        return extract_text_from_pptx(file)
    else:
        st.error("Unsupported file format. Please upload a PDF, DOCX, or PPTX file.")
        return None
    
def process_plag_file(file):
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

# def plag_model(zipfile, filename, file):
#     uploaded_files = zipfile
#     # uploaded_files = st.file_uploader("Upload student assignments (PDF, DOCX, PPTX, or ZIP)", type=["pdf", "docx", "pptx", "zip"], accept_multiple_files=True)
#     print(f'Uploaded files:{uploaded_files}')   
#     if uploaded_files:
#         with st.spinner("Processing files..."):
#             assignments = {}
#             id = {}
#             # for file in uploaded_files:
#             if uploaded_files.endswith(".zip"):
#                     extracted_files = process_zip_file(uploaded_files)
#                     print('extracted', extracted_files)
#                     for extracted_file in extracted_files:
#                         with open(extracted_file, "rb") as ef:
#                             file_name = os.path.basename(extracted_file)
#                             print('file_name',file_name)
#                             text = process_plag_file(ef)
#                             if text:
#                                 preprocessed_text = preprocess_text(text)
#                                 if preprocessed_text:
#                                     assignments[file_name] = preprocessed_text
#                                 else:
#                                     st.warning(f"File '{file_name}' contains no usable text after preprocessing.")
#                             else:
#                                 st.warning(f"File '{file_name}' could not be processed.")
#             else:
#                     text = process_plag_file(file)
#                     if text:
#                         preprocessed_text = preprocess_text(text)
#                         if preprocessed_text:
#                             assignments[file.name] = preprocessed_text
#                         else:
#                             st.warning(f"File '{file.name}' contains no usable text after preprocessing.")
#                     else:
#                         st.warning(f"File '{file.name}' could not be processed.")

#             if not assignments:
#                 st.error("No valid text was extracted from the uploaded files.")
#             else:
#                 similarity_results = []
#                 assignment_names = list(assignments.keys())
#                 print("Assignment names", assignment_names)  
#                 for i in range(len(assignment_names)):
#                     for j in range(i + 1, len(assignment_names)):
#                         name1 = assignment_names[i]
#                         name2 = assignment_names[j]
                        
#                         # Calculate semantic similarity using Sentence-BERT
#                         similarity_score = calculate_similarity(assignments[name1], assignments[name2])
#                         # Generate fingerprints for both assignments
#                         fingerprint1 = generate_fingerprint(assignments[name1])
#                         fingerprint2 = generate_fingerprint(assignments[name2])
#                         fingerprint_similarity = len(fingerprint1.intersection(fingerprint2)) / len(fingerprint1.union(fingerprint2))

#                         # Combine similarities
#                         combined_similarity = 0.75 * similarity_score + 0.25 * fingerprint_similarity

#                         similarity_results.append({
#                             "Assignment 1": name1,
#                             "Assignment 2": name2,
#                             "Semantic Similarity": similarity_score * 100,
#                             "Fingerprint Similarity": fingerprint_similarity * 100,
#                             "Combined Similarity": combined_similarity * 100
#                         })
#                 plag_result = []
#                 plag_result.append(similarity_results)
#                 results_df = pd.DataFrame(similarity_results)
#                 if not results_df.empty:
#                     st.subheader("Plagiarism Report")
#                     st.write(results_df)
                    
#                     # high_similarity_threshold = st.slider("Set similarity threshold", min_value=50, max_value=100, value=80, step=5)
#                     high_similarity_threshold = 80
#                     high_similarity_pairs = results_df[results_df["Combined Similarity"] >= high_similarity_threshold]
#                     print("High similarity pairs", high_similarity_pairs)
                    
#                     if not high_similarity_pairs.empty:
#                         st.warning("High similarity detected in the following assignments:")
#                         st.write(high_similarity_pairs)
#                         plag_result.append(high_similarity_pairs.to_numpy())
#                     else:
#                         st.success("No significantly similar assignments detected.")
#                         # return similarity_results
#                 else:
#                     st.error("No similarity scores were calculated. Please check your input files.")
#     return plag_result

# app = Flask(__name__)
# CORS(app)

# TEMP_DIR = "temp_files"
# ZIP_FILE_PATH = "temp_files.zip"
# os.makedirs(TEMP_DIR, exist_ok=True)

# @app.route('/checkplagiarism/<assignment_id>', methods=['POST'])
# def check_plagiarism(assignment_id):
#     try:
#         data = request.json
#         print('data hai mei', data)
#         if not data or 'fileDetails' not in data:
#             return jsonify({"success": False, "message": "Invalid request data"}), 400

#         file_details = data.get('fileDetails', [])
#         print('file', file_details)
#         downloaded_files = []
#         file_paths = []
#         files = {}
#         # Step 1: Download all files
#         for file in file_details:
#             student_id = file.get('studentId')
#             file_url = file.get('fileUrl')
#             print('student_id', student_id)
#             print('file_url', file_url)

#             if not file_url or not student_id:
#                 continue

#             response = requests.get(file_url)
#             print('hai na:',response)
#             if response.status_code == 200:
#                 file_name = f"{os.path.basename(file_url)}"
#                 file_path = os.path.join(TEMP_DIR, file_name)
#                 # file_name = unquote(file_name) 
#                 files[file_name] = student_id
#                 with open(file_path, 'wb') as f:
#                     f.write(response.content)
#                 file_paths.append(file_path)

#                 downloaded_files.append((student_id, file_path, file_name))

#             else:
#                 print(f"Failed to download file from: {file_url}")

#         if not downloaded_files:
#             return jsonify({"success": False, "message": "No files were downloaded"}), 400
       
#         # Create a ZIP file of all downloaded files
#         print('download',file_paths)
#         zip_file_name = create_zip_file(file_paths)
#         print(f"ZIP file created at: {zip_file_name}")

#         # Step 2: Process files through plagiarism model
#         plag_scores_results = {}
#         plag_res = []
#         print('download kro bhai', downloaded_files)
#         for student_id, file_path, file_name in downloaded_files:
#             try:
#                 # Assuming process_file() extracts text and plag_model() detects plagiarism
#                 text = process_file(file_path, file_name)
#                 if text:
#                     # preprocessed_text = preprocess_text(text)
#                     plag_result = plag_model(zip_file_name, file_name, file_path)
#                     plag_scores_results['plagiarism_result'] = plag_result[0]

#                 else:
#                     print('kaise ho error')
#                     plag_res.append({
#                         "studentId": student_id,
#                         "fileName": os.path.basename(file_path),
#                         "error": "File could not be processed."
#                     })
#             except Exception as e:
#                 print('hello mere pyare error')
#                 plag_res.append({
#                     "studentId": student_id,
#                     "fileName": os.path.basename(file_path),
#                     "error": str(e)
#                 })

#         # print('plag hai bhai:', plag_scores_results)
#         try:
#             plagiarism_results = []
#             print('mujhe maaf kro plag:', plag_scores_results['plagiarism_result'])
#             for plag_items in plag_scores_results['plagiarism_result']:
#                     plagiarism_results.append({
#                         "studentId1": files[plag_items['Assignment 1']],
#                         "studentId2": files[plag_items['Assignment 2']],
#                         "Assignment1": plag_items['Assignment 1'],
#                         "Assignment2": plag_items['Assignment 2'],
#                         "SemanticSimilarity": plag_items['Semantic Similarity'],  # Ensure this key matches the key in plag_result
#                         "FingerprintSimilarity": plag_items['Fingerprint Similarity'],  # Same here
#                         "CombinedSimilarity": plag_items['Combined Similarity'],  # And here
#                     })
#             # Step 3: Return aggregated results
#             print('plag:', plagiarism_results)
#         except Exception as e:
#             return jsonify({"error": str(e)}), 404
        
#         return jsonify({"success": True, "results": plagiarism_results})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
def create_zip_file(files):
    #Create a ZIP file from a list of file paths.
        with zipfile.ZipFile(ZIP_FILE_PATH, 'w') as zipf:
            for file in files:
                zipf.write(file, os.path.basename(file))  # Add file to the ZIP
            return ZIP_FILE_PATH
    
# if __name__ == '__main__':
#     # app.run(debug=True,port=8081)
#     app.run(port=8081)



from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from django.conf import settings
import json

# Temporary directory and ZIP file path
TEMP_DIR = "temp_files"
ZIP_FILE_PATH = "temp_files.zip"
os.makedirs(TEMP_DIR, exist_ok=True)

def plag_model(zipfile, filename, file):
    uploaded_files = zipfile
    # uploaded_files = st.file_uploader("Upload student assignments (PDF, DOCX, PPTX, or ZIP)", type=["pdf", "docx", "pptx", "zip"], accept_multiple_files=True)
    print(f'Uploaded files:{uploaded_files}')   
    if uploaded_files:
        with st.spinner("Processing files..."):
            assignments = {}
            id = {}
            # for file in uploaded_files:
            if uploaded_files.endswith(".zip"):
                    extracted_files = process_zip_file(uploaded_files)
                    print('extracted', extracted_files)
                    for extracted_file in extracted_files:
                        with open(extracted_file, "rb") as ef:
                            file_name = os.path.basename(extracted_file)
                            print('file_name',file_name)
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
                    text = process_plag_file(file)
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
                print("Assignment names", assignment_names)  
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

                        # Combine similarities
                        combined_similarity = 0.75 * similarity_score + 0.25 * fingerprint_similarity

                        similarity_results.append({
                            "Assignment 1": name1,
                            "Assignment 2": name2,
                            "Semantic Similarity": similarity_score * 100,
                            "Fingerprint Similarity": fingerprint_similarity * 100,
                            "Combined Similarity": combined_similarity * 100
                        })
                plag_result = []
                plag_result.append(similarity_results)
                results_df = pd.DataFrame(similarity_results)
                if not results_df.empty:
                    st.subheader("Plagiarism Report")
                    st.write(results_df)
                    
                    # high_similarity_threshold = st.slider("Set similarity threshold", min_value=50, max_value=100, value=80, step=5)
                    high_similarity_threshold = 80
                    high_similarity_pairs = results_df[results_df["Combined Similarity"] >= high_similarity_threshold]
                    print("High similarity pairs", high_similarity_pairs)
                    
                    if not high_similarity_pairs.empty:
                        st.warning("High similarity detected in the following assignments:")
                        st.write(high_similarity_pairs)
                        plag_result.append(high_similarity_pairs.to_numpy())
                    else:
                        st.success("No significantly similar assignments detected.")
                        # return similarity_results
                else:
                    st.error("No similarity scores were calculated. Please check your input files.")
    return plag_result

@method_decorator(csrf_exempt, name='dispatch')
class CheckPlagiarismView(View):
    def post(self, request, assignment_id):
        try:
            data = json.loads(request.body)
            if not data or 'fileDetails' not in data:
                return JsonResponse({"success": False, "message": "Invalid request data"}, status=400)

            file_details = data.get('fileDetails', [])
            downloaded_files = []
            file_paths = []
            files = {}

            # Step 1: Download all files
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

                    downloaded_files.append((student_id, file_path, file_name))
                else:
                    print(f"Failed to download file from: {file_url}")

            if not downloaded_files:
                return JsonResponse({"success": False, "message": "No files were downloaded"}, status=400)

            # Create a ZIP file of all downloaded files
            zip_file_name = create_zip_file(file_paths)

            # Step 2: Process files through plagiarism model
            plag_scores_results = {}
            plag_res = []

            for student_id, file_path, file_name in downloaded_files:
                try:
                    text = process_file(file_path, file_name)
                    if text:
                        plag_result = plag_model(zip_file_name, file_name, file_path)
                        plag_scores_results['plagiarism_result'] = plag_result[0]
                    else:
                        plag_res.append({
                            "studentId": student_id,
                            "fileName": os.path.basename(file_path),
                            "error": "File could not be processed."
                        })
                except Exception as e:
                    plag_res.append({
                        "studentId": student_id,
                        "fileName": os.path.basename(file_path),
                        "error": str(e)
                    })

            try:
                plagiarism_results = []
                for plag_items in plag_scores_results.get('plagiarism_result', []):
                    plagiarism_results.append({
                        "studentId1": files[plag_items['Assignment 1']],
                        "studentId2": files[plag_items['Assignment 2']],
                        "Assignment1": plag_items['Assignment 1'],
                        "Assignment2": plag_items['Assignment 2'],
                        "SemanticSimilarity": plag_items['Semantic Similarity'],
                        "FingerprintSimilarity": plag_items['Fingerprint Similarity'],
                        "CombinedSimilarity": plag_items['Combined Similarity'],
                    })

                return JsonResponse({"success": True, "results": plagiarism_results})
            except Exception as e:
                return JsonResponse({"error": str(e)}, status=404)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

from django.urls import path

urlpatterns = [
    path('checkplagiarism/<str:assignment_id>/', CheckPlagiarismView.as_view(), name='check_plagiarism'),
]