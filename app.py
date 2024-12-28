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

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

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
    if not text1 or not text2:
        return 0.0
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

def highlight_similarities(text1, text2):
    matcher = SequenceMatcher(None, text1, text2)
    matches = matcher.get_matching_blocks()
    highlighted_text1 = ""
    highlighted_text2 = ""

    for match in matches:
        start1, start2, length = match
        if length > 0:
            match_text1 = text1[start1:start1+length]
            match_text2 = text2[start2:start2+length]
            highlighted_text1 += f"<span style='background-color: yellow;'>{match_text1}</span> "
            highlighted_text2 += f"<span style='background-color: yellow;'>{match_text2}</span> "

    return highlighted_text1, highlighted_text2

st.title("Assignment Plagiarism Checker")

uploaded_files = st.file_uploader("Upload student assignments", type=["pdf", "docx", "pptx"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing files..."):
        assignments = {}
        for file in uploaded_files:
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
                    similarity_score = calculate_similarity(assignments[name1], assignments[name2])
                    similarity_results.append({
                        "Assignment 1": name1,
                        "Assignment 2": name2,
                        "Similarity Score": similarity_score * 100
                    })

            results_df = pd.DataFrame(similarity_results)
            if not results_df.empty:
                st.subheader("Plagiarism Report")
                st.write(results_df)

                high_similarity_threshold = st.slider("Set similarity threshold", min_value=50, max_value=100, value=80, step=5)
                high_similarity_pairs = results_df[results_df["Similarity Score"] >= high_similarity_threshold]

                if not high_similarity_pairs.empty:
                    st.warning("High similarity detected in the following assignments:")
                    st.write(high_similarity_pairs)
                else:
                    st.success("No significantly similar assignments detected.")
            else:
                st.error("No similarity scores were calculated. Please check your input files.")
