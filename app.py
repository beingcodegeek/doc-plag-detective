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

# Download NLTK stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    """Preprocess text: lowercase, remove punctuation, remove stopwords."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[\W_]+", " ", text)  # Remove punctuation and special characters
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

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
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

def highlight_similarities(text1, text2):
    matcher = SequenceMatcher(None, text1, text2)
    matches = matcher.get_matching_blocks()
    highlighted_text1 = ""
    highlighted_text2 = ""

    for i, match in enumerate(matches):
        start1, start2, length = match
        if length > 0:
            match_text1 = text1[start1:start1+length]
            match_text2 = text2[start2:start2+length]
            highlighted_text1 += f"<span style='background-color: yellow;'>{match_text1}</span> "
            highlighted_text2 += f"<span style='background-color: yellow;'>{match_text2}</span> "

    return highlighted_text1, highlighted_text2

st.title("Document Plagiarism Checker")

uploaded_file1 = st.file_uploader("Upload the first document", type=["pdf", "docx", "pptx"])
uploaded_file2 = st.file_uploader("Upload the second document", type=["pdf", "docx", "pptx"])

if uploaded_file1 and uploaded_file2:
    with st.spinner("Processing files..."):
        text1 = process_file(uploaded_file1)
        text2 = process_file(uploaded_file2)

        if text1 and text2:
            # Preprocess text
            preprocessed_text1 = preprocess_text(text1)
            preprocessed_text2 = preprocess_text(text2)

            # Calculate similarity
            similarity_score = calculate_similarity(preprocessed_text1, preprocessed_text2)

            # Highlight matches
            highlighted_text1, highlighted_text2 = highlight_similarities(text1, text2)

            st.write(f"**Similarity Score:** {similarity_score * 100:.2f}%")
            if similarity_score > 0.8:
                st.warning("The documents are highly similar!")
            else:
                st.success("The documents are not significantly similar.")

            st.subheader("Text Comparison")
            st.markdown("**Document 1 Highlighted Matches:**")
            st.markdown(f"<div style='overflow:auto; background-color: #f8f9fa; padding: 10px;'>{highlighted_text1}</div>", unsafe_allow_html=True)

            st.markdown("**Document 2 Highlighted Matches:**")
            st.markdown(f"<div style='overflow:auto; background-color: #f8f9fa; padding: 10px;'>{highlighted_text2}</div>", unsafe_allow_html=True)
