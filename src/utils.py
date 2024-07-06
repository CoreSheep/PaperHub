import streamlit as st
import PyPDF2


# Function to read PDF file and extract text
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    num_pages = len(pdf_reader.pages)
    all_text = ""
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        all_text += page.extract_text()
    return all_text


# Function to process each file (customize this function as needed)
def process_file(uploaded_file):
    # Example: Reading the file content
    if uploaded_file.type == "text/plain":
        content = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        content = read_pdf(uploaded_file)
    return content