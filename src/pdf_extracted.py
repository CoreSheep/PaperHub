import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import PyPDF2
import pandas as pd
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = ''
    for page_num in range(len(pdf.pages)):
        text += pdf.pages[page_num].extract_text()
    return text

def clean_text(text):
    # Remove special characters
    text = re.sub(r'\W', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Remove stop words
    words = [word for word in words if word not in stopwords.words('english')]

    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Remove extra whitespace
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text)

    return text

def query_mistral(text):
    
    api_key = os.getenv("MISTRAL_PAPERHUB_API_KEY")
    model = "mistral-large-latest"  # options on https://docs.mistral.ai/getting-started/models/
    client = MistralClient(api_key=api_key)

    questions = [
        "What are the primary arguments or topics discussed in the paper? Provide a succinct summary that captures the essence of the research.",
        "What are the groundbreaking findings or conclusions drawn from the research? Highlight their implications for the field."
    ]

    answers = []

    for question in questions:
        messages = [
            ChatMessage(role="user", content=f"Based on the provided paper: {text}. Please concisely articulate the answer to this question: {question} Tailor your response to an academic audience.")
        ]
        chat_response = client.chat(messages=messages, model=model)
        response = chat_response.choices[0].message.content
        answers.append(response)

    df = pd.DataFrame([answers], columns=["Insights", "Conclusion"])

    return df