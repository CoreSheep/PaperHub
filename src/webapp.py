import os
import io
import sys
import logging
from pathlib import Path
from query import *
from json import JSONDecodeError
import pandas as pd
import streamlit as st
from annotated_text import annotation
from markdown import markdown
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from streamlit import session_state as ss
from grobid_client.grobid_client import GrobidClient
import requests

# Initialize the Grobid client
grobid_client = GrobidClient(config_path="./config.json")

# Adjust to a question that you would like users to see in the search bar when they load the UI:
DEFAULT_QUESTION_AT_STARTUP = os.getenv("DEFAULT_QUESTION_AT_STARTUP",
                                        "What role does artificial intelligence play in analyzing bibliometric data "
                                        "within medical research?")
DEFAULT_ANSWER_AT_STARTUP = os.getenv("DEFAULT_ANSWER_AT_STARTUP", "RAG Model")

# Sliders
DEFAULT_DOCS_FROM_RETRIEVER = int(os.getenv("DEFAULT_DOCS_FROM_RETRIEVER", "3"))
DEFAULT_NUMBER_OF_ANSWERS = int(os.getenv("DEFAULT_NUMBER_OF_ANSWERS", "3"))

# Labels for the evaluation
EVAL_LABELS = os.getenv("EVAL_FILE", str(Path(__file__).parent / "random_questions.csv"))

# Whether the file upload should be enabled or not
DISABLE_FILE_UPLOAD = bool(os.getenv("DISABLE_FILE_UPLOAD"))


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

def process_pdf(file_content):
    url = 'https://kermitt2-grobid.hf.space/api/processFulltextDocument'
    files = {'input': ('filename.pdf', file_content, 'application/pdf')}
    response = requests.post(url, files=files)
    return response.text


def main():
    st.set_page_config(page_title="A Medical Question-Answering Demo Based On RAG")

    # Persistent state
    set_state_if_absent("question", DEFAULT_QUESTION_AT_STARTUP)
    set_state_if_absent("answer", DEFAULT_ANSWER_AT_STARTUP)
    set_state_if_absent("results", None)
    set_state_if_absent("raw_json", None)
    set_state_if_absent("random_question_requested", False)

    # Small callback to reset the interface in case the text of the question changes
    def reset_results(*args):
        st.session_state.answer = None
        st.session_state.results = None
        st.session_state.raw_json = None

    # Title
    st.write("# A Medical Question-Answering Demo Based On RAG")
    st.markdown(
        """
            Ask a question and see if our rag model can find the correct answer to your query!
            
            **Note:** search anything you are interested in the intelligence of medical area, 
            please do not use key words, enter the whole question.
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.header("Options")
    top_k_retriever = st.sidebar.slider(
        "Max. number of documents from retriever",
        min_value=1,
        max_value=10,
        value=DEFAULT_DOCS_FROM_RETRIEVER,
        step=1,
        on_change=reset_results,
    )

    # Define your options for the selector
    model_options = ["gpt-3.5-turbo", "gpt-4"]

    # Create a select box widget on the Streamlit app
    chat_model_selected = st.sidebar.selectbox("Choose a model:", model_options)

    st.sidebar.markdown(
        f"""
    <style>
        a {{
            text-decoration: none;
        }}
        .chat-footer {{
            text-align: center;
            
        }}
        .chat-footer h4 {{
            margin: 0.1rem;
            padding:0;
            
        }}
        footer {{
            opacity: 0;
        }}
    </style>
    <div class="chat-footer">
        <hr />
        <h4>View source Code <a href="https://github.com/Aayushtirmalle/QA-Robot-Med-INLPT-WS2023">QA-Med-Robot</a></h4>
        <h4>Built with <a href="https://python.langchain.com/docs/get_started/introduction">Langchain</a> 0.1.8 </h4>
        <p>Get it on <a href="https://github.com/langchain-ai/langchain">GitHub</a> &nbsp;&nbsp; - &nbsp;&nbsp; Read the <a href="https://python.langchain.com/docs/get_started/quickstart">Docs</a></p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Load csv into pandas dataframe
    try:
        df = pd.read_csv(EVAL_LABELS, sep=";")
    except Exception:
        st.error(
            f"The eval file was not found. Please check the demo's [README]("
            f"https://github.com/deepset-ai/haystack/tree/main/ui/README.md) for more information."
        )
        sys.exit(
            f"The eval file was not found under `{EVAL_LABELS}`. Please check the README (https://github.com/deepset-ai/haystack/tree/main/ui/README.md) for more information."
        )

    # Search bar
    question = st.text_input(
        value=st.session_state.question,
        max_chars=100,
        on_change=reset_results,
        label="question",
        label_visibility="hidden",
    )


    
     # File uploader
    if 'pdf_ref' not in ss:
        ss.pdf_ref = None
    somen = st.file_uploader("Or drop a PDF Document here:", type='pdf', accept_multiple_files=False, key='pdf', disabled=False, label_visibility="visible")
    if ss.pdf:
        ss.pdf_ref = ss.pdf 
    if ss.pdf_ref:
        binary_data = ss.pdf_ref.getvalue()
        pdf_viewer(input=binary_data, width=500, height=600)
        file_name = ss.pdf_ref.name
        file_bytes = io.BytesIO(somen.read().decode('utf-8', errors='ignore').encode('utf-8'))
        
        rsp = process_pdf(ss.pdf_ref.read())
        question = rsp


        print(rsp)
    
    col1, col2 = st.columns(2)
    col1.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)
    col2.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)


    # Run button
    run_pressed = col1.button("Run")

    # Get next random question from the CSV
    if col2.button("Random question"):
        reset_results()
        new_row = df.sample(1)
        while (
                new_row["Question Text"].values[0] == st.session_state.question
        ):
            new_row = df.sample(1)
        st.session_state.question = new_row["Question Text"].values[0]
        st.session_state.answer = new_row["Answer"].values[0]
        st.session_state.random_question_requested = True
        if hasattr(st, "scriptrunner"):
            raise st.scriptrunner.script_runner.RerunException(
                st.scriptrunner.script_requests.RerunData(widget_states=None)
            )
        raise st.runtime.scriptrunner.script_runner.RerunException(
            st.runtime.scriptrunner.script_requests.RerunData(widget_states=None)
        )
    st.session_state.random_question_requested = False

    # Initialize the MedChatbot and Pinecone vector store
    chatbot = MedChatbot()
    vc = chatbot.load_vectorstore()
    index = chatbot.get_index(vc)
    text_field = "text"
    run_query = (
                    run_pressed or question != st.session_state.question
                ) and not st.session_state.random_question_requested

    # Check the connection
    with st.spinner("⌛️ &nbsp;&nbsp; Connection is starting..."):
        if index is None:
            st.error("🚫 &nbsp;&nbsp; Index Error. Is vector store running?")
            run_query = False
            reset_results()
    
    # Get the query results from our RAG pipeline
    if run_query and question:
        reset_results()
        st.session_state.question = question

        with (st.spinner(
                "🔍 &nbsp;&nbsp; Performing neural search on vector store... \n "
        )):
            try:
                st.session_state.results, retriever_results = chatbot.query(index, question,
                                                                            top_k_retriever,
                                                                            text_field,
                                                                            chat_model_selected)
            except JSONDecodeError as je:
                st.error("👓 &nbsp;&nbsp; An error occurred reading the results. Is the document store working?")
                return
            except Exception as e:
                logging.exception(e)
                if "The server is busy processing requests" in str(e) or "503" in str(e):
                    st.error("🧑‍🌾 &nbsp;&nbsp; All our workers are busy! Try again later.")
                else:
                    st.error("🐞 &nbsp;&nbsp; An error occurred during the request.")
                return

    # Show the answers on the screen
    if st.session_state.results:
        st.write("## Answer:")
        st.write(
            markdown(str(annotation(st.session_state.results.content, "ANSWER", "#005C53"))),
            unsafe_allow_html=True
        )

        st.write("## Top-k Retrieval Results:")
        if run_query and question:
            for count, result in enumerate(retriever_results):
                page_content, metadata = result.page_content, result.metadata
                # Display each chunk
                st.markdown(f"##### Chunk {count + 1}")
                st.write(
                    markdown(str(annotation(page_content, "CHUNK", "#13678A"))),
                    unsafe_allow_html=True
                )
                st.markdown(f"**Title:** {metadata['title']}&nbsp;&nbsp;&nbsp;**Source:** {metadata['source']} ")


if __name__ == '__main__':
    main()
