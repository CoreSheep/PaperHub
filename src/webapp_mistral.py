import os
import sys
import logging
from pathlib import Path
from query import *
from json import JSONDecodeError
import pandas as pd
import streamlit as st
from annotated_text import annotation
from markdown import markdown
from dotenv import load_dotenv  # Only if using python-dotenv
# Load environment variables from .env file (if using python-dotenv)
load_dotenv()
import logging


# Default question and answer to be shown in the search bar at startup
DEFAULT_QUESTION_AT_STARTUP = os.getenv("DEFAULT_QUESTION_AT_STARTUP",
                                        "What is the difference between supervised and unsupervised learning?")
DEFAULT_ANSWER_AT_STARTUP = os.getenv("DEFAULT_ANSWER_AT_STARTUP", "RAG Model")

# Sliders' default values
DEFAULT_DOCS_FROM_RETRIEVER = int(os.getenv("DEFAULT_DOCS_FROM_RETRIEVER", "3"))
DEFAULT_NUMBER_OF_ANSWERS = int(os.getenv("DEFAULT_NUMBER_OF_ANSWERS", "3"))

# Labels for the evaluation
RANDOM_QUESTION = os.getenv("EVAL_FILE", str(Path(__file__).parent / "random_questions.csv"))

# Whether the file upload should be enabled or not
DISABLE_FILE_UPLOAD = bool(os.getenv("DISABLE_FILE_UPLOAD"))


def set_state_if_absent(key, value):
    """
    Sets a default state in Streamlit session state if the key is not already present.

    Args:
        key (str): The key to be checked in the session state.
        value: The value to be set if the key is absent.
    """
    if key not in st.session_state:
        st.session_state[key] = value


def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="PaperHub", page_icon="📚")

    # Initialize persistent state
    set_state_if_absent("question", DEFAULT_QUESTION_AT_STARTUP)
    set_state_if_absent("answer", DEFAULT_ANSWER_AT_STARTUP)
    set_state_if_absent("results", None)
    set_state_if_absent("raw_json", None)
    set_state_if_absent("random_question_requested", False)

    # Callback function to reset results when the question changes
    def reset_results(*args):
        st.session_state.answer = None
        st.session_state.results = None
        st.session_state.raw_json = None

    # Example color codes
    title_color = "#e36414"  # Orange
    label_color = "#e36414"  # Orange

    # Title with color
    st.markdown(f"""
    <h1 style='color: {title_color};'>📚🔎💡 A Question-Answering ChatBot on Academic Paper Based On RAG</h1>
    """, unsafe_allow_html=True)

    # Labels for the keywords
    st.markdown(f"""
    <div style="display: flex; gap: 10px;">
        <span style='background-color: {label_color}; color: white; padding: 5px; border-radius: 5px;'>Machine Learning</span>
        <span style='background-color: {label_color}; color: white; padding: 5px; border-radius: 5px;'>Natural Language Processing</span>
        <span style='background-color: {label_color}; color: white; padding: 5px; border-radius: 5px;'>Computer Vision</span>
    </div>
    """, unsafe_allow_html=True)

    # Description with bold keywords
    st.markdown(
        """
        Ask a question on **Machine Learning**, **NLP**, and **Computer Vision** and see if our RAG model can find the correct 
        answer to your query!

        **Note:** Search for anything you are interested in regarding **Machine Learning**, **NLP**, and **Computer Vision**. 
        Please enter the whole question instead of keywords.
        """,
        unsafe_allow_html=True,
    )

    # Sidebar options
    st.sidebar.header("Options")
    top_k_retriever = st.sidebar.slider(
        "Max. number of documents from retriever",
        min_value=1,
        max_value=10,
        value=DEFAULT_DOCS_FROM_RETRIEVER,
        step=1,
        on_change=reset_results,
    )

    # Define options for the chat model selector
    model_options = [
        "open-mistral-7b",
        "open-mixtral-8x7b",
        "open-mixtral-8x22b",
        "mistral-small-latest",
        "mistral-medium-latest",
        "mistral-large-latest"
    ]

    # Create a select box widget in the Streamlit sidebar
    chat_model_selected = st.sidebar.selectbox("Choose a model:", model_options)

    # Add a link to the source code and the libraries used
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
                <h4>View source Code <a href="https://github.com/CoreSheep/PaperHub">PaperHub</a></h4>
                <h4>Built with <a href="https://python.langchain.com/docs/get_started/introduction">Langchain</a> 0.1.8 </h4>
                <h4> & <a href="https://mistral.ai/">Mistral AI</a></h4>
            </div>
        """,
        unsafe_allow_html=True,
    )

    # Load random questions from CSV into a pandas dataframe
    try:
        df = pd.read_csv(RANDOM_QUESTION, sep=";")
    except Exception:
        st.error(
            f"The random question file was not found. Please check the demo's [README]("
            f"https://github.com/deepset-ai/haystack/tree/main/ui/README.md) for more information."
        )
        sys.exit(
            f"The random question file was not found under `{EVAL_LABELS}`. Please check the README (https://github.com/deepset-ai/haystack/tree/main/ui/README.md) for more information."
        )

    # Search bar for the question
    question = st.text_area(
        value=st.session_state.question,
        height=100,
        on_change=reset_results,
        label="question",
        label_visibility="hidden",
    )

    # Columns for buttons
    col1, col2 = st.columns(2)
    col1.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)
    col2.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)

    # Run button
    run_pressed = col1.button("Run")

    # Button to get a random question from the CSV
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

    # Initialize the PaperChatbot and Pinecone vector store
    chatbot = PaperChatBot()
    vc = chatbot.load_vectorstore()
    index = chatbot.get_index(vc)
    text_field = "text"

    # Check if the query should be run
    run_query = (
                    run_pressed or question != st.session_state.question
                ) and not st.session_state.random_question_requested

    # Check the connection to the vector store
    with st.spinner("⌛️ &nbsp;&nbsp; Connection is starting..."):
        if index is None:
            st.error("🚫 &nbsp;&nbsp; Index Error. Is vector store running?")
            run_query = False
            reset_results()

    # Get the query results from the RAG pipeline
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
        st.markdown("<h2 style='color: #e36414;'>Answer:</h2>", unsafe_allow_html=True)
        st.markdown(
            f"""
             <div style='padding: 10px; border-radius: 5px;'>
                 {markdown(str(st.session_state.results))}
             </div>
             """,
            unsafe_allow_html=True
        )
        st.markdown("<h2 style='color: #e36414;'>Top-k Retrieval Results:</h2>", unsafe_allow_html=True)
        if run_query and question:
            for count, result in enumerate(retriever_results):
                page_content, metadata = result.page_content, result.metadata
                # Display each chunk
                st.markdown(f"##### Chunk {count + 1}")
                st.markdown(
                    f"""
                        <div style='border: 2px solid #e36414; padding: 10px; border-radius: 5px;'>
                            <p>{page_content}</p>
                        </div>
                        """,
                    unsafe_allow_html=True
                )

                st.markdown(f"**Title:** {metadata['title']}&nbsp;&nbsp;&nbsp;**Source:** {metadata['source']} ")


if __name__ == '__main__':
    main()
