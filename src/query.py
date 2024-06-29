from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from langchain.vectorstores import Pinecone
from datasets import load_dataset
from pinecone import PodSpec
from langchain_openai import OpenAIEmbeddings
from tqdm.auto import tqdm  # for progress bar
from dotenv import load_dotenv  # Only if using python-dotenv

import datasets
import pandas as pd
import time
import requests
import os

# Load environment variables from .env file (if using python-dotenv)
load_dotenv()


class PaperChatBot:
    def __init__(self):
        """
        Initializes the MedChatbot class with necessary API keys and configurations.
        """
        # Initialize API_KEYS
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        self.data_path = "../assets/data/medical_articles.json"
        self.embedding_model = "text-embedding-ada-002"
        self.embed_model = OpenAIEmbeddings(model=self.embedding_model, api_key=self.OPENAI_API_KEY)
        self.spec = PodSpec(
            environment=self.PINECONE_ENVIRONMENT, pod_type="starter"
        )
        self.index_name = 'medical-intelligence-rag'

    def load_data(self):
        """
        Load data from a specified file path.

        Returns:
            pd.DataFrame: Data loaded from the JSON file.
        """
        # Temporarily disable SSL verification
        requests.packages.urllib3.disable_warnings()
        datasets.utils.VerificationMode = False

        # Read the JSON file
        with open(self.data_path, 'r', encoding='utf-8') as file:
            data = pd.read_json(file)

        # Convert the 'articles' list in the JSON data to a DataFrame
        df = pd.DataFrame(data['articles'])
        return df

    def load_vectorstore(self):
        """
        Initialize and return the Pinecone vector store.

        Returns:
            Pinecone: The initialized Pinecone vector store.
        """
        from pinecone import Pinecone
        return Pinecone(api_key=self.PINECONE_API_KEY)

    def create_index(self, vector_store, data, batch_size=100):
        """
        Create an index for the loaded data.

        Args:
            vector_store (Pinecone): The Pinecone vector store instance.
            data (pd.DataFrame): The data to be indexed.
            batch_size (int): The batch size for processing the data.

        Returns:
            Pinecone.Index: The created Pinecone index.
        """
        existing_indexes = [
            index_info["name"] for index_info in vector_store.list_indexes()
        ]

        # Check if index already exists (it shouldn't if this is first time)
        if self.index_name not in existing_indexes:
            # If does not exist, create index
            vector_store.create_index(
                self.index_name,
                dimension=1536,  # dimensionality of ada 002
                metric='dotproduct',
                spec=self.spec
            )
            # Wait for index to be initialized
            while not vector_store.describe_index(self.index_name).status['ready']:
                time.sleep(1)

        # Connect to index
        index = vector_store.Index(self.index_name)
        time.sleep(1)

        dataset = data  # This makes it easier to iterate over the dataset

        print("Creating embeddings for data ...\n")
        for i in tqdm(range(0, len(dataset), batch_size)):
            i_end = min(len(dataset), i + batch_size)
            # Get batch of data
            batch = dataset.iloc[i:i_end]

            # Generate unique ids for each chunk
            ids = [f"{x['id']}-{x['chunk-id']}" for i, x in batch.iterrows()]
            # Get text to embed
            texts = [x['chunk'] for _, x in batch.iterrows()]
            # Embed text
            embeds = self.embed_model.embed_documents(texts)

            # Get metadata to store in Pinecone
            metadata = [
                {'text': x['chunk'],
                 'source': x['source'],
                 'title': x['title'],
                 'authors': x['authors'],
                 'journal_ref': x['journal_ref'],
                 'published': x['published']
                 } for i, x in batch.iterrows()
            ]
            # Add to Pinecone
            index.upsert(vectors=zip(ids, embeds, metadata))
            print(index.describe_index_stats())
        return index

    def get_index(self, vectorstore):
        """
        Connect to and return the existing Pinecone index.

        Args:
            vectorstore (Pinecone): The Pinecone vector store instance.

        Returns:
            Pinecone.Index: The connected Pinecone index.
        """
        # Connect to index
        return vectorstore.Index(self.index_name)

    def augment_prompt(self, query, k, vectorstore):
        """
        Augment the query with relevant context from the knowledge base.

        Args:
            query (str): The user query.
            k (int): The number of top results to retrieve.
            vectorstore (Pinecone.Index): The Pinecone index instance.

        Returns:
            str: The augmented prompt.
            list: The retriever results.
        """
        # Get top k results from knowledge base
        results = vectorstore.similarity_search(query, k)
        # Get the text from the results
        source_knowledge = "\n".join([x.page_content for x in results])
        # Feed into an augmented prompt
        augmented_prompt = f"""Using the contexts below, answer the query.

        Contexts:
        {source_knowledge}

        Query: {query}"""
        return augmented_prompt, results

    def query(self, index, query, k, text_field="text", chat_model="open-mistral-7b"):
        """
        Query the vector store and return the AI response and retriever results.

        Args:
            index (Pinecone.Index): The Pinecone index instance.
            query (str): The user query.
            k (int): The number of top results to retrieve.
            text_field (str): The text field to use for similarity search.
            chat_model (str): The chat model to use for generating responses.

        Returns:
            str: The AI response.
            list: The retriever results.
        """
        # Initialize the vector store object
        from langchain.vectorstores import Pinecone
        vectorstore = Pinecone(
            index, self.embed_model, text_field
        )
        augment_prompt, retriever_results = self.augment_prompt(query, k, vectorstore)

        client = MistralClient(api_key=self.MISTRAL_API_KEY)

        messages = [
            ChatMessage(role="user", content=augment_prompt)
        ]

        # No streaming
        chat_response = client.chat(
            model=chat_model,
            messages=messages,
        )
        return chat_response.choices[0].message.content, retriever_results


if __name__ == '__main__':
    chatbot = PaperChatBot()
    vc = chatbot.load_vectorstore()
    index = chatbot.get_index(vc)

    text_field = "text"
    top_k_retriever = 3
    chat_model = "open-mistral-7b"
    question = "What is the best treatment for diabetes?"

    results, retriever_results = chatbot.query(index, question,
                                               top_k_retriever,
                                               text_field,
                                               chat_model)
    print(results)
    print(retriever_results)
