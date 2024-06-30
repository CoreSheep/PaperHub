from Bio import Entrez, Medline
from datetime import datetime
import json
import glob

from langchain.text_splitter import CharacterTextSplitter


def custom_text_splitter(text, chunk_size):
    """Split the text into chunks of specified size."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=20
    )
    return text_splitter.create_documents(text)


def get_chunks(search_term, chunk_size, email, output_file='../assets/data/medical_articles.json'):
    Entrez.email = email
    search_results = Entrez.read(
        Entrez.esearch(db="pubmed", term=f"{search_term}[Abstract]", retmax=100, datetype="pdat")
    )
    articles = []

    # Fetch and print information for each document
    for pmid in search_results['IdList']:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
        record = Medline.read(handle)
        # print(record)
        print(f"pmid: {pmid}")

        # Check if the abstract contains the term "intelligence"
        if 'AB' in record.keys() and search_term in record['AB'].lower():
            # Check if all the keys are present in the record
            required_keys = ['PMID', 'TI', 'AU', 'JT', 'DP', 'AB', 'AID']
            if all(key in record for key in required_keys):
                # Check if the publication date is within the specified range
                if 'DP' in record.keys():
                    # print(record['DP'])
                    try:
                        # Try parsing with format '%Y %b %d'
                        publication_date = datetime.strptime(record['DP'], "%Y %b %d").date()
                    except ValueError:
                        try:
                            # If the first format fails, try parsing with format '%Y %b'
                            publication_date = datetime.strptime(record['DP'], "%Y %b").date()
                        except ValueError:
                            try:
                                # If the second format fails, try parsing with format '%Y'
                                publication_date = datetime.strptime(record['DP'], "%Y").date()
                            except ValueError:
                                print(f"Could not parse the date: {record['DP']}")
                                # If date format is not one of the three formats, then we skip this date
                                publication_date = None

                    if publication_date != None and 2013 <= publication_date.year <= 2023:
                        # Save relevant information for the document
                        print(f"Processing document {pmid}\n")
                        abstract = record['AB']
                        chunks = custom_text_splitter([abstract], chunk_size)
                        for chunk_id, chunk in enumerate(chunks, start=1):
                            print(f"Processing chunk {chunk_id}\n")
                            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{record.get('PMID')}/"
                            article = {
                                'id': record.get('PMID', ''),
                                'doi': record.get('AID', ''),
                                'title': record.get('TI', ''),
                                'abstract': record.get('AB', ''),
                                'chunk-id': chunk_id,
                                'chunk': chunk,
                                'authors': record.get('AU', []),
                                'journal_ref': record.get('JT', ''),
                                'published': record.get('DP', ''),
                                'source': pubmed_url
                            }
                            # Depending on the availability and requirement, adjust the fields
                            articles.append(article)
                            print(article)

    # Write the chunks to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=4)

    print(f"Processed and saved {len(articles)} chunks to {output_file}.")


def merge_json_files(output_file, input_files_pattern):
    """
    Merge multiple JSON files into one JSON file.

    Args:
        output_file (str): The file path for the output merged JSON file.
        input_files_pattern (str): The pattern to match the input JSON files (e.g., 'data/*.json').

    Returns:
        None
    """
    combined_data = []

    # Get the list of JSON files to be merged
    input_files = glob.glob(input_files_pattern)
    print(input_files)

    # Iterate over each file and append its data to the combined_data list
    for file_path in input_files:
        with open(file_path, 'r') as file:
            data = json.load(file)
            combined_data.extend(data)

    # Write the combined data to the output file
    with open(output_file, 'w') as output_file:
        json.dump(combined_data, output_file, indent=4)


if __name__ == "__main__":
    # Example usage
    merge_json_files('./data_merged_chunked.json', '/Users/jiufeng/Documents/'
                                                   '24ss/HeiGenAI/PaperHub/assets/data/dataset_paperhub/*.json')

# # Initialize the text splitter with a specific chunk size
# chunk_size = 256  # Define your desired chunk size
#
# email = "leepual168@gmail.com"
# # Search for documents with the term "intelligence" in the abstract and published between 2013 and 2023
# search_term = "intelligence"
# if __name__ == '__main__':
#     get_chunks(search_term, chunk_size, email)
#     # example_ab = "Alan Schelten Ruan Silva Eric Michael Smith Ranjan Subramanian Xiaoqing Ellen Tan Binh Tang\nRoss Taylor Adina Williams Jian Xiang Kuan Puxin Xu Zheng Yan Iliyan Zarov Yuchen Zhang\nAngela Fan Melanie Kambadur Sharan Narang Aurelien Rodriguez Robert Stojnic\nSergey Edunov Thomas Scialom\x03\nGenAI, Meta\nAbstract\nIn this work, we develop and release Llama 2, a collection of pretrained and ﬁne-tuned\nlarge language models (LLMs) ranging in scale from 7 billion to 70 billion parameters.\nOur ﬁne-tuned LLMs, called L/l.sc/a.sc/m.sc/a.sc /two.taboldstyle-C/h.sc/a.sc/t.sc , are optimized for dialogue use cases. Our\nmodels outperform open-source chat models on most benchmarks we tested, and based on\nourhumanevaluationsforhelpfulnessandsafety,maybeasuitablesubstituteforclosedsource models. We provide a detailed description of our approach to ﬁne-tuning and safety"
#     # ab = "PURPOSE: Numerous uveitis articles were published in this century, underneath which hides valuable intelligence. We aimed to characterize the evolution and patterns in this field. METHODS: We divided the 15,994 uveitis papers into four consecutive time periods for bibliometric analysis, and applied latent Dirichlet allocation topic modeling and machine learning techniques to the latest period. . RESULTS: The yearly publication pattern fitted the curve: 1.21335x(2) - 4,848.95282x + 4,844,935.58876 (R(2) = 0.98311). The USA, the most productive country/region, focused on topics like ankylosing spondylitis and biologic therapy, whereas China (mainland) focused on topics like OCT and Behcet disease. The logistic regression showed the highest accuracy (71.6%) in the test set. CONCLUSION: In this century, a growing number of countries/regions/authors/journals are involved in the uveitis study, promoting the scientific output and thematic evolution. Our pioneering study uncovers the evolving academic trends and frontier patterns in this field using bibliometric analysis and AI algorithms."
#     # chunks = custom_text_splitter([ab], 256)
#     # print(chunks)