import datasets
import openai
import streamlit as st
import numpy as np
import requests

from huggingface_hub import InferenceClient

openai.api_key = st.secrets["OPENAI_API_TOKEN"]

@st.cache_data
def load_data():
    dataset = datasets.load_dataset('hxyue1/theology_qa')['train']
    dataset = dataset.add_faiss_index(column='embeddings')
    return dataset

def get_embeddings(question):
    client = InferenceClient(model="BAAI/bge-large-en-v1.5", token=st.secrets["HF_API_TOKEN"])

    # Get embeddings with automatic retry
    embedding = client.feature_extraction(question)
    return embedding

def query_chat_gpt(sources, question):
    user_content = f"\"\"\"{sources}\"\"\"\nQuestion: {question}"
    gpt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You will be provided with a document delimited by triple quotes and a question. Your task is to answer the question using only the provided document. If the document does not contain the information needed to answer this question then suggest the user to ask a more specific question, but do not make reference to the document. If the question requires an opinion or subjective evaluation, tell the user that you do not answer subjective questions and suggest that they ask a more factual question, but do not reference the document."},
            {"role": "user", "content":user_content },
        ]
    )
    return gpt_response

def run():
    dataset = load_data()

    st.title("Ask Theology")
    question = st.text_input("What would you like to ask?", placeholder="e.g. What is the doctrine of justification?")
        
    response = ""
    if question:
        response = get_embeddings(question)


    if isinstance(response, str): 
        print(response)
                
    elif isinstance(response, np.ndarray):
        try:
            scores, samples = dataset.get_nearest_examples("embeddings", response, k=5)
        except ValueError:
            print(response)

        sources_raw =' \n '.join(samples['chunked'])
        sources = sources_raw.replace('\n', ' ').replace('  ', '')

        # Query chatgpt
        st.header("Response from ChatGPT")
        gpt_response = query_chat_gpt(sources, question)
        st.write(gpt_response['choices'][0]['message']['content'])

        st.subheader("Were you satisfied with the response?")
        st.markdown("Please provide feedback [here](https://docs.google.com/forms/d/e/1FAIpQLSdiQ5t111iM4-s7o3LraaLjHTsNLCY5vnYA-f38fXVVcgQKug/viewform?usp=sf_link)")

        # Displaying source documents
        st.header("Further Reading")
        for title, text in zip(samples["title"], samples["chunked"]):
            st.subheader(title)
            st.write(text)

if __name__  == "__main__":
    run()
