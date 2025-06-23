import os
import re

from transformers import AutoTokenizer, AutoModel
import datasets
import huggingface_hub
import pandas as pd
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter


def read_files(directory, txt_files):
    book_data_list = []
    regex_pattern = re.compile(r"(.*) by (.*).txt")
    for file_name in txt_files:
        with open(directory + file_name, encoding="utf8") as f:

            # Get title and author
            match = re.match(regex_pattern, file_name)
            if match:
                title = match[1]
                authors = match[2]

                book_data = {}
                book_data["title"] = title
                book_data["authors"] = authors
                book_data["text"] = f.read()
                book_data_list.append(book_data)

    return book_data_list


def document_chunking(document, minimum_chunk_size=None):
    """
    document (str)
    minimum_chunk_size (int)
    """

    # Split text by full stop, but add fullstop at the end of each str
    sentences = document.split(".")
    sentences = [sentence + "." for sentence in sentences]

    # Accumulate sentences until minimum_chunk_size reached
    chunked_sentences = []
    chunk = ""
    for sentence in sentences:
        # Add sentence if chunk below threshold
        if (
            len(chunk) < minimum_chunk_size
            and len(chunk + sentence) < 2 * minimum_chunk_size
        ):
            chunk += sentence
        # Otherwise store chunk and reset chunk
        else:
            chunked_sentences.append(chunk)
            # Check sentence length
            if len(sentence) > 2 * minimum_chunk_size:
                sentence_split = [
                    sentence[i : i + minimum_chunk_size]
                    for i in range(0, len(sentence), 2 * minimum_chunk_size)
                ]
                chunked_sentences.extend(sentence_split)
            chunk = ""

    return chunked_sentences


def convert_to_dataset(data_dict):
    """Convert dictionary to hf dataset"""
    data_dict_df = pd.DataFrame(data_dict)
    data_dict_df = data_dict_df.drop("text", axis=1)
    data_dict_df = data_dict_df.fillna(method="ffill")
    data_hf = datasets.Dataset.from_pandas(data_dict_df)
    return data_hf


def tokenizer_len(text):
    tokens = tokenizer.encode(
        text,
    )
    return len(tokens)


def generate_embeddings(model, tokenizer, hf_dataset):

    def cls_pooling(model_output):
        return model_output.last_hidden_state[:, 0]

    def get_embeddings(text_list):
        encoded_input = tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        encoded_input = {k: v for k, v in encoded_input.items()}
        model_output = model(**encoded_input)
        return cls_pooling(model_output)

    with torch.no_grad():
        embeddings_dataset = hf_dataset.map(
            lambda x: {"embeddings": get_embeddings(x["chunked"]).cpu().numpy()[0]}
        )

    return embeddings_dataset


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_ckpt = "BAAI/bge-large-en-v1.5"
    # model_ckpt = 'thenlper/gte-large'
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, max_length=512)
    model = AutoModel.from_pretrained(model_ckpt, max_length=512).to(device)
    directory = "c:/Users/hxyue/Documents/python-projects/LLM/data/raw/"

    txt_files = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            txt_files.append(file)

    book_data_list = read_files(directory, txt_files)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=50,
        # length_function = tokenizer_len,
        length_function=len,
        is_separator_regex=False,
    )

    # Convert to hf dataset
    hf_ds_list = []
    for book_data in book_data_list:
        print(book_data["title"])
        book_data["chunked"] = text_splitter.split_text(book_data["text"])
        hf_ds_list.append(convert_to_dataset(book_data))

    ds_with_embeddings = []
    for ds in hf_ds_list:
        ds_with_embeddings.append(generate_embeddings(model, tokenizer, ds))

    dataset_full = datasets.concatenate_datasets(ds_with_embeddings)
    dataset_full.save_to_disk(directory + "ask_theology")
    dataset_full.push_to_hub("hxyue1/ask_theology")
    huggingface_hub.create_tag("hxyue1/ask_theology", tag="0.4", repo_type="dataset")
