# imports
import pandas as pd
from config import *
try:
    import tiktoken
    print("tiktoken is installed and accessible.")
except ImportError:
    print("tiktoken is not installed or not accessible.")
    
import openai
import os
from flask import jsonify
openai.api_key = OPENAI_API_KEY

from openai.embeddings_utils import get_embedding

def create_embeddings():

    try:    
        embedding_model = "text-embedding-ada-002"
        embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
        max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
        # load & inspect dataset
        input_datapath = "data/wur_ranking_summary.csv"  # to save space, we provide a pre-filtered dataset
        df = pd.read_csv(input_datapath, index_col=None)
        df = df[["Entity_id","Category","Summary"]]
        df = df.dropna()
        df["combined"] = (
            "Title: " + df.Category.str.strip() + "; Content: " + df.Summary.str.strip()
        )
        df.head(2)
        print("After combining")
        top_n = 400
        #df = df.sort_values("Time").tail(top_n * 2)  # first cut to first 2k entries, assuming less than half will be filtered out
        #df.drop("Time", axis=1, inplace=True)

        encoding = tiktoken.get_encoding(embedding_encoding)

        # omit reviews that are too long to embed
        df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
        df = df[df.n_tokens <= max_tokens].tail(top_n)
        len(df)
        print("After tokenisation")

        # Ensure you have your API key set in your environment per the README: https://github.com/openai/openai-python#usage

        # This may take a few minutes
        df["embeddings"] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model))
        df.to_csv("data/wur_ranking_summary_embeddings.csv")
        print("After Embeddings")
        return jsonify({"embeddings created": True})
    except Exception as e:
        print(e)
        return ""


