import requests
import re
import urllib.request
#from bs4 import BeautifulSoup
#from collections import deque
#from html.parser import HTMLParser
#from urllib.parse import urlparse
import os
import pandas as pd
import numpy as np
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search

from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
# models
EMBEDDING_MODEL = "text-embedding-ada-002"
#GPT_MODEL = "gpt-3.5-turbo"

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    print("Start of strings_and_relatednesses logic")
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["combined"], relatedness_fn(query_embedding, row["embeddings"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    print("after strings_and_relatednesses logic")
    return strings[:top_n], relatednesses[:top_n]

# examples
#strings, relatednesses = strings_ranked_by_relatedness("curling gold medal", df, top_n=5)
#for string, relatedness in zip(strings, relatednesses):
#    print(f"{relatedness=:.3f}")
#    display(string)

def num_tokens(text: str) -> int:
    """Return the number of tokens in a string."""
    #print("Start of num_tokens func")
    #encoding = tiktoken.encoding_for_model(model)
    embedding_encoding = "cl100k_base"
    encoding = tiktoken.get_encoding(embedding_encoding)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    print("Start of query_message")
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below text on the QS Articles, Rankings, Events, Course Matching Tool, etc. to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    #question = f"\n\nQuestion: {query}"
    message = introduction
    print("After strings_ranked_by_relatedness")
    for string in strings:
        next_article = f'\n\nQS Information:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article)
            > token_budget
        ):
            break
        else:
            message += next_article
    
    print("End of query_message")
    return message


def answer_question_from_embeddings(
    question: str,
) -> str:
    
    # Subtracting 500 tokens for Question text and some buffer.
    prompt_token_budget = 2048 - 500
    print_message=True
    model="text-davinci-003"
    max_tokens=1500
    stop_sequence=None

    df=pd.read_csv('data/wur_ranking_summary_embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

    print("After Embed Read")

    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    context = query_message(question, df, model=model, token_budget=prompt_token_budget)
    if print_message:
        print(context)
    '''messages = [
        {"role": "system", "content": "You answer questions about QS Rankings, Events, Course Matching Tool, Articles."},
        {"role": "user", "content": message},
    ]'''
    try:
        # Create a completions using the questin and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0.1,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
            n=1
        )
        print(response["choices"][0]["text"].strip())
        return response["choices"][0]
    except Exception as e:
        print(e)
        return ""
'''    response = openai.Completion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]
    print(response_message)
    return response_message
'''
#answer_question_from_embeddings('What is the rank of Tokyo University of Agriculture and Technology?')
#ask('Which athletes won the gold medal in curling at the 2022 Winter Olympics?')