################################################################################
### Step 1
################################################################################

import requests
import re
import urllib.request
#from bs4 import BeautifulSoup
#from collections import deque
#from html.parser import HTMLParser
#from urllib.parse import urlparse
import os
import pandas as pd
#import tiktoken
import openai
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

################################################################################
### Step 12
################################################################################

def create_context(
    question, df, max_len=1800, size="ada"
):
    try:
        #print(openai.api_key)
        """
        Create a context for a question by finding the most similar context from the dataframe
        """

        # Get the embeddings for the question
        q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

        # Get the distances from the embeddings
        df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


        returns = []
        cur_len = 0

        # Sort by distance and add the text to the context until the context is too long
        for i, row in df.sort_values('distances', ascending=True).iterrows():
            
            # Add the length of the text to the current length
            cur_len += row['n_tokens'] + 4
            
            # If the context is too long, break
            if cur_len > max_len:
                break
            
            # Else add it to the text that is being returned
            returns.append(row['combined'])

        # Return the context
        return "\n\n###\n\n".join(returns)
    except Exception as e:
        print(e)
        return str(e)

def answer_question_from_embeddings(
    question="Which are the best universities to study in london?"
):
    try:
        model="text-davinci-003"
        #max_len=1800
        # Subtracting 500 tokens for Question text and some buffer.
        prompt_token_budget = 2048 - 500
        size="ada"
        debug=True
        max_tokens=1500
        stop_sequence=None

        df=pd.read_csv('data/wur_ranking_summary_embeddings.csv', index_col=0)
        df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

        print("After reading embeddings in normal way")

        """
        Answer a question based on the most similar context from the dataframe texts
        """
        context = create_context(
            question,
            df,
            max_len=prompt_token_budget,
            size=size,
        )
        # If debug, print the raw model response
        if debug:
            print("Context:\n" + context)
            print("\n\n")


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
        return str(e)

################################################################################
### Step 13
################################################################################

#print("before answers")
#print(answer_question(df, question="Which are the best gluten free healthy bars?", debug=True))
#print(answer_question(df, question="Can you recommend a good product for itchy skin?"))
#print(answer_question(df, question="Name the best candy bars?"))

'''print(answer_question_from_embeddings(df, question="Which is the best university in United Kingdom?", debug=False))
print(answer_question_from_embeddings(df, question="Where are the campuses of New York University?"))
print(answer_question_from_embeddings(df, question="What is the score and rank of Brown University?"))'''
#print(answer_question_from_embeddings(question="What is the rank of Tokyo University of Agriculture and Technology?"))
