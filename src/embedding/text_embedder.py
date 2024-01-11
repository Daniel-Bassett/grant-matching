import json
import requests
import re
import string
import pandas as pd
import numpy as np
from openai import OpenAI
import streamlit as st

import spacy
import nltk
from nltk.corpus import stopwords
# nltk.download('punkt')
# nltk.download('stopwords')


class TextProcessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')    
    
    def remove_duplicate_lines(self, text):
        return (pd.Series(text.split('\n'))
                .drop_duplicates()
                .to_list())

    def normalize_text(self, text):
        text = ''.join(text)
        text = text.lower()
        text = ''.join([char if char not in '“”.!?,—():-/;–•' else ' ' for char in text])
        text = ''.join([char for char in text if char not in string.punctuation])
        text = re.sub('\s{2,}', ' ', text)
        return text

    def tokenize_and_remove_stopwords(self, text):
        return [word for word in text.split(' ') if word not in stopwords.words('english')]
    
    def lemmatize_tokens(self, tokens):
        doc = self.nlp(' '.join(tokens))
        tokens = [token.lemma_ for token in doc]
        tokens = [token for token in tokens if not token.isdigit() and len(token) > 1]
        return tokens

    def generate_tokens(self, text):
        text = self.remove_duplicate_lines(text)
        text = self.normalize_text(text)
        tokens = self.tokenize_and_remove_stopwords(text)
        tokens = self.lemmatize_tokens(tokens)
        return tokens

    def generate_ngrams(self, tokens, num_ngrams):
        ngrams = zip(*[tokens[i:] for i in range(num_ngrams)])
        ngrams = [" ".join(ngram) for ngram in ngrams]
        ngrams = set(ngrams)
        ngrams = list(ngrams)
        return ngrams
