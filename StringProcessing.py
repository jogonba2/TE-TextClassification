from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np


class StringProcessing:

    @staticmethod
    def load_samples(csv_path):
        f = pd.read_csv(csv_path, delimiter = "\t", encoding="utf8")
        return f["ID"].tolist(), f["TWEET"].tolist(), f["CLASS"].tolist()

    @staticmethod
    def represent_documents(documents, max_words, d_emb,
                            w2v, word_delimiter = " "):

        return [StringProcessing.represent_document(document, max_words,
                                                    d_emb, w2v, word_delimiter)
                for document in documents]

    # Padding PRE, Truncating POST #
    @staticmethod
    def represent_document(document, max_words, d_emb,
                           w2v, word_delimiter = " "):

        repr = []
        s = document.split()
        for w in s:
            if w in w2v:
                repr.append(w2v[w])
            else:
                repr.append(w2v["unk"])
        return pad_sequences([repr], maxlen = max_words, dtype='float32', padding='pre', truncating='post', value=0.0)[0]
