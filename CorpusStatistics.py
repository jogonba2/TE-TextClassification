# https://stackoverflow.com/questions/47624742/how-to-use-stanford-word-tokenizer-in-nltk
from nltk import tokenize
import json, jsonlines


path = "./release/tok_train.jsonl"
n_samples = 0.
n_sentences_article = 0.
n_word_sent_article = 0.
n_sentences_summary = 0.
n_word_sent_summary = 0.

with jsonlines.open(path) as fr:
    for ln in fr:
        txt = ln["text"]
        sum = ln["summary"]
        n_samples += 1
        n_sentences_article += len(txt.split(" . "))
        n_sentences_summary += len(sum.split(" . "))
        n_word_sent_article += len(txt.split(" "))
        n_word_sent_summary += len(sum.split(" "))
fr.close()

print("Article avg sents: %.3f" % (n_sentences_article / n_samples))
print("Summary avg sents: %.3f" % (n_sentences_summary / n_samples))
print("Article avg words/sent: %.3f" % (n_word_sent_article / n_sentences_article))
print("Summary avg words/sent: %.3f" % (n_word_sent_summary / n_sentences_summary))
print("Total: %.3f samples" % n_samples)


"""
Article avg sents: 29.909
Summary avg sents: 1.403
Article avg words/sent: 25.864
Summary avg words/sent: 21.649
Total: 995041.000 samples
"""
