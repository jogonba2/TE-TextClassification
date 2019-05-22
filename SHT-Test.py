from StringProcessing import StringProcessing
from BuildVocabulary import BuildVocabulary as bv
from TransformerEncoder import TransformerEncoder
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from nltk.tokenize.toktok import ToktokTokenizer
from Visualization import Visualization
from gensim.models import Word2Vec
from Preprocess import Preprocess
from Utils import Utils
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import numpy as np
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
""" MEJOR
Acc: 0.595525
MF1: 0.522083
    # Text parameters #
    max_words = 50
    test_path = "./tass-corpus/es/dev.csv"

    # Training Parameters #
    path_model = "./best-models/transformer_1646NO.hdf5"
    r_cats = {0:"N", 1:"NEU", 2:"NONE", 3:"P"}
    tokenizer = ToktokTokenizer()
    w2v_path = "./twitter87/twitter87.model"
    w2v = Word2Vec.load(w2v_path)

    # Encoder Parameters # # MEJOR
    dropout_input = 0.7 #0.7
    dropout_output = 0. # 0.
    pe = False #False
    embedding_dims = w2v.vector_size
    n_encoders = 1 #2 #1
    attention_dims = 64 #32 #64
    n_heads = 6 #8
    dim_h = 128 #256
    final_h = False #False
    pool_mode = "average" #"average"
"""

if __name__ == "__main__":

    # Text parameters #
    max_words = 50
    test_path = "./tass-corpus/es/dev.csv"

    # Training Parameters #
    path_model = "./best-models/transformer_1648NO.hdf5"
    r_cats = {0:"N", 1:"NEU", 2:"NONE", 3:"P"}
    tokenizer = ToktokTokenizer()
    w2v_path = "./twitter87/twitter87.model"
    w2v = Word2Vec.load(w2v_path)

    # Encoder Parameters # # MEJOR
    dropout_input = 0.7 #0.7
    dropout_output = 0. # 0.
    pe = False #False
    embedding_dims = w2v.vector_size
    n_encoders = 1 #2 #1
    attention_dims = 64 #32 #64
    n_heads = 8 #8
    dim_h = 128 #256
    final_h = False #False
    pool_mode = "average" #"average"

    output_encoder_dims = [embedding_dims for i in range(n_encoders)]
    attention_dims = [attention_dims for i in range(n_encoders)]
    n_heads = [n_heads for i in range(n_encoders)]

    ids_dv, x_dv, y_dv = StringProcessing.load_samples(test_path)

    # Preprocess #
    x_dv = [Preprocess.preprocess(x, tokenizer) for x in x_dv]

    # Represent #
    rx_dv = np.array(StringProcessing.represent_documents(x_dv, max_words, embedding_dims, w2v, word_delimiter = " "))

    # Masks #
    masks_dv = np.array([((rx!=0).sum(axis=1)>0).astype("int") for rx in rx_dv])

    # Positional Encodings #
    matrix_pos_encodings = Utils.precompute_sent_pos_encodings(max_words, embedding_dims)
    pe_dv = np.array([Utils.build_pe_sent_encodings(matrix_pos_encodings, m) for m in masks_dv])

    ht = TransformerEncoder(max_words = max_words,
                 embedding_dims = embedding_dims,
                 output_encoder_dims = output_encoder_dims,
                 attention_dims = attention_dims,
                 n_heads = n_heads,
                 dropout_input = dropout_input,
                 dropout_output = dropout_output,
                 pe = pe,
                 dim_h = dim_h,
                 final_h = final_h,
                 pool_mode = pool_mode)

    ht.build()
    print(ht.model.summary())
    ht.compile(ht.model)
    ht.load(ht.model, path_model)

    preds = ht.model.predict([rx_dv, masks_dv, pe_dv], batch_size=256)
    preds = [r_cats[p.argmax()] for p in preds]
    truths = y_dv
    print("Acc: %f" % accuracy_score(truths, preds))
    print("MF1: %f" % f1_score(truths, preds, average="macro"))
    print("MP: %f" % precision_score(truths, preds, average="macro"))
    print("MR: %f" % recall_score(truths, preds, average="macro"))
    print("Conf Matrix\n", confusion_matrix(truths, preds))
    print("Classification Report\n", classification_report(truths, preds))

    attns = ht.attn_model.predict([rx_dv, masks_dv, pe_dv], batch_size=256)

    inp = int(input())
    #s = input()
    # Chulas: 19 (N), 141 (N), 56 (P), 99 (P)

    while inp != -1:
    #while s != "":

        s = Preprocess.preprocess(x_dv[inp], tokenizer)
        rs = np.array(StringProcessing.represent_documents([s], max_words, embedding_dims, w2v, word_delimiter=" "))
        mask = np.array([((r != 0).sum(axis=1) > 0).astype("int") for r in rs])
        pe = np.array([Utils.build_pe_sent_encodings(matrix_pos_encodings, m) for m in mask])
        pred =  ht.model.predict([rs, mask, pe], batch_size=256)
        attn_i = ht.attn_model.predict([rs, mask, pe], batch_size=256)[0]
        l_words = s.split()
        print("Pred: %s" % (pred))
        print("Truth: %s" % (y_dv[inp]))
        pad = 0
        while len(l_words) < max_words:
            l_words.insert(0, "<pad>")
            pad += 1
        l_words = l_words[pad:]
        attn_i = attn_i[:, pad:, pad:]
        attn_i = attn_i[2]
        #attn_i = attn_i.sum(axis=0)
        output_file = "sentiment_att.pdf"
        Visualization.visualize_attentions(attn_i, 30, 30, rows=1, columns=1, ticks=l_words, output_file=output_file,
                                           save=True)
        inp = int(input())

        """
        attn_i = attns[inp]
        l_words = x_dv[inp].split()
        print("Pred: %s, True: %s" % (preds[inp], truths[inp]))
        pad = 0
        while len(l_words) < max_words:
            l_words.insert(0, "<pad>")
            pad += 1
        l_words = l_words[pad:]
        attn_i = attn_i[:, pad:, pad:]
        #attn_i = attn_i.sum(axis=0)
        output_file = "ejemplo_%d.png" % inp
        Visualization.visualize_attentions(attn_i, 40, 40, rows = 2, columns = 4, ticks = l_words, output_file = output_file, save = False)
        inp = int(input())
        """