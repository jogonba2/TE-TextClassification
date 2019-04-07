from StringProcessing import StringProcessing
from BuildVocabulary import BuildVocabulary as bv
from TransformerEncoder import TransformerEncoder
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from nltk.tokenize.toktok import ToktokTokenizer
from gensim.models import Word2Vec
from Preprocess import Preprocess
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from Utils import Utils
import numpy as np
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    # Text parameters #
    max_words = 50 #50
    train_path = "./tass-corpus/es/augmented_train_unique.csv" # train
    dev_path = "./tass-corpus/es/dev.csv"

    # Training Parameters #
    batch_size = 32
    epochs = 250
    path_models = "./models/"
    name_models = "transformer_1646NO"
    cats = {"N":0, "NEU":1, "NONE":2, "P":3}
    r_cats = {0:"N", 1:"NEU", 2:"NONE", 3:"P"}
    class_weight = {0:1., 1:2., 2:2., 3:1.3} # BEST
    tokenizer = ToktokTokenizer()
    w2v_path = "./twitter87/twitter87.model"
    w2v = Word2Vec.load(w2v_path)

    # Encoder Parameters # # MEJOR
    dropout_input = 0.7 #0.7
    dropout_output = 0. #0.15 # 0.
    pe = False #False
    embedding_dims = w2v.vector_size
    n_encoders = 1 #2 #1
    attention_dims = 64 #32 #64
    n_heads = 6 #8
    dim_h = 128 #256
    final_h = False #False
    pool_mode = "average" #"average"

    output_encoder_dims = [embedding_dims for i in range(n_encoders)]
    attention_dims = [attention_dims for i in range(n_encoders)]
    n_heads = [n_heads for i in range(n_encoders)]

    ids_tr, x_tr, y_tr = StringProcessing.load_samples(train_path)
    ids_dv, x_dv, y_dv = StringProcessing.load_samples(dev_path)

    ry_tr = to_categorical([cats[c] for c in y_tr], 4)
    ry_dv = to_categorical([cats[c] for c in y_dv], 4)

    # Preprocess #
    x_tr = [Preprocess.preprocess(x, tokenizer) for x in x_tr]
    x_dv = [Preprocess.preprocess(x, tokenizer) for x in x_dv]

    # Represent #
    rx_tr = np.array(StringProcessing.represent_documents(x_tr, max_words, embedding_dims, w2v, word_delimiter = " "))
    rx_dv = np.array(StringProcessing.represent_documents(x_dv, max_words, embedding_dims, w2v, word_delimiter = " "))

    # Masks #
    masks_tr = np.array([((rx!=0).sum(axis=1)>0).astype("int") for rx in rx_tr])
    masks_dv = np.array([((rx!=0).sum(axis=1)>0).astype("int") for rx in rx_dv])

    # Positional Encodings #
    matrix_pos_encodings = Utils.precompute_sent_pos_encodings(max_words, embedding_dims)
    pe_tr = np.array([Utils.build_pe_sent_encodings(matrix_pos_encodings, m) for m in masks_tr])
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

    """
    chkpath = path_models + "/" + name_models + "-{epoch:05d}-{val_loss:.3f}-{val_acc:.3f}-{val_macro_f1}.hdf5"

    checkpoint = ModelCheckpoint(chkpath, monitor='val_macro_f1',
                                 verbose=1, save_best_only=True,
                                 mode='max')
    callbacks = [checkpoint]
    ht.model.fit(x = [rx_tr, masks_tr, pe_tr], y = ry_tr,
                 epochs = epochs,
                 validation_data = ([rx_dv, masks_dv, pe_dv], ry_dv),
                 verbose = 1,
                 batch_size = batch_size, callbacks = callbacks)


    """



    truths = y_dv
    best_mf1 = float("-inf")
    for e in range(epochs):
        ht.model.fit(x = [rx_tr, masks_tr, pe_tr], y = ry_tr,
                     epochs = 1,
                     validation_data = ([rx_dv, masks_dv, pe_dv], ry_dv),
                     verbose = 0,
                     batch_size = batch_size, class_weight = class_weight)

        preds = ht.model.predict([rx_dv, masks_dv, pe_dv], batch_size = 256)
        preds = [r_cats[p.argmax()] for p in preds]
        acc = accuracy_score(truths, preds)
        mf1 = f1_score(truths, preds, average="macro")
        if mf1 > best_mf1:
            best_mf1 = mf1
            acc = accuracy_score(truths, preds)
            mp = precision_score(truths, preds, average="macro")
            mr = recall_score(truths, preds, average="macro")
            print("BEST: %d" % e)
            print("Acc: %f" % acc)
            print("MF1: %f" % mf1)
            print("MP: %f" % mp)
            print("MR: %f" % mr)
            ht.model.save_weights(path_models + "/" + name_models + ".hdf5")
            print("\n\n" + "-"*50 + "\n\n")
