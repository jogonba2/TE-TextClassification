from SentenceEncoderBlock import SentenceEncoderBlock
from PositionalEncoding import PositionalEncoding
from MyMasking import MyMasking
from LayerNormalization import LayerNormalization
from keras.models import Model
from keras import backend as K
from keras.layers import (Input, GlobalMaxPooling1D, Dense, GlobalAveragePooling1D,
                          Masking, Embedding, Flatten,
                          TimeDistributed, Lambda, Concatenate, SpatialDropout1D, Dropout)
import Losses
import Metrics

class TransformerEncoder():

    def __init__(self, max_words = 25,
                 embedding_dims = 300,
                 output_encoder_dims = [300, 300, 300],
                 attention_dims = [64, 64, 64],
                 n_heads = [8, 8, 8],
                 dropout_input = 0.,
                 dropout_output = 0.,
                 pe = True,
                 dim_h = 256,
                 final_h = True,
                 pool_mode = "max"):

        self.max_words = max_words
        self.embedding_dims = embedding_dims
        self.output_encoder_dims = output_encoder_dims
        self.attention_dims = attention_dims
        self.n_heads = n_heads
        self.pe = pe
        self.n_encoders = len(self.output_encoder_dims)
        self.dropout_output = dropout_output
        self.dropout_input = dropout_input
        self.dim_h = dim_h
        self.final_h = final_h
        self.pool_mode = pool_mode

    def build(self):

        self.input_document = Input(shape=(self.max_words, self.embedding_dims))
        self.mask = Input(shape=(self.max_words,))
        self.pos_encoding = Input(shape=(self.max_words, self.embedding_dims))

        # Padding #
        self.z_input = MyMasking()(self.input_document, mask = self.mask)

        # Positional Encoding#
        if self.pe:
            self.z_input = PositionalEncoding()(self.z_input, mask = self.pos_encoding)

        # Dropout at input (sentence level)
        self.z_input = SpatialDropout1D(self.dropout_input)(self.z_input)

        self.all_attns = []
        ant_layer = self.z_input
        for i in range(self.n_encoders):
            self.sentence_encoder = SentenceEncoderBlock(self.output_encoder_dims[i],
                                                         self.attention_dims[i],
                                                         self.n_heads[i], dropout = self.dropout_output)
            self.document_encoder = self.sentence_encoder(ant_layer)
            self.z_encoder = Lambda(lambda x: x[0])(self.document_encoder)
            self.attn_encoder = Lambda(lambda x: x[1])(self.document_encoder)

            self.all_attns.append(self.attn_encoder)

            # Masking entre cada capa #
            self.z_encoder = MyMasking()(self.z_encoder, mask = self.mask)

            ant_layers = (self.z_encoder)

        # Prepare all attentions #
        if self.n_encoders > 1:
            self.all_attns = [Lambda(lambda a: K.expand_dims(a, 1))(x) for x in self.all_attns]
            self.all_attns = Concatenate(axis=1)(self.all_attns)

        ##########################

        if self.pool_mode == "max":
            self.z_encoder = GlobalMaxPooling1D()(self.z_encoder)

        else:
            self.z_encoder = GlobalAveragePooling1D()(self.z_encoder)

        #self.z_encoder = Dropout(0.3)(self.z_encoder) # En el mejor, no estaba
        self.z_encoder = LayerNormalization()(self.z_encoder) # En el mejor, esto activado!

        self.h = self.z_encoder

        if self.final_h:
            self.h = Dense(self.dim_h, activation="relu")(self.h)
            #self.h = Dropout(0.3)(self.h) # Este no está en el mejor
            self.h = LayerNormalization()(self.h) # En el mejor, esto activado!

        self.output = Dense(4, activation="softmax")(self.h)

        self.model = Model(inputs = [self.input_document, self.mask, self.pos_encoding],
                           outputs = [self.output])

        self.attn_model = Model(inputs = [self.input_document, self.mask, self.pos_encoding],
                                outputs = self.all_attns)

    def compile(self, model):
        model.compile(optimizer="adam", # adam
                      loss = "categorical_crossentropy",
                      metrics = ["accuracy"]) #, Metrics.macro_f1])

    def save(self, model, f_name):
        model.save_weights(f_name)

    def load(self, model, f_name):
        model.load_weights(f_name)

    def __str__(self):
        pass
