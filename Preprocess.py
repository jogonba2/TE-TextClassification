#!/usr/bin/python
# -*- coding: utf-8 -*-
import unicodedata as uc
import re

class Preprocess:

    @staticmethod
    def preprocess(x, tokenizer):
        # 1) reemplazar ñ por ny
        x = x.replace("ñ", "ny").replace("Ñ", "ny")
        # 2) Eliminar ¿?¡!
        x = x.replace("?","").replace("¿","").replace("¡","").replace("!","")
        # 3) Eliminar acentos
        x = Preprocess.remove_accents(x)
        # 4) Tokenizado
        x = tokenizer.tokenize(x)
        # 5) Menciones, hashtags y urls a token de clase
        x = [Preprocess.replace_entity(w) for w in x]
        # 6) Minusculas
        x = " ".join(x).lower()
        # 7) Elongated words
        x = re.sub(r'(.)\1+', r'\1\1', x)
        return x

    @staticmethod
    def replace_entity(w):
        if w[0] == "#": return "hashtag"
        elif w[0] == "@": return "user"
        elif "http" in w or "www." in w: return "url"
        else: return w

    @staticmethod
    def remove_accents(sequence):
        return "".join([c for c in uc.normalize('NFD', sequence) if uc.category(c) != 'Mn'])

