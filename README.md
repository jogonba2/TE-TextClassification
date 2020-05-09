If you use this work, please cite the following references:

```
@article{GONZALEZ2020102262,
title = "Transformer based contextualization of pre-trained word embeddings for irony detection in Twitter",
journal = "Information Processing & Management",
volume = "57",
number = "4",
pages = "102262",
year = "2020",
issn = "0306-4573",
doi = "https://doi.org/10.1016/j.ipm.2020.102262",
url = "http://www.sciencedirect.com/science/article/pii/S0306457320300200",
author = "José Ángel González and Lluís-F. Hurtado and Ferran Pla",
keywords = "Irony detection, Twitter, Deep learning, Transformer encoders",
abstract = "Human communication using natural language, specially in social media, is influenced by the use of figurative language like irony. Recently, several workshops are intended to explore the task of irony detection in Twitter by using computational approaches. This paper describes a model for irony detection based on the contextualization of pre-trained Twitter word embeddings by means of the Transformer architecture. This approach is based on the same powerful architecture as BERT but, differently to it, our approach allows us to use in-domain embeddings. We performed an extensive evaluation on two corpora, one for the English language and another for the Spanish language. Our system was the first ranked system in the Spanish corpus and, to our knowledge, it has achieved the second-best result on the English corpus. These results support the correctness and adequacy of our proposal. We also studied and interpreted how the multi-head self-attention mechanisms are specialized on detecting irony by means of considering the polarity and relevance of individual words and even the relationships among words. This analysis is a first step towards understanding how the multi-head self-attention mechanisms of the Transformer architecture address the irony detection problem."
}
```

```
 @article{selfatttc,
  added-at = {},
  author = {González, José-Ángel and Hurtado, Lluís-F. and Pla, Ferran},
  biburl = {},
  ee = {},
  interhash = {},
  intrahash = {},
  journal = {Journal of Intelligent and Fuzzy Systems},
  number = None,
  pages = {},
  title = {Self-Attention for Twitter Sentiment Analysis in Spanish},
  url = {},
  volume = {},
  year = 2020
 }
```


# TASS2019

Este es el modelo que mejor ha funcionado en comparación a los modelos del año pasado (puede ser que ajustando los modelos del año pasado funcionen mejor que este, pero en todas las ejecuciones que he hecho con este modelo y con casi cualquier hiper-parámetro, pasa los 50 de MF1:

  * SVM-BOW  Acc: 53.01 | MF1: 42.89
  * SVM-BOC Acc: 54.04 | MF1: 39.16
  * SVM-SumaEmbeddings Acc: 54.91 | MF1: 43.21 (este no estaba en la tabla del año pasado)
  * Att-BLSTM Acc: Acc: 0.590361 | MF1: 0.488154
  * CNN Acc: 0.612737 | MF1: 0.476809
  * DAN (es-run1): Acc: 0.569707 | MF1: 0.482551
  * Transformer: Acc: 0.595525 | MF1: 0.522083


Mejor modelo (Transformer):

    Acc: 0.595525
    MF1: 0.522083
    MP: 0.529196
    MR: 0.521423
    
Conf Matrix

    N 201 30 13 22
    NEU 31 29 10 13
    NONE 16 10 30 8
    P 46 22 14 86
 
Classification Report

           precision    recall  f1-score   support

           N       0.68      0.76      0.72       266
         NEU       0.32      0.35      0.33        83
        NONE       0.45      0.47      0.46        64
           P       0.67      0.51      0.58       168

    micro avg       0.60      0.60      0.60       581
    macro avg       0.53      0.52      0.52       581
    weighted avg       0.60      0.60      0.59       581


El modelo tiene una sola capa con 6 cabezales de atención. Lo que se muestra son los 6 cabezales para cada muestra (más amarillo más peso, más morado, menos peso).

Algunas cosas que he visto:

 * El primer cabezal reacciona siempre a los usuarios (token user) y lo que hace referencia a ellos (si no está el token, ni idea)
    
 * El 2º cabezal parece reaccionar a palabras de "tiempo" (hola, saludos, manyana, dias, directo, noche, ...), pero no termino de entenderlo

 * El 5º cabezal reacciona a palabras con polaridades extremas (genial, maravilloso, horrible, ...) (cuando no hay, ni idea)
    
 * El 3º cabezal reacciona siempre a las palabra "no", "ni" (en caso de que no estén, no lo entiendo, parece controlar la negación marca los segmentos negados)
    
 * El 6º cabezal reacciona a casi todo, menos a determinantes, preposiciones, conjunciones, etc. (generalmente a palabras con "significado")

 * Si no hay palabras que tienen mucha importancia según el cabezal (negaciones, tiempos, usuarios, etc.) todos parecen reaccionar a palabras con polaridad alta (bien positiva o negativa)
 
 * Para las clases NEU y NONE, las atenciones forman patrones complicados de entender, para las P y N suelen marcar palabras con polaridades altas y se entienden mejor las atenciones (aunque en algunos casos, también son complicados).
    
    
Atenciones para varias muestras del conjunto de validación:

Muestra 12 (Pred:N Truth:N)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_12.png)
---

Muestra 128 (Pred:NONE Truth:NEU)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_128.png)
---

Muestra 13 (Pred:NEU Truth:NEU)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_13.png)
---

Muestra 141 (Pred:N Truth:N)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_141.png)
---

Muestra 19 (Pred:N Truth:N)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_19.png)
---

Muestra 222 (Pred:N Truth:N)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_222.png)
---

Muestra 30 (Pred:N Truth:N)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_30.png)
---

Muestra 505 (Pred:N Truth:N)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_505.png)
---

Muestra 508 (Pred:P Truth:N)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_508.png)
---

Muestra 99 (Pred:NEU Truth:NEU)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_99.png)

---

Muestra 0 (Pred:N Truth:NONE)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_0.png)
---

Muestra 1 (Pred:N Truth:N)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_1.png)
---

Muestra 136 (Pred:N Truth:P)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_136.png)
---

Muestra 17 (Pred:P Truth:P)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_17.png)
---
