# TASS2019

Este es el modelo que mejor ha funcionado en comparación a los modelos del año pasado (puede ser que ajustando los modelos del año pasado funcionen mejor que este, pero en todas las ejecuciones que he hecho con este modelo y con casi cualquier hiper-parámetro, pasa los 50 de MF1:

  * SVM-BOW  Acc: 53.01 | MF1: 42.89
  * SVM-BOC Acc: 54.04 | MF1: 39.16
  * SVM-SumaEmbeddings Acc: 54.91 | MF1: 43.21 (este no estaba en la tabla del año pasado)
  * Att-BLSTM Acc: Acc: 0.590361 | MF1: 0.488154
  * CNN Acc: 0.612737 | MF1: 0.476809
  * DAN (es-run1): Acc: 0.569707 | MF1: 0.482551
  * Transformer: Acc: 0.595525 | MF1: 0.522083

El modelo tiene una sola capa con 6 cabezales de atención. Lo que se muestra son los 6 cabezales para cada muestra (más amarillo más peso, más morado, menos peso).

Algunas cosas que he visto:

 * El primer cabezal reacciona siempre a los usuarios (token user) y lo que hace referencia a ellos (si no está el token, ni idea)
    
 * El 2º cabezal parece reaccionar a palabras de "tiempo" (hola, saludos, manyana, dias, directo, noche, ...), pero no termino de entenderlo

 * El 5º cabezal reacciona a palabras con polaridades extremas (genial, maravilloso, horrible, ...) (cuando no hay, ni idea)
    
 * El 3º cabezal reacciona siempre a las palabra "no", "ni" (en caso de que no estén, no lo entiendo, parece controlar la negación marca los segmentos negados)
    
 * El 6º cabezal reacciona a casi todo_, parece componer las palabras de alguna manera.

 * Si no hay palabras que tienen mucha importancia según el cabezal (negaciones, tiempos, usuarios, etc.) parece reaccionar a palabras con polaridad alta (bien positiva o negativa)
    
    * Para las clases NEU y NONE, los patrones forman "cuadros" complicados de entender, para las P y N suelen marcar palabras con polaridades altas y se entienden mejor las atenciones
    
    
Atenciones para varias muestras del conjunto de validación:

12)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_12.png)
---

128)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_128.png)
---

13)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_13.png)
---

141)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_141.png)
---

19)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_19.png)
---

222)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_222.png)
---

30)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_30.png)
---

505)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_505.png)
---

508)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_508.png)
---

99)
![alt text](https://github.com/jogonba2/TASS2019/blob/master/figures/ejemplo_99.png)
