from keras import backend as K

def macro_f1(y_true, y_pred, epsilon = 1e-16, beta = 1, n_classes = 4):
        y_p = K.one_hot(K.argmax(y_pred, axis=1), n_classes)
        tp = K.sum(y_true * y_p, axis=0)
        fp = K.sum((1. - y_true) * y_p, axis=0)
        fn = K.sum(y_true * (1. - y_p), axis=0)
        pr = (tp / (tp + fp + epsilon) + epsilon)
        rc = (tp / (tp + fn + epsilon) + epsilon)
        f1 = ((1. + (beta**2)) * pr * rc) / ((beta**2) * pr + rc)
        return K.mean(f1)
