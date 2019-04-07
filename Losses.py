from keras import backend as K


def macro_f1(normalize=False, epsilon=1e-16, beta=1):

    def f_loss(y_true, y_pred):
        if normalize:
            y_pred = max_min(y_pred)
        tp = K.sum(y_true * y_pred, axis=0)
        fp = K.sum((1. - y_true) * y_pred, axis=0)
        fn = K.sum(y_true * (1. - y_pred), axis=0)
        pr = (tp / (tp + fp + epsilon) + epsilon)
        rc = (tp / (tp + fn + epsilon) + epsilon)
        f1 = ((1. + (beta**2)) * pr * rc) / ((beta**2) * pr + rc)
        return -K.mean(f1)

    return f_loss

