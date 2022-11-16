def split_data(X, Y, perc_train=0.80):
    return X[0: int(perc_train * X.shape[0])],\
            X[int(perc_train * X.shape[0]): ],\
            Y[0: int(perc_train * Y.shape[0])],\
            Y[int(perc_train * Y.shape[0]): ]
