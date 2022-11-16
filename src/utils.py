import pandas as pd
import numpy as np

def read_data(path_to_data, train=False, seed=1505):
    data = pd.read_csv(path_to_data, header=None).drop([0], axis=1, inplace=False)
    Y = None
    labels = None
    if train is True:
        data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
        labels = {lab: i for i, lab in enumerate(sorted(list(set(data[1]))))}
        Y = np.asarray(data[1].map(labels))
        data = data.drop([1], axis=1, inplace=False)
    for key in data:
        data = data.fillna(value={key: data[key].mean()}, inplace=False)
    return np.asarray(data), Y, labels