from sklearn import neural_network
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os
import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from tqdm import tqdm


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def main_worker():

    # load data
    x_train = []
    y_train = []
    for i in range(1, 6):
        train_data = unpickle(os.path.join("../cifar-10-python/cifar-10-batches-py", "data_batch_{}".format(i)))
        train = train_data[b'data']
        labels = train_data[b'labels']
        x_train.extend(train)
        y_train.extend(labels)

    x_train = np.array(x_train) / 255.0
    y_train = np.array(y_train)
    test_data = unpickle("../cifar-10-python/cifar-10-batches-py/test_batch")
    x_test, y_test = np.array(test_data[b'data']), np.array(test_data[b'labels'])
    x_test = x_test / 255.0

    # mlp = neural_network.MLPClassifier(
    #     hidden_layer_sizes=(100, ),
    #     solver='sgd',
    #     activation="relu",
    #     alpha=0.001,
    #     max_iter=200,
    #     verbose=1,
    #     early_stopping=True,
    #     tol=0.0001,
    # )
    mlp = neural_network.MLPClassifier()

    mlp.fit(x_train, y_train)
    y_pred = mlp.predict(x_test)

    num = 0
    for i in range(len(y_pred)):
        if int(y_pred[i]) == int(y_test[i]):
            num += 1
    rate = float(num) / len(y_pred)
    print("The testing accuracy is {}".format(rate))

    y_pred = mlp.predict(x_train)

    num = 0
    for i in range(len(y_pred)):
        if int(y_pred[i]) == int(y_train[i]):
            num += 1
    rate = float(num) / len(y_pred)
    print("The training accuracy is {}".format(rate))

    print("The train score is {}".format(mlp.score(x_train, y_train)))
    print("The test score is {}".format(mlp.score(x_test, y_test)))


if __name__ == "__main__":
    main_worker()
