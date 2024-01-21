import os
import cv2
import math
import time
import numpy as np
import tqdm
import skimage
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn import neural_network


class Classifier(object):
    def __init__(self, filePath):
        self.filePath = filePath

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dic = pickle.load(fo, encoding='bytes')
        return dic

    def get_data(self):
        TrainData = []
        TestData = []
        for b in range(1, 6):
            f = os.path.join(self.filePath, 'data_batch_%d' % (b,))
            data = self.unpickle(f)
            train = np.reshape(data[b'data'], (10000, 3, 32 * 32))
            labels = np.reshape(data[b'labels'], (10000, 1))
            fileNames = np.reshape(data[b'filenames'], (10000, 1))
            datalebels = zip(train, labels, fileNames)
            TrainData.extend(datalebels)
        f = os.path.join(self.filePath, 'test_batch')
        data = self.unpickle(f)
        test = np.reshape(data[b'data'], (10000, 3, 32 * 32))
        labels = np.reshape(data[b'labels'], (10000, 1))
        fileNames = np.reshape(data[b'filenames'], (10000, 1))
        TestData.extend(zip(test, labels, fileNames))

        print("data read finished!")
        return TrainData, TestData

    def get_feat(self, TrainData, TestData):
        train_feat = []
        test_feat = []
        for data in tqdm.tqdm(TestData):
            image = np.reshape(data[0].T, (32, 32, 3))
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255.
            fd = hog(gray, 9, [8, 8], [2, 2])
            fd = np.concatenate((fd, data[1]))
            test_feat.append(fd)
        test_feat = np.array(test_feat)
        print(test_feat.shape)
        exit()
        np.save("test_feat.npy", test_feat)
        print("Test features are extracted and saved.")
        for data in tqdm.tqdm(TrainData):
            image = np.reshape(data[0].T, (32, 32, 3))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.
            fd = hog(gray, 9, [8, 8], [2, 2])
            fd = np.concatenate((fd, data[1]))
            train_feat.append(fd)
        train_feat = np.array(train_feat)
        np.save("train_feat.npy", train_feat)
        print("Train features are extracted and saved.")
        return train_feat, test_feat

    def classification(self, train_feat, test_feat):
        t0 = time.time()
        clf = neural_network.MLPClassifier()
        print("Training a MLP Classifier.")
        clf.fit(train_feat[:, :-1], train_feat[:, -1])
        predict_result = clf.predict(test_feat[:, :-1])
        num = 0
        for i in range(len(predict_result)):
            if int(predict_result[i]) == int(test_feat[i, -1]):
                num += 1
        rate = float(num) / len(predict_result)
        t1 = time.time()
        print('The testing classification accuracy is %f' % rate)
        print('The testing cast of time is :%f' % (t1 - t0))

        predict_result2 = clf.predict(train_feat[:, :-1])
        num2 = 0
        for i in range(len(predict_result2)):
            if int(predict_result2[i]) == int(train_feat[i, -1]):
                num2 += 1
        rate2 = float(num2) / len(predict_result2)
        print('The Training classification accuracy is %f' % rate2)

    def run(self):
        if os.path.exists("train_feat.npy") and os.path.exists("test_feat.npy"):
            train_feat = np.load("train_feat.npy")
            test_feat = np.load("test_feat.npy")
        else:
            TrainData, TestData = self.get_data()
            train_feat, test_feat = self.get_feat(TrainData, TestData)
        self.classification(train_feat, test_feat)


if __name__ == '__main__':
    filePath = r'../cifar-10-python/cifar-10-batches-py'
    cf = Classifier(filePath)
    cf.run()