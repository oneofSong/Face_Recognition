from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy import savez_compressed
from numpy import expand_dims
import sys, os
from random import choice

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from mtcnn.mtcnn import MTCNN
import keras
import cv2

class modelHandler():
    def __init__(self):
        self.model = load_model("../00.Resource/model/facenet_keras.h5")
        # print("model.layers len :: ", len(self.model.layers))
        # print("model.layers[0] :: ", self.model.layers[0])
        # self.layer = self.model.layers[0]
        # print(self.layer)
        # print("asdasd")
        # self.w = self.layer.get_weights()
        # print("w.type")
        # print(type(self.w))
        # print("w.len")
        # print(len(self.w))
        # print("model.layers[0].w :: ", self.w[0])
        # print("model.layers[0].b :: ", self.w[1])
        # print("slkdjfh;asgdlh")

    def extractModel(self, saveNm):
        """
        weights = model.layers[0].get_weights()[0]
        biases = model.layers[0].get_weights()[1]
        :param saveNm:
        :return:
        """
        self.model.save_weights("./{}.h5".format(saveNm))

    def getWeights(self):
        for layer in self.model.layers:
            weight = layer.get_weights()
            print(weight)
            print("===================================================")

if __name__ == "__main__":
    print("start model handle")
    handler = modelHandler()
    handler.getWeights()
    # handler.extractModel()