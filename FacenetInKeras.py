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


class facenetInKeras():
    """
    # 케라스를 이용하여 페이스넷 구현

    """
    def __init__(self):
        self.faceModel = load_model('./00.Resource/model/facenet_keras.h5')
        print(self.faceModel.inputs)
        print("==========================================================")
        print(self.faceModel.outputs)
        print("==========================================================")

    def detectFace(self, url):
        """
        # MTCNN 을 이용하여 이미지 내 얼굴 감지
        인자 url을 image 데이터로 변경해야함
        :return:
        """
        image = Image.open(url)
        image = image.convert('RGB')
        imagePixels = np.asarray(image)

        detector = MTCNN()
        cropFace = detector.detect_faces(imagePixels)

        plt.imshow(image)
        plt.show()

        x1, y1, width, height = cropFace[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

    def loadFaces(self, dirPath):
        """
        # 데이터셋 내 이미지 로드
        """
        face = list()
        for fileName in os.listdir(dirPath):
            path = dirPath + fileName
            img = Image.open(path)
            faceArray = np.asarray(img)
            face.append(faceArray)

        return face

    def loadDataset(self, dirPath):
        """
        # 데이터 셋 로드
        """
        x, y = list(), list()
        for subdir in os.listdir(dirPath):
            path = dirPath + '/' + subdir + '/'
            print("path :: ", path)
            if not os.path.isdir(path):
                continue

            # subdir 내 이미지 리스트리턴 / 라벨 생성
            faceList = self.loadFaces(path)
            labels = [subdir for _ in range(len(faceList))]

            print('>loaded {} examples for class: {}'.format(len(faceList), subdir))

            x.extend(faceList)
            y.extend(labels)

        return np.asarray(x), np.asarray(y)

    def getEmbedding(self, model, facePixels):
        """
        # 입력된 얼굴데이터를 모델에서 임베딩한 결과 반환
        """
        # float32 로 형변환
        facePixels = facePixels.astype('float32')

        # mean(평균), std(표준편차)
        # array.mean(, axis=(평균값을 구하기위한 축 혹은 여러개의 축 Default None(모든 요소의 평균값)))
        mean, std = facePixels.mean(), facePixels.std()
        print("facePixels.mean :: ", mean)
        print("facePixels.std :: ", std)

        facePixels = (facePixels - mean) / std
        print("facePixels = (facePixels-mean) / std :: ", facePixels)

        # transform face into one sample
        # expand_dims() 은 차원을 늘리는 함수, 반대는 squeeze()
        # param(array, axis(0=행방향, 1=열방향, 2=채널방향)
        samples = expand_dims(facePixels, axis=0)
        yHat = model.predict(samples)

        return yHat[0]


    def predictDataset(self, data):
        # data setting
        trainX, trainY, testX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        print('trainX :: ', trainX.shape)
        print('trainY :: ', trainY.shape)
        print('testX :: ', testX.shape)
        print('testY :: ', testY.shape)
        print('trainX.shape[0] :: ', trainX.shape[0])
        print('testX.shape[0] :: ', testX.shape[0])

        # normalize(정규화, 특정범위로 데이터 변환) input vectors
        inEncoder = Normalizer(norm='l2')
        print("before trainX[0] :: ", trainX[0])
        trainX = inEncoder.transform(trainX)
        print("after trainX[0] :: ", trainX[0])
        testX = inEncoder.transform(testX)

        # Label Encode (라벨의 문자를 숫자로 변환?)
        outEncoder = LabelEncoder()
        print("before trainY[0] :: ", trainY[0])
        outEncoder.fit(trainY)
        trainY = outEncoder.transform(trainY)
        print("before trainY[0] :: ", trainY[0])
        testY = outEncoder.transform(testY)

        # fit model (모델 적용)
        model = SVC(kernel='linear', probability=True)
        model.fit(trainX, trainY)


        # predict (테스트 데이터를 이용하여 예측 수행)
        yHatTrain = model.predict(trainX)
        yHatTest = model.predict(testX)

        # score (예측 결과 수치확인)
        scoreTrain = accuracy_score(trainY, yHatTrain)
        scoreTest = accuracy_score(testY, yHatTest)
        print("================================ Accuracy")
        print("train :: {}".format(scoreTrain * 100))
        print("test :: {}".format(scoreTest * 100))

    def predictImg(self):
        """
        이미지 검중
        :return:
        """
        data = np.load('./00.Resource/dataset/twice-faces-dataset.npz')
        testX_faces = data['arr_2']
        # print("testX_faces.shape :: ", testX_faces.shape)
        # print("testX_faces[0] :: ", testX_faces[0])

        # load testFace Embedding
        data = np.load('./00.Resource/embedding/twice-faces-embeddings.npz')
        trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

        # normalize input vectors
        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)
        testX = in_encoder.transform(testX)

        # label encode targets
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        trainy = out_encoder.transform(trainy)
        testy = out_encoder.transform(testy)

        # fit model
        model = SVC(kernel='linear', probability=True)
        model.fit(trainX, trainy)

        # test model on a random example from the test dataset
        selection = choice([i for i in range(testX.shape[0])])
        random_face_pixels = testX_faces[selection]
        random_face_emb = testX[selection]
        random_face_class = testy[selection]
        random_face_name = out_encoder.inverse_transform([random_face_class])

        # prediction for the face
        samples = expand_dims(random_face_emb, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)

        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
        print('Predicted: ClaaName :: %s Accuracy :: (%.3f) %%' % (predict_names[0], class_probability))
        print('Expected: %s' % random_face_name[0])

        # plot for fun
        # plt.imshow(random_face_pixels)
        # title = 'Class NaME :: {} Accuracy :: {} % '.format(predict_names[0], class_probability)
        # plt.title(title)
        # plt.show()


if __name__ == "__main__":
    print("run facenet Keras")

    from time import sleep

    # facenet class init
    faceNet = facenetInKeras()

    while(True):
        sleep(3)
        faceNet.predictImg()


