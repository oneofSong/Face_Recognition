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


class facenetInKeras():
    """
    # 케라스를 이용하여 페이스넷 구현

    """
    def __init__(self):
        self.faceModel = None
        self.faceEmdDataSet = np.load('./00.Resource/embedding/twice-faces-embeddings.npz')
        self.trainX, self.trainY = self.faceEmdDataSet['arr_0'], self.faceEmdDataSet['arr_1']
        self.inEncoder = Normalizer(norm='l2')
        self.outEncoder = LabelEncoder()
        self.outEncoder.fit(self.trainY)
        self.trainX = self.inEncoder.transform(self.trainX)
        self.trainY = self.outEncoder.transform(self.trainY)
        # self.svcModel = SVC(kernel='linear', probability=True)
        # self.svcModel = SVC(kernel='rbf', gamma=2, probability=True)
        self.svcModel = SVC(kernel='rbf', probability=True)
        # self.svcModel = SVC(kernel='sigmoid', probability=True)
        self.svcModel.fit(self.trainX, self.trainY)


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

    def getEmbedding(self, facePixels):
        """
        # 입력된 얼굴데이터를 모델에서 임베딩한 결과 반환
        """
        # float32 로 형변환
        facePixels = facePixels.astype('float32')
        mean, std = facePixels.mean(), facePixels.std()
        facePixels = (facePixels - mean) / std
        samples = expand_dims(facePixels, axis=0)
        yHat = self.faceModel.predict(samples)

        return yHat[0]


    def predictDataset(self, data):
        # data setting
        trainX, trainY, testX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

        # normalize(정규화, 특정범위로 데이터 변환) input vectors
        inEncoder = Normalizer(norm='l2')
        trainX = inEncoder.transform(trainX)
        testX = inEncoder.transform(testX)

        # Label Encode (라벨의 문자를 숫자로 변환?)
        outEncoder = LabelEncoder()
        outEncoder.fit(trainY)
        trainY = outEncoder.transform(trainY)
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
        # print("================================ Accuracy")
        # print("train :: {}".format(scoreTrain * 100))
        # print("test :: {}".format(scoreTest * 100))

    def predictDatasetImg(self):
        """
        이미지 검중
        :return:
        """
        data = np.load('./00.Resource/dataset/twice-faces-dataset.npz')
        testX_faces = data['arr_2']

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

        # fit model (train 데이터를 맞춘다)
        model = SVC(kernel='linear', probability=True)
        model.fit(trainX, trainy)

        # test model on a random example from the test dataset
        selection = choice([i for i in range(testX.shape[0])])
        random_face_pixels = testX_faces[selection]     # 정답 이미지의 어레이
        random_face_emb = testX[selection]              # 이미지의 임베딩데이터
        random_face_class = testy[selection]
        random_face_name = out_encoder.inverse_traensform([random_face_class])

        # 출력 확인
        # print("random_face_pixels :: ")
        # print(random_face_pixels)
        # print("random_face_emb :: ")
        # print(random_face_emb)
        # print("random_face_class :: ")
        # print(random_face_class)
        # print("random_face_name :: ")
        # print(random_face_name)

        # prediction for the face
        samples = expand_dims(random_face_emb, axis=0)

        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)

        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
        # print('[sample 1] Predicted: ClaaName :: %s Accuracy :: (%.3f) %%' % (predict_names[0], class_probability))
        # print('[sample 1] Expected: %s' % random_face_name[0])

        # plot for fun
        # plt.imshow(random_face_pixels)
        # title = 'Class NaME :: {} Accuracy :: {} % '.format(predict_names[0], class_probability)
        # plt.title(title)
        # plt.show()

    def predictImg(self, emdData):
        """
        임베딩 데이터를 검증한다.
        :param emdData:
        :return:
        """

        targetEmdData = expand_dims(emdData, axis=0)
        yHatClass = self.svcModel.predict(targetEmdData)
        yHatProb = self.svcModel.predict_proba(targetEmdData)

        # print("yHatClass.length :: ", len(yHatClass))
        targetClassIdx = yHatClass[0]
        targetClassProba = yHatProb[0, targetClassIdx] * 100
        predictName = self.outEncoder.inverse_transform(yHatClass)

        print("predictName :: ", predictName)
        print("Accurency :: ", targetClassProba)
        return predictName, targetClassProba


    def extract_face(self, image, required_size=(160, 160)):
        """
        얼굴 데이터를 검출하고 얼굴영역을 160*160 사이즈 이미지로 리턴한다
        :param filePath:
        :param required_size:
        :return:
        """
        # load image from file
        detector = MTCNN()

        results = detector.detect_faces(image)
        face_array = list()

        if results == []:
            return results, face_array

        for idx in results:
            x1, y1, width, height = idx['box']

            # bug fix
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            # extract the face
            face = image[y1:y2, x1:x2]

            # resize pixels to the model size
            face_array.append(cv2.resize(face, required_size))

        return results, face_array


# if __name__ == "__main__":
#     print("run facenet Keras")
#
#     from time import sleep
#
#     # facenet class init
#     faceNet = facenetInKeras()
#
#     imgFace = cv2.imread("/home/bit/Downloads/twice.jpg", cv2.IMREAD_COLOR)
#     faceDetResults, faceImgArr = faceNet.extract_face(imgFace)
#
#     # print("faceDetResults :: ")
#     # print(faceDetResults)
#     #
#     # print("faceImgArr :: ")
#     # print(faceImgArr)
#
#     # 박스 갯수만큼 루프
#     for boxData in faceDetResults:
#         x, y, w, h = boxData['box']
#         print("{} :: {} :: {} :: {} ".format(x, y, w, h))
#
#         # bug fix
#         x, y = abs(x), abs(y)
#         x2, y2 = x + w, y + h
#
#         # extract the face
#         # faceImg = imgFace[y:y2, x:x2]
#
#         cv2.rectangle(imgFace, (x, y), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(imgFace, "NAME", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#
#     faceImg = cv2.resize(imgFace, dsize=(1280, 720), interpolation=cv2.INTER_AREA)
#     cv2.imwrite("./twice_box.jpg", faceImg)

    # 프레임 내 얼굴갯수만큼 루프
    # for idx in range(len(faceImgArr)):
    #     print("============== Run Target")
    #     imgToEmd = faceNet.getEmbedding(faceNet.faceModel, faceImgArr[idx])
    #     predictNm, predictPer = faceNet.predictImg(imgToEmd)
    #     # sleep(3)
    #     # crop img save
    #     cv2.imwrite("./twice{}_{}_{}.jpg".format(idx, predictNm, predictPer), faceImgArr[idx])
    #     print("============== End Target")

    # while(True):
    #     faceNet.predictDatasetImg()     # 데이터셋 이미지로 테스트 펑션
    #     sleep(3)
    #     faceNet.predictImg(imgPixel)


