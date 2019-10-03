import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import model_from_json
import os
from inception_resnet_v1 import *

class facenetRealtime():

    def __init__(self):
        # self.face_cascade = cv2.CascadeClassifier('./00.Resource/dataset/haarcascade_frontalface_default.xml')
        # self.face_cascade = cv2.CascadeClassifier('/home/bit/anaconda3/envs/faceRecognition/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier('C:/Users/JK/Anaconda3/envs/faceRecognition/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

        # self.face_cascade = cv2.CascadeClassifier('/home/bit/anaconda3/envs/faceRecognition/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_extended.xml')
        print("face_cascade built :: C:/Users/JK/Anaconda3/envs/faceRecognition/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
        # ------------------------
        # with open("../00.Resource/model/facenet_model.json", 'r') as file_handle:
        #     self.model = model_from_json(file_handle.read())
        self.model = InceptionResNetV1()
        print("model built")

        # https://drive.google.com/file/d/1971Xk5RwedbudGgTIrGAL4F7Aifu7id1/view?usp=sharing
        self.model.load_weights('../00.Resource/weights/facenet_weights.h5')
        print("weights loaded")
        # ------------------------

        self.employee_pictures = "F:/sampleData/blackpink_crop"
        self.cap = cv2.VideoCapture("F:/sampleData/bp1.mp4")  # videoFile
        # ------------------------

        self.threshold = 21  # tuned threshold for l2 disabled euclidean distance
        print("threshold setting :: ".format(self.threshold))
        self.metric = "cosine"  # cosine, euclidean
        if self.metric == "cosine":
            threshold = 0.45
        else:
            threshold = 0.95
        # ------------------------

    def findCosineDistance(self, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def findEuclideanDistance(self, source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(160, 160))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # preprocess_input normalizes input in scale of [-1, +1]. You must apply same normalization in prediction.
        # Ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py (Line 45)
        img = preprocess_input(img)
        return img

    def representationFileSetting(self, employees):
        """
        사진이미지를 이용하여 모델 설정
        :return: dict
        """
        for subDir in os.listdir(self.employee_pictures):
            path = self.employee_pictures + '/' + subDir + '/'
            # path = self.employee_pictures + '/' + subDir
            if not os.path.isdir(path):
                continue

            # faceImgs = self.loadFaces(path)
            for file in os.listdir(path):
                if not os.path.isdir(file):
                    employee, extension = file.split(".")
                    img = self.preprocess_image(path + '%s.%s' % (employee, extension))
                    representation = self.model.predict(img)[0, :]
                    employees[employee] = representation
        print("employee representations retrieved successfully")
        return employees
    
    def runFacenet(self, img):
        """
        facenet 구동
        :param img:
        :return: 
        """
        faces = self.face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in faces:
            if w > 80:  # discard small detected faces
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw rectangle to main image

                detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
                detected_face = cv2.resize(detected_face, (160, 160))  # resize to 224x224

                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                # employee dictionary is using preprocess_image and it normalizes in scale of [-1, +1]
                img_pixels /= 127.5
                img_pixels -= 1

                captured_representation = self.model.predict(img_pixels)[0, :]

                distances = []

                for i in self.employees:
                    employee_name = i
                    source_representation = self.employees[i]

                    if self.metric == "cosine":
                        distance = self.findCosineDistance(captured_representation, source_representation)
                    elif self.metric == "euclidean":
                        distance = self.findEuclideanDistance(captured_representation, source_representation)

                    # print(employee_name,": ",distance)
                    distances.append(distance)

                label_name = 'unknown'
                index = 0
                for i in self.employees:
                    employee_name = i
                    if index == np.argmin(distances):
                        if distances[index] <= self.threshold:
                            # print("detected: ",employee_name)
                            # label_name = "%s (distance: %s)" % (employee_name, str(round(distance,2)))

                            if self.metric == "euclidean":
                                similarity = 100 + (90 - 100 * distance)
                            elif self.metric == "cosine":
                                similarity = 100 + (40 - 100 * distance)
                            # similarity = 100 + (20 - distance)
                            # print("similarity :: ", similarity)
                            if similarity > 99.99: similarity = 99.99
                            label_name = "%s (%s%s)" % (employee_name, str(round(similarity, 2)), '%')
                            break

                    index = index + 1

                cv2.putText(img, label_name, (int(x + w + 15), int(y - 64)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
                # connect face and text
                cv2.line(img, (x + w, y - 64), (x + w - 25, y - 64), (0, 255, 0), 2)
                cv2.line(img, (int(x + w / 2), y), (x + w - 25, y - 64), (0, 255, 0), 2)
        return img

    def runVideo(self):
        self.employees = dict()
        self.employees = self.representationFileSetting(self.employees)

        while (True):
            ret, img = self.cap.read()
            img = cv2.resize(self.runFacenet(img), dsize=(640,480), interpolation=cv2.INTER_AREA)
            cv2.imshow('img', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                break

        # kill open cv things
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("========== run facenetRealTime.py")

    facenet = facenetRealtime()
    facenet.runVideo()
