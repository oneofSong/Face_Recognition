from PySide2 import QtGui, QtWidgets, QtCore
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *
from time import sleep
import cv2,time
import os
from keras.models import load_model
from keras import backend as bk

# 3가지 형식의 모델 구성
import FacenetInKeras
import facenetRealTime
import openfaceRealTime

class cv_video_player(QThread):
    changePixmap = Signal(QImage)
    # changeTime = Signal(int,int)
    changeExtFrame = Signal(list)

    def __init__(self, file_path, parent=None):
        QThread.__init__(self)
        # self.openVideo()
        self.play = True
        self.cap = None
        self.out = None
        print(file_path)
        self.cap = cv2.VideoCapture(file_path)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        self.faceNet = FacenetInKeras.facenetInKeras()
        # print("========== facenet model build")
        # self.faceNet2 = facenetRealTime.facenetRealtime()
        # self.faceNet2.defaultSetFacenet2()
        # print("========== facenet2 model build")
        # self.openface = openfaceRealTime.openfaceRealTime()
        # self.openface.defaultSetOpenface()
        # print("========== openface model build")

        ###################
        # 시간 emit 추가시 필요
        ###################
        if file_path:
            self.total_frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.cur_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.duration = self.total_frame / self.fps
            self.minutes = int(self.duration / 60)
            self.seconds = int(self.duration % 60)


    def run(self):
        # 비디오 저장 테스트
        wFcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        wFps = 20.0
        self.out = cv2.VideoWriter("./test.avi", wFcc, wFps, (640, 480))

        # 비디오 업로드 시 모델 업로드 처리
        self.faceNet.faceModel = load_model('./00.Resource/model/facenet_keras.h5')

        while True:
            convertToQtFormat = ""
            rgbImage = ""
            if self.play and self.cap.isOpened():
                ret, frame = self.cap.read()
                self.cur_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                if ret:
                    rgbImage = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    # rgbImage = cv2.cvtColor(frame, cv2.IMREAD_COLOR)

                    # if int(self.cur_frame) % 30 == 0: # 30프레임당 1회 검출
                    # keras_facenet #1
                    rgbImage = self.faceRecog_keras_facenet(rgbImage)

                    # keras_facenet #2
                    # rgbImage = self.faceRecog_keras_facenet2(rgbImage)

                    # openface run
                    # rgbImage = self.faceRecog_keras_openface(rgbImage)

                    # 영상 저장
                    self.out.write(frame)

                    convertToQtFormat = QImage(rgbImage.data,rgbImage.shape[1],rgbImage.shape[0],
                                               rgbImage.shape[1] * rgbImage.shape[2],QImage.Format_RGB888)
                    self.changePixmap.emit(convertToQtFormat.copy())

                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                    self.play = False

            if not self.cur_frame % round(self.fps):
                # 초에 한번씩 프레임데이터를 검출결과테이블로 전달(데모를 위함)
                if int(self.cur_frame / round(self.fps)) % 2 == 0:
                    continue

            time.sleep(0.025)
        self.cap.release()
        self.out.release()


    def faceRecog_keras_facenet(self, rgbImage):
        """
        # keras를 이용한 facenet 얼굴 검출 처리 #1
        :param rgbImage:
        :return: rgbImage
        """
        # print("======================검출을 수행합니다.(faceRecog_keras_facenet)")
        faceDetResults, faceImgArr = self.faceNet.extract_face(rgbImage)

        # 이미지 내 검출된 얼굴 갯수만큼 루프
        for idx in range(len(faceDetResults)):
            # 검출된 얼굴박스 하나에 대하여 임베딩 처리
            imgToEmd = self.faceNet.getEmbedding(faceImgArr[idx])
            # 임베딩 된 얼굴데이터의 검증 수행
            predictNm, predictPer = self.faceNet.predictImg(imgToEmd)

            # predict 결과치가 특정 퍼센트 이상일 때만 박스 생성
            if 65 < int(predictPer):
                x, y, w, h = faceDetResults[idx]['box']
                # print("left : {} _ top : {} _ right : {} _ bottom : {}".format(x, y, w, h))

                # bug fix
                x, y = abs(x), abs(y)
                x2, y2 = x + w, y + h

                # extract the face
                # faceImg = rgbImage[y:y2, x:x2]
                # rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.rectangle(rgbImage, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(rgbImage, str(predictNm), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                continue

        # 박스 처리된 이미지 저장
        # cv2.imwrite("./twice_{}frame.jpg".format(self.cur_frame), rgbImage)

        return rgbImage

    def faceRecog_keras_facenet2(self, rgbImage):
        """
        facenet 2 run
        :param rgbImage:
        :return:rgbImage
        """
        # print("======================검출을 수행합니다.(faceRecog_keras_facenet)")
        rgbImage = self.faceNet2.runPredictFacenet2(rgbImage)
        return rgbImage

    def faceRecog_keras_openface(self, rgbImage):
        """
        openface run
        :param rgbImage:
        :return:
        """
        # print("======================검출을 수행합니다.(faceRecog_keras_openface)")
        rgbImage = self.openface.runPredictOpenface(rgbImage)
        return rgbImage

    def pauseVideo(self):
        self.play = False

    def playVideo(self):
        if self.cap is None:
            return

        if not self.isRunning():
            self.start()

        self.play = True

    def stopVideo(self):
        pass


    def openVideo(self,file_path):
        print(file_path)
        self.cap = cv2.VideoCapture(file_path)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        if file_path:
            self.total_frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.cur_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.duration = self.total_frame / self.fps
            self.minutes = int(self.duration/60)
            self.seconds = int(self.duration%60)
            # self.changeTime.emit(int(self.cur_frame / self.fps),int(self.duration))

        # 창을 다시 열었을 때를 위해 upload시에 라벨을 특정 색으로 초기화
        # convertToQtFormat = QImage(rgbImage.data,w,h,bytesPerLine,QImage.Format_RGB888)
        # p = convertToQtFormat.scaled(1280,1040,Qt.KeepAspectRatio)
        # self.changePixmap(p)

    def moveFrame(self, frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
        ret, frame = self.cap.read()
        self.cur_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        # self.changeTime.emit(int(self.cur_frame / self.fps), int(self.duration))

        if ret:
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0],
                                       rgbImage.shape[1] * rgbImage.shape[2], QImage.Format_RGB888)
            self.changePixmap.emit(convertToQtFormat.copy())

    def initScreen(self):
        black_image = QImage(1920,1280, QImage.Format_Indexed8)
        black_image.fill(QtGui.qRgb(0,0,0))
        self.changePixmap.emit(black_image.copy())

    def play_Real(self):
        while True:
            if self.play and self.cap.isOpened():
                ret,frame = self.cap.read()
                self.cur_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                if ret:
                    rgbImage = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    convertToQtFormat = QImage(rgbImage.data,rgbImage.shape[1],rgbImage.shape[0],
                                               rgbImage.shape[1] * rgbImage.shape[2],QImage.Format_RGB888)
                    self.changePixmap.emit(convertToQtFormat.copy())
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                    self.play = False

            # if not self.cur_frame % round(self.fps):
            #    # print("cur frame : {} total frame : {} ".format(self.cur_frame, self.total_frame))
            #    # print("fps : {} {}".format(round(self.fps), self.cur_frame / round(self.fps)))
            #     self.changeTime.emit(int(self.cur_frame / self.fps),int(self.duration))

            time.sleep(0.025)

    def play_Demo(self):
        print("thread start")
        while True:

            if self.play and self.cap.isOpened():
                ret,frame = self.cap.read()
                self.cur_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                if ret:
                    rgbImage = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    convertToQtFormat = QImage(rgbImage.data,rgbImage.shape[1],rgbImage.shape[0],
                                               rgbImage.shape[1] * rgbImage.shape[2],QImage.Format_RGB888)
                    self.changePixmap.emit(convertToQtFormat.copy())
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                    self.play = False

            # if not self.cur_frame % round(self.fps):
            #     # print("cur frame : {} total frame : {} ".format(self.cur_frame, self.total_frame))
            #     # print("fps : {} {}".format(round(self.fps), self.cur_frame / round(self.fps)))
            #     self.changeTime.emit(int(self.cur_frame / self.fps),int(self.duration))

            time.sleep(0.025)