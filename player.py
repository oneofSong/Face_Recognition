from PySide2 import QtGui, QtWidgets, QtCore
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *
from time import sleep
import cv2,time
import os
from keras.models import load_model
from keras import backend as bk
import FacenetInKeras

class cv_video_player(QThread):
    changePixmap = Signal(QImage)
    # changeTime = Signal(int,int)
    changeExtFrame = Signal(list)

    def __init__(self, file_path, parent=None):
        QThread.__init__(self)
        # self.openVideo()
        self.play = True
        self.cap = None
        print(file_path)
        self.cap = cv2.VideoCapture(file_path)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        self.faceNet = FacenetInKeras.facenetInKeras()


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
        # faceModel = load_model('./00.Resource/model/facenet_keras.h5')

        while True:
            convertToQtFormat = ""
            rgbImage = ""
            if self.play and self.cap.isOpened():
                ret, frame = self.cap.read()
                self.cur_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                if ret:
                    rgbImage = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    # rgbImage = cv2.cvtColor(frame, cv2.IMREAD_COLOR)

                    # 30프레임당 1회 검출
                    if int(self.cur_frame) % 30 == 0:
                        print("======================검출을 수행합니다.")
                        faceDetResults, faceImgArr = self.faceNet.extract_face(rgbImage)
                        for idx in range(len(faceDetResults)):
                            imgToEmd = self.faceNet.getEmbedding(faceImgArr[idx])
                            # imgToEmd = self.faceNet.getEmbedding(faceImgArr[idx])
                            # bk.clear_session()
                            predictNm, predictPer = self.faceNet.predictImg(imgToEmd)
                            # bk.clear_session()

                            # predict 결과치가 80% 이상일 때만 박스 생성
                            if 95 < int(predictPer):
                                x, y, w, h = faceDetResults[idx]['box']
                                print("left : {} _ top : {} _ right : {} _ bottom : {}".format(x, y, w, h))

                                # bug fix
                                x, y = abs(x), abs(y)
                                x2, y2 = x + w, y + h

                                # extract the face
                                # faceImg = rgbImage[y:y2, x:x2]
                                # rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                rgbImage = cv2.rectangle(rgbImage, (x, y), (x2, y2), (0, 255, 0), 1)
                                cv2.imwrite("./twice_{}frame_{}.jpg".format(self.cur_frame, idx), rgbImage)
                                rgbImage = cv2.putText(rgbImage, predictNm, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0), 2)
                            else:
                                continue

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