from PySide2 import QtGui, QtWidgets, QtCore
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *
from time import sleep
import cv2,time
import os

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

        ###################
        # 시간 emit 추가시 필요
        ###################
        # if file_path:
        #     self.total_frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        #     self.cur_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        #     self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        #     self.duration = self.total_frame / self.fps
        #     self.minutes = int(self.duration / 60)
        #     self.seconds = int(self.duration % 60)

    def run(self):
        while True:
            convertToQtFormat = ""
            rgbImage = ""
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

            if not self.cur_frame % round(self.fps):
                print("cur frame : {} total frame : {} ".format(self.cur_frame,self.total_frame))
                print("fps : {} {}".format(round(self.fps),self.cur_frame / round(self.fps)))
            #     self.changeTime.emit(int(self.cur_frame / self.fps),int(self.duration))
            #
                # 3초에 한번씩 프레임데이터를 검출결과테이블로 전달(데모를 위함)
                if int(self.cur_frame / round(self.fps)) % 3 == 0:
                    # 이미지 프레임 rgb list (추가 데이터 존재를 위해 이중list 처리
                    checkFrameData = [rgbImage, str(self.cur_frame)]
                    self.changeExtFrame.emit(checkFrameData)
            #         print("프레임 emit 실행")
            #         # 검출을 위해 이미지를 검출procClass 로 보내고 리턴받는 작업 필요
            #         resultData = ["1", "2", "3", str(self.cur_frame)]
            #         self.changeExtFrame.emit(rgbImage, resultData)
            #         # self.changeExtFrame.emit(convertToQtFormat.copy(),resultData)

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