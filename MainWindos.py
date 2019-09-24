# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'player.ui',
# licensing of 'player.ui' applies.
#
# Created: Tue Sep 24 13:43:34 2019
#      by: pyside2-uic  running on PySide2 5.9.0~a1
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets
from player import cv_video_player
import faceRecognition

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1075, 629)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.video_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.video_label.sizePolicy().hasHeightForWidth())
        self.video_label.setSizePolicy(sizePolicy)
        self.video_label.setTextFormat(QtCore.Qt.RichText)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setWordWrap(False)
        self.video_label.setObjectName("label")
        self.verticalLayout.addWidget(self.video_label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.paly_btn = QtWidgets.QPushButton(self.centralwidget)
        self.paly_btn.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.paly_btn)
        self.stop_btn = QtWidgets.QPushButton(self.centralwidget)
        self.stop_btn.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.stop_btn)
        self.verticalLayout.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1075, 20))
        self.menubar.setObjectName("menubar")
        self.menuFace_Recognition = QtWidgets.QMenu(self.menubar)
        self.menuFace_Recognition.setObjectName("menuFace_Recognition")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuFace_Recognition.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        ############################
        # player 사용시 업로드할 file_path를 입력
        ########################
        self.video_player = cv_video_player(file_path="C:/Users/bit/Downloads/test.mp4")

        ## faceRecognotion class
        self.fr = faceRecognition()

        ###########
        # button
        ###########

        self.paly_btn.clicked.connect(self.play_clicked)
        self.stop_btn.clicked.connect(self.stop_clicked)


    def play_clicked(self):
        if self.video_player.isRunning():
            self.video_player.playVideo()
        else:
            self.video_player.changePixmap.connect(self.setPixMap)
            self.video_player.playVideo()

        # play 이후 추출 connect 실행(데모용, 검출 시작 버튼 클릭시 처리하는걸로 변경필요)
        self.cm.video_player.changeExtFrame.connect(self.insertAtResultListData)

    def stop_clicked(self):
        self.video_player.pauseVideo()

    @QtCore.Slot(list)
    def insertAtResultListData(self, dataList):
        """
        dlib 를 이용한 얼굴 검출
        :param dataList:
        :return:
        """
        self.fr.findFaceInImg(imgFilePath=None, img=dataList[0])

    @QtCore.Slot(QtGui.QImage)
    def setPixMap(self,image):
        image = QtGui.QPixmap.fromImage(image)
        image = image.scaled(self.video_label.size(),QtCore.Qt.KeepAspectRatio)
        self.video_label.setPixmap(image)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtWidgets.QApplication.translate("MainWindow", "Face Recognition", None, -1))
        self.video_label.setText(QtWidgets.QApplication.translate("MainWindow","Video",None,-1))
        self.paly_btn.setText(QtWidgets.QApplication.translate("MainWindow","play",None,-1))
        self.stop_btn.setText(QtWidgets.QApplication.translate("MainWindow","stop",None,-1))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

