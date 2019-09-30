from PySide2 import QtGui, QtWidgets, QtCore
import cv2
from skimage import io
import dlib

class faceRecognition():

    def __init__(self):
        print("initialized faceRecognotion class")

    def rgbToGray(self, image):
        """
        # rgb 이미지를 gray scale 로 변환한다.
        :param image:
        :return:
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    def rgbResize(self, image, sizeX=None, sizeY=None):
        """
        # 이미지 사이즈를 변환하여 리턴한다.
        :param image:   rgb image data
        :param sizeX:   resize x (int)
        :param sizeY:   resize y (int)
        :return: resize image data
        """
        imgHeight, imgWidth, imgChanel = ""
        if image is None:
            print("##### 리사이즈 처리할 이미지가 없습니다.")
            return 0
        else:
            imgHeight, imgWidth, imgChanel = image.shape

        if sizeX is None and sizeY is None:
            sizeX = 300
            sizeY = 300

        print("sizeX : {}, sizeY : {}".format(sizeX, sizeY))
        print("imgHeight : {}, imgWidth : {}, imgChanel : {}".format(imgHeight, imgWidth, imgChanel))

        # image 의 사이즈 측정
        # image 의 사이즈가 Default size 300x300 보다 크면 300x300 으로 축소
        # image 의 사이즈가 Default size 300x300 보다 작으면 300x300 으로 확대
        if imgHeight > 300 and imgWidth > 300:
            return cv2.resize(image, dsize=(sizeX, sizeY), interpolation=cv2.INTER_AREA)  # 축소하는경우
        else:
            return cv2.resize(image, dsize=(sizeX, sizeY), interpolation=cv2.INTER_LINEAR)  # 확대하는경우


    def findFaceInImg(self, imgFilePath=None, img=None):
        """
        # NOTE : crop 되지않은 이미지에서 face 데이터를 검출한다.
        # DATE : 19.09.20
        # parmas : imgFilePath(이미지파일경로), img(이미지데이터)
        # return :
        """
        # QImage to cv2 mat

        # 이미지 데이터 use check
        image = ""
        if img is None and imgFilePath is not None:
            image = io.imread(imgFilePath)
        elif img is not None and imgFilePath is None:
            image = img
        else:
            print("imgFilePath, img(data) 둘중 하나는 입력되어야 합니다.")
            return 0

        # image 를 grayscale로 변환 (속도?)
        image = self.rgbToGray(image)
        # image = self.rgbResize(image)

        # dlib class 를 이용하여 얼굴 검출
        faceDetector = dlib.get_frontal_face_detector()
        detectedFaces = faceDetector(image, 1)

        # 얼굴 landmarks 처리
        predictorModel = "00.Resource/shape_predictor_68_face_landmarks.dat"
        facePosePredictor = dlib.shape_predictor(predictorModel)

        # 얼굴 이미지 변형
        # faceAligner = openface.AlignDlib(predictorModel)

        # UI 가 없는 창(colab)에서 열려고 하면 오류가 발생한다
        # colab 에서는 matplotlib 사용
        # plt.imshow(image)
        # plt.show()
        # 이미지 확인을 위한 윈도우 창 (UI Interface 전용)
        win = dlib.image_window()
        win.set_image(image)

        # face found in the image
        for idx, faceRect in enumerate(detectedFaces):
            # 검출된 얼굴좌표값을 윈도우로 전송
            win.add_overlay(faceRect)

            # 인식된 좌표에서 랜드마크 추출
            poseLandmarks = facePosePredictor(image, faceRect)
            print("shape.part(33) :안면부 코위치 중앙 좌표: {}".format(poseLandmarks.part(33)))
            win.add_overlay(poseLandmarks)

            # 얼굴 이미지 변형 후 이미지 파일 생성 처리 (비교를 위함)
            # alignedFace = faceAligner.align(534, image, faceRect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            #
            # cv2.imwrite("alignedFace_{}.jpg".format(idx), alignedFace)
        dlib.hit_enter_to_continue()




