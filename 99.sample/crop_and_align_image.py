#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy

import sys
import os
import dlib
import glob
import numpy as np
import cv2

# embedding 거리를 리턴
def encoding_distance(known_encodings,encoding_check):
    if len(known_encodings) == 0:
        return np.empty(0)s

    return np.linalg.norm(known_encodings - encoding_check,axis=0)

# 이미지의 embedding을 리턴
def get_embeddings(img,detector,sp,facerec):
    embd_list = []
    dets = detector(img,1)

    for k,d in enumerate(dets):
        shape = sp(img,d)
        face_chip = dlib.get_face_chip(img,shape)
        face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(face_chip)

        embd_list.append(np.array(face_descriptor_from_prealigned_image))

    return embd_list


# os 확인
if 'win' in sys.platform :
    path = os.path.normpath(os.path.abspath('../')).replace('\\','/') + '/00.Resource/'
else:
    path = os.path.abspath('../') + '/00.Resource/'

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(path + "shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1(path + "dlib_face_recognition_resnet_model_v1.dat")

win = dlib.image_window()

embd_list = []
cnt = 0

for f in glob.glob(os.path.join('C:\\Users\\bit\\Downloads\\nam',"*.jpg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)

    win.clear_overlay()
    win.set_image(img)

    #     # Ask the detector to find the bounding boxes of each face. The 1 in the
    #     # second argument indicates that we should upsample the image 1 time. This
    #     # will make everything bigger and allow us to detect more faces.
    dets = detector(img,1)
    print("Number of faces detected: {}".format(len(dets)))

    #     # Now process each face we found.
    for k,d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k,d.left(),d.top(),d.right(),d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = sp(img,d)
        face = img[d.top():d.bottom(),d.left():d.right()]
        cv2.imshow("crop",cv2.cvtColor(face,cv2.COLOR_RGB2BGR))

        # Draw the face landmarks on the screen so we can see what face is currently being processed.
        win.clear_overlay()
        win.add_overlay(d)
        win.add_overlay(shape)

        # 원본 이미지와 shape로 embedding을 계산
        # face_descriptor = facerec.compute_face_descriptor(img, shape)

        print("Computing descriptor on aligned image ..")

        # Let's generate the aligned image using get_face_chip
        face_chip = dlib.get_face_chip(img,shape)
        cv2.imshow("aligned",cv2.cvtColor(face_chip,cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(-1)

        if key == ord('q'):
            break

        face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(face_chip)
        embd_list.append(face_descriptor_from_prealigned_image)

        if not k == 1:
            print("distance {}".format(encoding_distance(np.array(embd_list[k]),np.array(embd_list[k - 1]))))

        cv2.destroyAllWindows()

