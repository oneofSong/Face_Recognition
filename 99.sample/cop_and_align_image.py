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


def encoding_distance(known_encodings,encoding_check):
    if len(known_encodings) == 0:
        return np.empty(0)

    return np.linalg.norm(known_encodings - encoding_check,axis=0)


def get_embeddings(img,detector,sp,facerec):
    embd_list = []
    dets = detector(img,1)

    for k,d in enumerate(dets):
        shape = sp(img,d)
        face_chip = dlib.get_face_chip(img,shape)
        face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(face_chip)

        embd_list.append(np.array(face_descriptor_from_prealigned_image))

    return embd_list


# if len(sys.argv) != 4:
#     print(
#         "Call this program like this:\n"
#         "   ./face_recognition.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat ../examples/faces\n"
#         "You can download a trained facial shape predictor and recognition model from:\n"
#         "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
#         "    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
#     exit()

# predictor_path = sys.argv[1]
# face_rec_model_path = sys.argv[2]
# faces_folder_path = sys.argv[3]

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.

# os 확인
if 'win' in sys.platform :
    path = os.path.normpath(os.path.abspath('../')).replace('\\','/') + '/00.Resource/'
else:
    path = os.path.abspath('../') + '/00.Resource/'

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(path + "shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1(path + "dlib_face_recognition_resnet_model_v1.dat")

win = dlib.image_window()

# Now process all the images
# for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
# img_path = "/home/song/bitproject/data/전지현/3684185.jpg"
# img_path_2 = "/home/song/bitproject/data/전지현/3684247.jpg"
# img_path_3 = "/home/song/bitproject/data/옹홍/3586678.jpg"


# img = dlib.load_rgb_image(img_path)
# img_2 = dlib.load_rgb_image(img_path_2)
# embe_list = get_embeddings(img,detector,sp,facerec)
# embe_list_2 = get_embeddings(img_2,detector,sp,facerec)
# # print("{} {}".format(type(embe_list[0]),embe_list[0]))
#
# print(encoding_distance(embe_list[0],embe_list_2[0]))
#
# img_2 = dlib.load_rgb_image(img_path_3)
# embe_list_2 = get_embeddings(img_2,detector,sp,facerec)
#
# print(encoding_distance(embe_list[0],embe_list_2[0]))

# for k, d in enumerate(dets):
#     shape = sp(img, d)
#     face_chip = dlib.get_face_chip(img, shape)
#     face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(face_chip)
#     embd_list.append(face_descriptor_from_prealigned_image)

embd_list = []
cnt = 0
img_path = "C:/Users/bit/Downloads/남상미"
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
        # dlib.hit_enter_to_continue()

