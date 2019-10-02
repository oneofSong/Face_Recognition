import sys, os, glob
import numpy as np
import dlib
import cv2

def encoding_distance(known_encodings,encoding_check):
    if len(known_encodings) == 0:
        return np.empty(0)

    return np.linalg.norm(known_encodings - encoding_check,axis=0)


def get_embeddings(img,detector,sp,facerec):
    embd_list = []
    dets = detector(img,1)

    if len(dets) == 1:
        shape = sp(img,dets[0])
        face_chip = dlib.get_face_chip(img,shape)
        face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(face_chip)

        return np.array(face_descriptor_from_prealigned_image)
    else:
        for k,d in enumerate(dets):
            shape = sp(img,d)
            face_chip = dlib.get_face_chip(img,shape)
            face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(face_chip)

            embd_list.append(np.array(face_descriptor_from_prealigned_image))
        return embd_list

def compare_embedding(known_embedding, cur_embedding):
    min_dist = 1
    index = 0
    for i, embd in enumerate(known_embedding):
        dist = encoding_distance(embd,cur_embedding)
        if min_dist > dist:
            min_dist = dist
            index = i
    
    print(min_dist)
    return index, min_dist

def get_name(index):
    known_label = ["dahyun", "jeongyeon", "momo", "sana", "tzuyu"]

    return known_label[index]


# os 확인
if 'win' in sys.platform :
    path = os.path.normpath(os.path.abspath('../')).replace('\\','/') + '/00.Resource/'
else:
    path = os.path.abspath('../') + '/00.Resource/'

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(path + "shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1(path + "dlib_face_recognition_resnet_model_v1.dat")


img_paths = ["/home/song/Downloads/twice/crop_dahyun/dahyun214.jpeg",
    "/home/song/Downloads/twice/crop_jeongyeon/jeongyeon225.jpeg",
     "/home/song/Downloads/twice/crop_momo/momo267.jpeg",
     "/home/song/Downloads/twice/crop_sana/sana010.jpeg",
     "/home/song/Downloads/twice/crop_tzuyu/tzuyu004.jpeg"]


known_embd = []
for img_path in img_paths:
    img = dlib.load_rgb_image(img_path)
    known_embd.append(np.array(get_embeddings(img,detector,sp,facerec)))


cap = cv2.VideoCapture("/home/bit/Downloads/test2.mp4")
cnt = 0

cap.set(cv2.CAP_PROP_POS_FRAMES,1200)

while cap.isOpened():

    ret, frame = cap.read()
    cv2.imshow("play", frame)

    #     # Ask the detector to find the bounding boxes of each face. The 1 in the
    #     # second argument indicates that we should upsample the image 1 time. This
    #     # will make everything bigger and allow us to detect more faces.
    if cnt % 1 == 0:
        dets = detector(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),1)
        print("Number of faces detected: {}".format(len(dets)))

    #     # Now process each face we found.
        for k,d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k,d.left(),d.top(),d.right(),d.bottom()))

            shape = sp(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),d)
            face_chip = dlib.get_face_chip(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),shape)
            face_descriptor_from_prealigned_image = np.array(facerec.compute_face_descriptor(face_chip))

            ind, dist = compare_embedding(known_embd, face_descriptor_from_prealigned_image)
            if dist < 0.53:
                text = get_name(ind)
                frame = cv2.rectangle(frame,(d.left(),d.top()), (d.right(),d.bottom()), (0, 255, 0), 1)
                frame = cv2.putText(frame, text, (d.left(),d.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                text = "Unkown"
            

        frame = cv2.resize(frame, dsize=(1280, 720), interpolation=cv2.INTER_AREA)
        cv2.imshow("play", frame)
        cv2.waitKey(1)
    
    cnt += 1    

cap.release()
cv2.destroyAllWindows()

