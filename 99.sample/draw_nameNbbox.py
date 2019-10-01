import sys,os
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
        print(dist, min_dist)
        if min_dist > dist:
            min_dist = dist
            index = i
    
    return index, min_dist

def get_name(index):
    known_label = ["dahyun", "jeongyeon", "momo", "sana", "tzuyu"]

    return known_label[index]

if __name__ == '__main__':

    if 'win' in sys.platform :
        path = os.path.normpath(os.path.abspath('../')).replace('\\','/') + '/00.Resource/'
    else:
        path = os.path.abspath('../') + '/00.Resource/'

    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(path + "shape_predictor_68_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1(path + "dlib_face_recognition_resnet_model_v1.dat")

    img_paths = ["/home/song/Downloads/twice/crop_dahyun/dahyun214.jpeg",
    "/home/song/Downloads/twice/crop_jeongyeon/jeongyeon036.jpeg",
     "/home/song/Downloads/twice/crop_momo/momo025.jpeg",
     "/home/song/Downloads/twice/crop_sana/sana010.jpeg",
     "/home/song/Downloads/twice/crop_tzuyu/tzuyu004.jpeg"]


    known_embd = []
    for img_path in img_paths:
        img = dlib.load_rgb_image(img_path)
        known_embd.append(np.array(get_embeddings(img,detector,sp,facerec)))

    img = dlib.load_rgb_image("/home/song/Downloads/twice.jpg")
    dets = detector(img,1)

    for k,d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k,d.left(),d.top(),d.right(),d.bottom()))

        shape = sp(img,d)
        face_chip = dlib.get_face_chip(img,shape)
        face_descriptor_from_prealigned_image = np.array(facerec.compute_face_descriptor(face_chip))

        ind, dist = compare_embedding(known_embd, face_descriptor_from_prealigned_image)
        if dist < 0.5:
            text = get_name(ind)
        else:
            text = "Unkown"
        
        # bbox, text 추가
        img = cv2.rectangle(img,(d.left(),d.top()), (d.right(),d.bottom()), (0, 255, 0), 1)
        img = cv2.putText(img, text, (d.left(),d.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)


    cv2.imshow("draw", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(-1)
    get_embeddings(img,detector,sp,facerec)




        