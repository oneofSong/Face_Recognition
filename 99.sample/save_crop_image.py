from mtcnn.mtcnn import MTCNN
import cv2
import argparse
import os, sys

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--read_path', type=str, default='', help='path to image file or directory to images')
    parser.add_argument('--name', type=str, default='', help='path to image file or directory to images')

    return parser.parse_args()

def extract_face(filename, required_size=(160, 160)):
    # load image from file
	print(filename)
	image = cv2.imread(filename)
	detector = MTCNN()

	results = detector.detect_faces(image)
    
	if results == []:
		return None, None
	x1, y1, width, height = results[0]['box']
    
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
    
	# extract the face
	face = image[y1:y2, x1:x2]

	# resize pixels to the model size
	face_array = cv2.resize(face, required_size)
    
	return results, face_array

def load_faces(directory):
    faces = list()
    results = list()

    # enumerate files
    for filename in os.listdir(directory):
        # path
        path = directory + filename

        if os.path.isdir(path):
            continue
        
        # get face
        result, face = extract_face(path)

        if result is None:
            continue
                
        # store
        results.append(result)
        faces.append([filename,face])
        
    return results, faces

def save_face_image(name, save_path, faces):
    for i, face in enumerate(faces):
        path = save_path+ face[0]
        print(path)
        cv2.imwrite(path, face[1])


if __name__ == '__main__':
    args = get_args()

    read_path = args.read_path
    name = args.name

    if not os.path.exists(read_path):
        print("존재하지 않는 경로입니다.")
        sys.exit(-1)
    else:
        if read_path[-1] == '/':
            save_path = read_path+"crop/"
        else:
            read_path += '/'
            save_path = read_path+"crop/"
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)

_, faces = load_faces(read_path)
save_face_image(name=name, save_path=save_path, faces = faces)

