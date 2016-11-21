import os
import cv2
import dlib


def build_data():
    count = 0
    for folder in os.listdir('Data/Images'):
        for inner_folder in os.listdir('Data/Images/' + folder):
            if os.path.isdir('Data/Images/' + folder + '/' + inner_folder):
                for input_file in os.listdir('Data/Images/' + folder + '/' + inner_folder):
                    label_file = 'Data/Labels/' + folder + '/' + inner_folder + '/' + input_file[:-4] + '_emotion.txt'
                    if os.path.isfile(label_file):
                        f = open(label_file, 'r')
                        label = int(float(f.readline()))
                        image_file = 'Data/Images/' + folder + '/' + inner_folder + '/' + input_file
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        image = clahe.apply(cv2.imread(image_file, cv2.IMREAD_GRAYSCALE))
                        detector = dlib.get_frontal_face_detector()
                        detections = detector(image, 1)
                        for _, d in enumerate(detections):
                            left, right, top, bottom = d.left() - 20, d.right() + 20, d.top() - 20, d.bottom() + 20
                            face = image[top:bottom, left:right]
                            face = cv2.resize(face, (96, 96))
                            patches = [face[0:88, 0:88], face[8:96, 0:88], face[0:88, 8:96],
                                       face[8:96, 8:96], face[4:92, 4:92]]
                            for i in range(len(patches)):
                                cv2.imwrite('Images/'+str(label)+'/'+input_file[:-4]+str(i)+'.png',patches[i])
                                count += 1
                                print label
    print count

build_data()