import os
import cv2
import dlib
import numpy


def build_data(image_dir, label_dir, output_dir, itype='png'):
    """ Builds a dataset using data augmentation and normalization built for the CK+ Emotion Set.
    :param image_dir: A directory of input images.
    :type image_dir: str
    :param label_dir: A directory of input labels.
    :type label_dir: str
    :param output_dir: A directory for new images to be sorted.
    :type output_dir: str
    :param itype: File type for output images.
    :type itype: str
    :return: void
    """
    count = 0
    for folder in os.listdir(image_dir):
        if os.path.isdir(image_dir + folder):
            for inner_folder in os.listdir(image_dir + folder):
                if os.path.isdir(image_dir + folder + '/' + inner_folder):
                    for input_file in os.listdir(image_dir + folder + '/' + inner_folder):
                        label_file = label_dir + folder + '/' + inner_folder + '/' + input_file[:-4] + '_emotion.txt'
                        if os.path.isfile(label_file):
                            f = open(label_file, 'r')
                            label = int(float(f.readline()))
                            for image_file in os.listdir(image_dir + folder + '/' + inner_folder):
                                print image_file + "\tCurrent count: " + str(count)
                                if image_file.split('.')[1] != itype:
                                    break
                                image_file = image_dir + folder + '/' + inner_folder + '/' + image_file
                                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                                image = clahe.apply(cv2.imread(image_file, cv2.IMREAD_GRAYSCALE))
                                detector = dlib.get_frontal_face_detector()
                                detections = detector(image, 1)
                                for _, d in enumerate(detections):
                                    left, right, top, bottom = d.left()-20, d.right()+20, d.top()-20, d.bottom()+20
                                    face = image[top:bottom, left:right]
                                    face = cv2.resize(face, (96, 96))
                                    patches = [face[0:88, 0:88], face[8:96, 0:88], face[0:88, 8:96],
                                               face[8:96, 8:96], face[4:92, 4:92]]
                                    for i in range(len(patches)):
                                        name = output_dir + str(label) + '/' + image_file[-21:-4] + str(count) + '.' + itype
                                        print name
                                        cv2.imwrite(name, patches[i])
                                        count += 1
    return count


def get_data(input_dir, num_classes):
    """ Gets the data from a directory of images organised by classification.
    :param input_dir: The directory of images.
    :type input_dir: str
    :return: A list of images and labels.
    :rtype: list of tuples each containing an image and a int
    """
    data, label = [], 0
    for folder in sorted(os.listdir(input_dir)):
        for image_file in os.listdir(input_dir + folder):
            labels = numpy.zeros(num_classes)
            labels[int(label)] = 1
            data.append((cv2.imread(input_dir + folder + '/' + image_file, cv2.IMREAD_GRAYSCALE), labels))
        label += 1
    return data


def get_face(image_file):
    """ Gets a normalised image from a individual file path.
    :param image_file: A image path to be processed.
    :type image_file: str
    :return: A normalised image of the face from the inputted image.
    :rtype: image
    """
    clahe = cv2.createCLAHE(2.0, (8, 8))
    image = clahe.apply(cv2.imread(image_file, cv2.IMREAD_GRAYSCALE))
    detector = dlib.get_frontal_face_detector()
    dets, data = detector(image, 1), []
    for _, d in enumerate(dets):
        left, right, top, bottom = d.left() - 20, d.right() + 20, d.top() - 20, d.bottom() + 20
        face = image[top:bottom, left:right]
        data.append(cv2.resize(face, (88, 88)))
    return data
