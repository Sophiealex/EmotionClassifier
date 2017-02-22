import os
import cv2
import dlib
import numpy
import shutil
import random



def build_dataset(image_dir, label_dir, output_dir, itype='png', advanced_augmentation=True):
    """ Builds a dataset using data augmentation and normalization built for the CK+ Emotion Set.
    :param image_dir: A directory of input images.
    :type image_dir: str
    :param label_dir: A directory of input labels.
    :type label_dir: str
    :param output_dir: A directory for new images to be sorted.
    :type output_dir: str
    :param itype: File type for output images.
    :type itype: str
    :param advanced_augmentation: If advanced augmentation is enabled.
    :type: advanced_augmentation: bool
    :return: The number of images.
    :rtype: int
    """
    image_files = []
    for outer_folder in os.listdir(image_dir):
        if os.path.isdir(image_dir + '/' + outer_folder):
            for inner_folder in os.listdir(image_dir + '/' + outer_folder):
                if os.path.isdir(image_dir + '/' + outer_folder + '/' + inner_folder):
                    for input_file in os.listdir(image_dir + '/' + outer_folder + '/' + inner_folder):
                        if input_file.split('.')[1] != itype:
                            break
                        label_file = label_dir+'/'+outer_folder+'/'+inner_folder+'/'+input_file[:-4] + '_emotion.txt'
                        if os.path.isfile(label_file):
                            read_file = open(label_file, 'r')
                            label = int(float(read_file.readline()))
                            image_files.append((image_dir+'/'+outer_folder+'/'+inner_folder+'/'+input_file, label))
                            neutral_file = sorted(os.listdir(image_dir+'/'+outer_folder+'/'+inner_folder))[0]
                            if neutral_file.split('.')[1] != itype:
                                neutral_file = sorted(os.listdir(image_dir+'/'+outer_folder+'/'+inner_folder))[1]
                            image_files.append((image_dir+'/'+outer_folder+'/'+inner_folder+'/'+neutral_file, 0))

    print '-------------Files Collected----------------------------------------------------------------------'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        for i in range(8):
            os.makedirs(output_dir+'/'+str(i))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    detector, count = dlib.get_frontal_face_detector(), 0
    for image_file in image_files:
        image = clahe.apply(cv2.imread(image_file[0], cv2.IMREAD_GRAYSCALE))
        detections = detector(image, 1)
        for _, detection in enumerate(detections):
            left, right, top, bottom = detection.left()-20,detection.right()+20,detection.top()-20,detection.bottom()+20
            face = image[top:bottom, left:right]
            face = cv2.resize(face, (96, 96))
            patches = [face[0:88, 0:88], face[8:96, 0:88], face[0:88, 8:96], face[8:96, 8:96], face[4:92, 4:92]]
            for patch in patches:
                name = output_dir + '/' + str(image_file[1]) + '/' + image_file[0][-21:-4] + str(count) + '.' + itype
                cv2.imwrite(name, patch)
                name = output_dir + '/' + str(image_file[1]) + '/' + image_file[0][-21:-4] + str(count + 1) + '.' + itype
                cv2.imwrite(name, cv2.flip(patch, 1))
                count += 2
                if advanced_augmentation:
                    params = [[0, 44, 0, 88], [44, 88, 0 , 88], [0, 88, 0, 44], [0, 88, 44, 88]]
                    for param in params:
                        section = numpy.zeros((88, 88), dtype=numpy.uint8)
                        for i in range(param[0], param[1]):
                            for j in range(param[2], param[3]):
                                section[i, j] = patch[i, j]
                        count+=1
                        name = output_dir+'/'+str(image_file[1])+'/'+image_file[0][-21:-4]+str(count)+'.'+itype
                        cv2.imwrite(name, section)

        if count % 100 == 0:
            print 'Current count = ' + str(count)

    return count


def normalize(output_dir):
    """ Normalize the data in the directory.
    :param output_dir: The directory to normalize.
    :type output_dir: str
    :return: The number of files in each folder.
    :rtype: int
    """
    minimum, num_files = 1000000, []
    for folder in os.listdir(output_dir):
        path = output_dir + '/' + folder
        files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        num_files.append([files, path])
        print path + ' has ' + str(files) + ' files'
        if files < minimum:
            minimum = files

    print 'minimum = ' + str(minimum)

    for i in range(len(num_files)):
        while num_files[i][0] > minimum:
            os.remove(num_files[i][1] + '/' + random.choice(os.listdir(num_files[i][1])))
            num_files[i][0] -= 1

    return minimum * len(num_files)


def split_data(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir + '/1')
        shutil.copytree(input_dir + '/0', output_dir + '/0')
        for i in [1, 3, 4, 6]:
            for image_file in os.listdir(input_dir + '/' + str(i)):
                shutil.copy(input_dir+'/'+str(i)+'/'+image_file, output_dir+'/'+str(1)+'/'+image_file)
        os.makedirs(output_dir + '/2')
        for i in [2, 5, 7]:
            for image_file in os.listdir(input_dir + '/' + str(i)):
                shutil.copy(input_dir+'/'+str(i)+'/'+image_file, output_dir+'/'+str(2)+'/'+image_file)


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


def get_face(image):
    """ Gets a normalised image from a image.
    :param image: A image to be processed.
    :type image: OpenCV image
    :return: A normalised image of the face from the inputted image.
    :rtype: image
    """
    try:
        detector = dlib.get_frontal_face_detector()
        dets, data = detector(image, 1), []
        for _, d in enumerate(dets):
            left, right, top, bottom = d.left() - 20, d.right() + 20, d.top() - 20, d.bottom() + 20
            face = image[top:bottom, left:right]
            data.append(cv2.resize(face, (88, 88)))
        return data
    except Exception:
        return 0


def get_face_from_file(image_file):
    """ Gets a normalised image from a individual file path.
    :param image_file: A image path to be processed.
    :type image_file: str
    :return: A normalised image of the face from the inputted image.
    :rtype: image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(cv2.imread(image_file, cv2.IMREAD_GRAYSCALE))
    return get_face(image)


def get_face_from_frame(image_frame):
    """ Gets a normalised image from a image.
    :param image_frame: The frame to extract the face.
    :type image_frame: OpenCV image
    :return: A normalised image of the face from the inputted image.
    :rtype: image
    """
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY))
        return get_face(image)
    except Exception:
        return 0
