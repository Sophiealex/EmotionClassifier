import sys
import time
import builddata
import emotionclassifier


def main():
    if len(sys.argv) >= 2:
        mode = sys.argv[1]
        if mode == 'train':
            if len(sys.argv) > 5:
                start = time.clock()
                faces = builddata.get_data(sys.argv[2], int(sys.argv[5]))
                training_data, testing_data = emotionclassifier.divide_data(faces, 0.2)
                print 'number of training examples = ' + str(len(training_data))
                print 'number of testing examples  = ' + str(len(testing_data)) + '\n'

                classifier = emotionclassifier.EmotionClassifier(int(sys.argv[5]), sys.argv[3])
                accuracy = classifier.train(training_data, testing_data, int(sys.argv[4]), intervals=1)
                end = time.clock()
                print 'Testing Accuracy: ' + '{:.9f}'.format(accuracy)
                print 'Training Time: ' + '{:.2f}'.format(end - start) + 's'
            else:
                print 'Please add \'Image Dir\' \'Session Save Path\' \'Number of Epochs\' \'Number of Classes\''

        elif mode == 'classify':
            if len(sys.argv) > 3:
                start = time.clock()
                face = builddata.get_face(sys.argv[2])
                classifier = emotionclassifier.EmotionClassifier(8, sys.argv[3])
                classification = classifier.classify(face)
                end = time.clock()
                print sys.argv[2] + ' classified as type ' + str(classification[1]) + ' in ' + str(end - start) + 's'
            else:
                print 'Please add \'Image Path\' \'Session Save Path\''

        elif mode == 'build':
            if len(sys.argv) > 4:
                start = time.clock()
                count = builddata.build_data(sys.argv[2], sys.argv[3], sys.argv[4])
                end = time.clock()
                print 'Augmented Dataset built ' + str(count) + ' images in ' + str(end - start) + 's'
            else:
                'Please add \'Image Dir\' \'Label Dir\' \'Output Dir\''

        else:
            print 'Please add either \'train\', \'classify\' or \'build\' as command line arguments.'

    else:
        print 'Please add either \'train\', \'classify\' or \'build\' as command line arguments.'


if __name__ == '__main__':
    main()
