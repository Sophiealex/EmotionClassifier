import sys
import builddata
import emotionclassifier


def main():
    if len(sys.argv) >= 2:
        mode = sys.argv[1]
        if mode == 'train':
            if len(sys.argv) > 4:
                faces = builddata.get_data(sys.argv[2])
                training_data, testing_data = emotionclassifier.split_data(faces)
                print 'number of training examples = ' + str(len(training_data))
                print 'number of testing examples  = ' + str(len(testing_data)) + '\n'
                classifier = emotionclassifier.EmotionClassifier(8, sys.argv[3])
                classifier.train(training_data, testing_data, int(sys.argv[4]))
            else:
                print 'Please add \'Image Dir\' \'Session Save Path\' \'Number of Epochs\''

        elif mode == 'classify':
            if len(sys.argv) > 3:
                face = builddata.get_face(sys.argv[2])
                classifier = emotionclassifier.EmotionClassifier(8, sys.argv[3])
                classification = classifier.classify(face)
                print sys.argv[2] + ' -> ' + str(classification[1])
            else:
                print 'Please add \'Image Path\' \'Session Save Path\''

        elif mode == 'build':
            if len(sys.argv) > 4:
                builddata.build_data(sys.argv[2], sys.argv[3], sys.argv[4])
            else:
                'Please add \'Image Dir\' \'Label Dir\' \'Output Dir\''

        else:
            print 'Please add either \'train\', \'classify\' or \'build\' as command line arguments.'

    else:
        print 'Please add either \'train\', \'classify\' or \'build\' as command line arguments.'


if __name__ == '__main__':
    main()