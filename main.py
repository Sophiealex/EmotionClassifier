import sys
import cv2
import time
import numpy
import select
import builddata
import threading
import websocket
import tensorflow as tf

import emotionclassifier


averages, running = [], True

def main():
    if len(sys.argv) >= 2:
        mode = sys.argv[1]
        if mode == 'train':
            if len(sys.argv) > 5:
                start = time.clock()
                faces = builddata.get_data(sys.argv[2], int(sys.argv[5]))
                training_data, testing_data = emotionclassifier.divide_data(faces, 0.1)
                print 'number of training examples = ' + str(len(training_data))
                print 'number of testing examples  = ' + str(len(testing_data)) + '\n'
                classifier = emotionclassifier.EmotionClassifier(int(sys.argv[5]), sys.argv[3])
                accuracy = classifier.train(training_data, testing_data, epochs=int(sys.argv[4]), intervals=1, log=sys.argv[6])
                end = time.clock()
                print 'Testing Accuracy: ' + '{:.9f}'.format(accuracy)
                print 'Training Time: ' + '{:.2f}'.format(end - start) + 's'
            else:
                print 'Please add \'Image Dir\' \'Session Save Path\' \'Number of Epochs\' \'Number of Classes\' \'log\''

        elif mode == 'accuracy':
            if len(sys.argv) > 4:
                start = time.clock()
                faces = builddata.get_data(sys.argv[2], int(sys.argv[4]))
                _, testing_data = emotionclassifier.divide_data(faces, 0.1)
                print 'number of testing examples  = ' + str(len(testing_data)) + '\n'

                classifier = emotionclassifier.EmotionClassifier(int(sys.argv[4]), sys.argv[3])
                accuracy = classifier.accuracy(testing_data)
                end = time.clock()
                print 'Accuracy: ' + str(accuracy) + ' in ' + str(end - start) + 's'
            else:
                print 'Please add \'Image Dir\' \'Session Save Path\' \'Number of Classes\''

        elif mode == 'classify':
            if len(sys.argv) > 4:
                with tf.Session() as sess:
                    classifier = emotionclassifier.EmotionClassifier(int(sys.argv[4]), sys.argv[3])
                    init, saver = tf.global_variables_initializer(), tf.train.Saver()
                    sess.run(init)
                    saver.restore(sess, classifier.save_path + 'model')
                    start = time.clock()
                    face = builddata.get_face_from_file(sys.argv[2])
                    classification = classifier.classify(face, sess)
                end = time.clock()
                print sys.argv[2]+' classified as '+str(numpy.argmax(classification[0]))+' in '+str(end-start)+'s'
                print classification[0]
            else:
                print 'Please add \'Image Path\' \'Session Save Path\' \'Number of Classes\''

        elif mode == 'run':
            if len(sys.argv) > 3:
                try:
                    run_thread = MyThread(0)
                    network_thread = MyThread(1)
                    run_thread.start()
                    network_thread.start()
                    while network_thread.is_alive():
                        continue
                except Exception as err:
                    print "Error: unable to start thread"
                    print err.message

            else:
                print 'Please add \'Session Save Path\' \'Number of Classes\''

        elif mode == 'build':
            if len(sys.argv) > 5:
                start = time.clock()
                args = [sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]]
                for i in range(len(args)):
                    if args[i][-1] == '/':
                        args[i] = args[i][:len(args[i])-1]
                count = builddata.build_dataset(args[0], args[1], args[2], args[3], mode=int(sys.argv[6]))
                end = time.clock()
                print 'Augmented Dataset built ' + str(count) + ' images in ' + str(end - start) + 's'
            else:
                print 'Please add \'Image Dir\' \'Label Dir\' \'Output Dir\' \'Resources Dir\''

        elif mode == 'split':
            if len(sys.argv) > 2:
                start = time.clock()
                builddata.split_data(sys.argv[2], sys.argv[3])
                end = time.clock()
                print 'The dataset was split into Positive and Negative in ' + str(end - start) + 's'

        elif mode == 'normalize':
            if len(sys.argv) > 2:
                start = time.clock()
                count = builddata.normalize(sys.argv[2])
                end = time.clock()
                print 'Normalized Dataset ' + str(count) + ' images in ' + str(end - start) + 's'
            else:
                print 'Please add \'Image Dir\''
        else:
            print 'Please add either \'train\', \'classify\' or \'build\' as command line arguments.'

    else:
        print 'Please add either \'train\', \'classify\' or \'build\' as command line arguments.'


def get_address_from_config(config_file):
    try:
        with open(config_file, 'r') as file:
            for line in file.readlines():
                line.strip()
                if line and len(line) > 0:
                    pair = line.split("=")
                    if pair[0] == 'EmotionPort':
                        return 'ws://localhost:' + pair[1].replace('\n', '') + '/websocket'
        return 0
    except IOError as err:
        print err.message


class MyThread (threading.Thread):
    def __init__(self, id):
        threading.Thread.__init__(self)
        self.id = id

    def run(self):
        if self.id == 0:
            print 'Starting ' + str(self.id)
            run()
            print 'Exiting ' + str(self.id)
        elif self.id == 1:
            print 'Starting ' + str(self.id)
            run_network()
            print 'Exiting ' + str(self.id)


def run():
    global averages
    init, saver = tf.global_variables_initializer(), tf.train.Saver()
    start, video, q = time.clock(), cv2.VideoCapture(0), []
    classifier = emotionclassifier.EmotionClassifier(int(sys.argv[3]), sys.argv[2])
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, classifier.save_path+'model')
        if video.grab():
            while running:
                _, frame = video.read()
                face = builddata.get_face_from_frame(frame)
                if face:
                    temp = []
                    print 'Face Found'
                    classification = classifier.classify(face, sess)[0]
                    if len(q) < 10:
                        q.insert(0, classification)
                    else:
                        q.pop()
                        q.insert(0, classification)
                    for i in range(len(q[0])):
                        average = 0
                        for j in range(len(q)):
                            average += q[j][i] / 10
                        temp.append(average)
                    averages = temp
                    print averages
                else:
                    print 'Face Not Found'
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    raw_input()
                    break
        else:
            print 'No Camera'


def softmax(array):
    e = numpy.exp(numpy.array(array) / 1.0)
    dist = e / numpy.sum(e)
    return dist


def on_open(ws):
    ws.send('EMOTION')
    print 'Connected to server!'


def on_close(ws):
    print 'Connection Closed'


def on_error(ws, err):
    print err.message


def on_data(ws, data):
    print 'Recived: ' + data
    if data == 'REQUEST':
        string = ''
        for i in averages:
            string += str(i) + ' '
        print 'Sent: ' + string
        ws.send(string.replace('[','').replace(']',''))
    elif data == 'CLOSE':
        ws.close()


def run_network():
    global running
    address = get_address_from_config('../config.txt')
    try:
        ws = websocket.WebSocketApp(address, on_message=on_data, on_error=on_error, on_close=on_close)
        ws.on_open = on_open
        ws.run_forever()
        running = False
    except Exception as err:
        print err.message


if __name__ == '__main__':
    main()
