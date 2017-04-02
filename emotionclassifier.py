import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score


def divide_data(data, test_percent=0.1):
    """ Divides input data into training and testing data.
    :param data: A list to be split
    :type data: A list
    :param test_percent: The percentage for testing data. Default is 0.2.
    :type test_percent: A floating point value between 0 and 1.
    :return: Two lists, training data and testing data.
    :rtype: Two lists.
    """
    test_length = int(round(test_percent * len(data)))
    shuffled = data[:]
    random.shuffle(shuffled)
    training_data = shuffled[test_length:]
    testing_data = shuffled[:test_length]
    return training_data, testing_data


def split_data(seq, num):
    """ Splits a list into a number of smaller list of a given size.
    :param seq: The data to be split.
    :type seq: list.
    :param num: The size of each smaller list.
    :type num: int.
    :return: A list of contraing the split lists.
    :rtype: list.
    """
    count, out = -1, []
    while count < len(seq):
        temp = []
        for i in range(num):
            count += 1
            if count >= len(seq):
                break
            temp.append(seq[count])
        if len(temp) != 0:
            out.append(temp)
    return out


def print_confusion_matrix(p_labels, t_labels):
    p_labels = pd.Series(p_labels)
    t_labels = pd.Series(t_labels)
    df_confusion = pd.crosstab(t_labels, p_labels, rownames=['Actual'], colnames=['Predicted'], margins=True)
    return df_confusion


def confusion_matrix(labels, y_pred, not_partial):
    y_actu = my_own_shit_function(labels)
    df = print_confusion_matrix(y_pred, y_actu)
    print '\n',df
    if not_partial:
       print '\n',classification_report(y_actu, y_pred)


def my_own_shit_function(thing):
    new_thing = np.zeros(len(thing))
    for i in range(len(thing)):
        for j in range(len(thing[i])):
            if thing[i][j] == 1.0:
                new_thing[i] = int(j)
    return new_thing


def do_eval(message, sess, correct_prediction, accuracy, pred, X_, y_, x, y, dropout):
    predictions = sess.run([correct_prediction], feed_dict={x: X_, y: y_, dropout: 1.})
    prediction = tf.argmax(pred,1)
    labels = prediction.eval(feed_dict={x: X_, y: y_, dropout: 1.}, session=sess)
    print message, accuracy.eval({x: X_, y: y_}),'\n'
    confusion_matrix('Partial Confusion matrix', y_,predictions[0], False)  # Partial confusion Matrix
    confusion_matrix('Complete Confusion matrix', y_,labels, True)  # complete confusion Matrix


class EmotionClassifier:

    def __init__(self, num_classes, save_path=''):
        """ Constructor for EmotionClassifier that builds placeholders and the learning model.
        :param num_classes: The number of different classifications.
        :type num_classes: int
        :param save_path: A file path for the session variables to be saved. If not set the session will not be saved.
        :type save_path: A file path.
        """
        self.x = tf.placeholder('float', [None, 88, 88])
        self.y = tf.placeholder('float', [None, num_classes])
        self.keep_prob = tf.placeholder(tf.float32)
        self.model = self.build_model(num_classes)
        self.save_path = save_path

    def build_model(self, num_classes, activation = True):
        """ Builds the Neural model for the classifier.
        :param num_classes: The number of different classifications.
        :type num_classes: int
        :return: The Neural model for the system.
        :rtype: A TensorFlow model.
        """
        weights = {
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 64])),
            'wc2': tf.Variable(tf.random_normal([5, 5, 64, 32])),
            'lc1': tf.Variable(tf.random_normal([3, 3, 32, 1])),
            'lc2': tf.Variable(tf.random_normal([3, 3, 32, 1])),
            'out': tf.Variable(tf.random_normal([15488, num_classes]))
        }
        biases = {
            'bc1': tf.Variable(tf.random_normal([64])),
            'bc2': tf.Variable(tf.random_normal([32])),
            'bl1': tf.Variable(tf.random_normal([32])),
            'bl2': tf.Variable(tf.random_normal([32])),
            'out': tf.Variable(tf.random_normal([num_classes]))
        }

        input = tf.reshape(self.x, shape=[-1, 88, 88, 1])

        conv1 = tf.nn.bias_add(tf.nn.conv2d(input, weights['wc1'], [1, 1, 1, 1], 'SAME'), biases['bc1'])
        if activation: conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, weights['wc2'], [1, 1, 1, 1], 'SAME'), biases['bc2'])
        if activation: conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        local1 = self.local(conv2, 3, 32, 'l1_weights', 'b1_weights')
        if activation: local1 = tf.nn.relu(local1)
        local1 = tf.nn.dropout(local1, self.keep_prob)

        local2 = self.local(local1, 3, 32, 'l2_weights', 'b2_weights')
        if activation: local2 = tf.nn.relu(local2)
        local2 = tf.nn.dropout(local2, self.keep_prob)

        fc1 = tf.reshape(local2, [-1, 15488])
        fc1 = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return fc1

    def local(self, previous_layer, kernel_size, channels, weight_name, bias_name):
        shape = previous_layer.get_shape()
        height = shape[1].value
        width = shape[2].value
        patch_size = (kernel_size ** 2) * shape[3].value

        patches = tf.extract_image_patches(previous_layer, [1,kernel_size,kernel_size,1], [1,1,1,1], [1,1,1,1], 'SAME')
        with tf.device('/cpu:0'):
            weights = tf.get_variable(weight_name, [1, height, width, patch_size, channels],
                                      initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float16))
            biases = tf.get_variable(bias_name, [height, width, channels], initializer=tf.constant_initializer(0.1))

        mul = tf.multiply(tf.expand_dims(patches, axis=-1), weights)
        ssum = tf.reduce_sum(mul, axis=3)
        return tf.add(ssum, biases)

    def train(self, training_data, testing_data, epochs=5000, batch_size=32, intervals=1, log='log'):
        """ Trains a classifier with inputted training and testing data for a number of epochs.
        :param training_data: A list of tuples used for training the classifier.
        :type training_data: A list of tuples each containing a list of landmarks and a list of classifications.
        :param testing_data: A list of tuples used for testing the classifier.
        :type testing_data: A list of tuples each containing a list of landmarks and a list of classifications.
        :param epochs: The number of cycles to train the classifier for. Default is 50000.
        :type epochs: int.
        :param batch_size: The size of each batch to process.
        :type batch_size: int.
        :param intervals: The interval number to print epoch information. If set to 0 no print. Default is 10.
        :type intervals: int.
        """
        start = time.clock()
        batches = split_data(training_data, batch_size)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.model, self.y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        init, saver = tf.global_variables_initializer(), tf.train.Saver()
        correct_prediction = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        avg_loss, avg_acc = 0, 0

        with tf.Session() as sess:
            sess.run(init)
            summary_writer = tf.summary.FileWriter(self.save_path+log, graph=tf.get_default_graph())
            for epoch in range(epochs):
                avg_loss, avg_acc = 0, 0
                for i in range(len(batches)):
                    x, y = [m[0] for m in batches[i]], [n[1] for n in batches[i]]
                    _, loss, acc = sess.run([optimizer, cost, accuracy],
                                                     feed_dict={self.x: x, self.y: y, self.keep_prob: 1.})
                    avg_loss += loss
                    avg_acc += acc
                    summary = tf.Summary()
                    summary.value.add(tag='Accuracy', simple_value=(avg_acc / len(batches)))
                    summary.value.add(tag='Loss', simple_value=(avg_loss / len(batches)))
                summary_writer.add_summary(summary, epoch)
                if epoch % intervals == 0 and intervals != 0 and i == (len(batches)-1):
                    saver.save(sess, self.save_path+'Checkpoints/model', global_step=epoch) if self.save_path != '' else ''
                    end = time.clock()
                    print 'Epoch', '%03d' % (epoch + 1), ' Time = {:.2f}'.format(end - start),\
                        ' Loss = {:.5f}'.format(avg_loss / len(batches))
                    # ' Accuracy = {:.5f}'.format(avg_acc / len(batches)), \

            saver.save(sess, self.save_path+'model') if self.save_path != '' else ''
            batches = split_data(testing_data, batch_size)
            avg_acc, labels, _y = 0, np.zeros(0), []
            for batch in batches:
                x, y = [m[0] for m in batch], [n[1] for n in batch]
                _y += y
                acc = accuracy.eval({self.x: x, self.y: y, self.keep_prob: 1.})
                avg_acc += acc / len(batches)
                prediction = tf.argmax(self.model, 1)
                label = prediction.eval(feed_dict={self.x: x, self.y: y, self.keep_prob: 1.}, session=sess)
                labels = np.append(labels, label)

            confusion_matrix(_y, labels, True)
            return avg_acc

    def accuracy(self, testing_data, batch_size=128):
        """ Finds the accuracy of a model.
        :param testing_data: A list of tuples used for testing the classifier.
        :type testing_data: A list of tuples each containing a list of landmarks and a list of classifications.
        :param batch_size: The size for each batch.
        :type batch_size: int
        :return: The accuracy from the program
        :rtype: double
        """
        init, saver = tf.global_variables_initializer(), tf.train.Saver()
        correct_prediction = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, self.save_path+'model')
            batches = split_data(testing_data, batch_size)
            avg_acc = 0
            for batch in batches:
                x, y = [m[0] for m in batch], [n[1] for n in batch]
                acc = accuracy.eval({self.x: x, self.y: y, self.keep_prob: 1.})
                avg_acc += acc / len(batches)
            return avg_acc

    def classify(self, data):
        """ Loads the pre-trained model and uses the input data to return a classification.
        :param data: The data that is to be classified.
        :type data: A list.
        :return: A classification.
        :rtype: int.
        """
        init, saver = tf.global_variables_initializer(), tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, self.save_path)
            return np.asarray(sess.run(self.model, feed_dict={self.x: data, self.keep_prob: 1.}))
