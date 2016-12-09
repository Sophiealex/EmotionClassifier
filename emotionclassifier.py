import random
import numpy as np
import tensorflow as tf


def divide_data(data, test_percent=0.2):
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
        out.append(temp)
    return out


class EmotionClassifier:

    def __init__(self, num_classes, save_path=''):
        """ Constructor for EmotionClassifier that builds placeholders and the learning model.
        :param num_classes: The number of different classifications.
        :type num_classes: int
        :param save_path: A file path for the session variables to be saved. If not set the session will not be saved.
        :type save_path: A file path.
        """
        self.x = tf.placeholder("float", [None, 88, 88])
        self.y = tf.placeholder("float", [None, num_classes])
        self.model = self.build_model(num_classes)
        self.save_path = save_path

    def build_model(self, num_classes):
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
            'fc1': tf.Variable(tf.random_normal([15488, 1024])),
            'out': tf.Variable(tf.random_normal([1024, num_classes]))
        }
        biases = {
            'bc1': tf.Variable(tf.random_normal([64])),
            'bc2': tf.Variable(tf.random_normal([32])),
            'bl1': tf.Variable(tf.random_normal([32])),
            'bl2': tf.Variable(tf.random_normal([32])),
            'fc1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([num_classes]))
        }

        input = tf.reshape(self.x, shape=[-1, 88, 88, 1])
        conv1 = tf.nn.bias_add(tf.nn.conv2d(input, weights['wc1'], [1, 1, 1, 1], 'SAME'), biases['bc1'])
        conv1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, weights['wc2'], [1, 1, 1, 1], 'SAME'), biases['bc2'])
        conv2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        local1 = tf.nn.bias_add(tf.nn.depthwise_conv2d(conv2, weights['lc1'], [1, 1, 1, 1], 'SAME'), biases['bl1'])
        local1 = tf.nn.dropout(local1, 0.5)
        local2 = tf.nn.bias_add(tf.nn.depthwise_conv2d(local1, weights['lc2'], [1, 1, 1, 1], 'SAME'), biases['bl2'])
        local2 = tf.nn.dropout(local2, 0.5)

        shape = local2.get_shape().as_list()
        fc1 = tf.reshape(local2, [-1, shape[1] * shape[2] * shape[3]])
        fc1 = tf.add(tf.matmul(fc1, weights['fc1']), biases['fc1'])
        fc1 = tf.nn.relu(fc1)
        return tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    def train(self, training_data, testing_data, epochs=50000, batch_size=10, verbose=True):
        """ Trains a classifier with inputted training and testing data for a number of epochs.
        :param training_data: A list of tuples used for training the classifier.
        :type training_data: A list of tuples each containing a list of landmarks and a list of classifications.
        :param testing_data: A list of tuples used for testing the classifier.
        :type testing_data: A list of tuples each containing a list of landmarks and a list of classifications.
        :param epochs: The number of cycles to train the classifier for. Default is 50000.
        :type epochs: int.
        :param batch_size: The size of each batch to process.
        :type batch_size: int.
        """
        batches = split_data(training_data, batch_size)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.model, self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
        init, saver = tf.initialize_all_variables(), tf.train.Saver()
        correct_prediction = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(epochs):
                avg_acc = 0
                for batch in batches:
                    x, y = [m[0] for m in batch], [n[1] for n in batch]
                    _, acc = sess.run([optimizer, accuracy], feed_dict={self.x: x, self.y: y})
                    avg_acc += acc
                if epoch % 10 == 0 and verbose:
                    print 'Epoch', '%04d' % epoch, ' Training Accuracy = ', '{:.9f}'.format(avg_acc/batch_size)

            saver.save(sess, self.save_path) if self.save_path != '' else ''
            return accuracy.eval({self.x: [m[0] for m in testing_data], self.y: [n[1] for n in testing_data]})

    def classify(self, data):
        """ Loads the pre-trained model and uses the input data to return a classification.
        :param data: The data that is to be classified.
        :type data: A list.
        :return: A classification.
        :rtype: int.
        """
        init, saver = tf.initialize_all_variables(), tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, self.save_path)
            classification = np.asarray(sess.run(self.model, feed_dict={self.x: data}))
            return np.unravel_index(classification.argmax(), classification.shape)
