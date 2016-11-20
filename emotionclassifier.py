import random
import numpy as np
import tensorflow as tf


def split_data(data, test_percent=0.2):
    """ Splits input data into training and testing data.
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


class EmotionClassifier:

    def __init__(self, num_classes, save_path=''):
        """ Constructor for EmotionClassifier that builds placeholders and the learning model.
        :param num_classes: The number of different classifications.
        :type num_classes: int
        :param save_path: A file path for the session variables to be saved. If not set the session will not be saved.
        :type save_path: A file path.
        """
        self.x = tf.placeholder("float", [None, 134])
        self.y = tf.placeholder("float", [None, num_classes])
        self.model = self.build_model(num_classes)
        self.save_path = save_path

    def build_model(self, num_inputs, num_classes):
        """ Builds the Neural model for the classifier.
        :param num_classes: The number of different classifications.
        :type num_classes: int
        :return: The Neural model for the system.
        :rtype: A TensorFlow model.
        """
        weights = {
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
            'out': tf.Variable(tf.random_normal([1024, num_classes]))
        }
        biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([num_classes]))
        }

        m = tf.reshape(self.x, shape=[-1, 28, 28, 1])
        m = tf.nn.bias_add(tf.nn.conv1d(m, weights['wc1'], stride=[1, 1, 1, 1], padding='SAME'), biases['bc1'])
        m = tf.nn.max_pool(m, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        m = tf.nn.bias_add(tf.nn.conv1d(m, weights['wc2'], stride=[1, 1, 1, 1], padding='SAME'), biases['bc2'])
        m = tf.nn.max_pool(m, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        m = tf.reshape(m, [-1, weights['wd1'].get_shape().as_list()[0]])
        m = tf.add(tf.matmul(m, weights['wd1']), biases['bc1'])
        m = tf.nn.relu(m)
        m = tf.nn.dropout(m, 0.7)
        return tf.add(tf.matmul(m, weights['out']), biases['bd1'])

    def train(self, training_data, testing_data, epochs=50000):
        """ Trains a classifier with inputted training and testing data for a number of epochs.
        :param training_data: A list of tuples used for training the classifier.
        :type training_data: A list of tuples each containing a list of landmarks and a list of classifications.
        :param testing_data: A list of tuples used for testing the classifier.
        :type testing_data: A list of tuples each containing a list of landmarks and a list of classifications.
        :param epochs: The number of cycles to train the classifier for. Default is 50000.
        :type epochs: int.
        """
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.model, self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
        init, saver = tf.initialize_all_variables(), tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            correct_prediction = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            for epoch in range(epochs):
                batch_x, batch_y = [m[0] for m in training_data], [n[1] for n in training_data]
                _, avg_cost = sess.run([optimizer, cost], feed_dict={self.x: batch_x, self.y: batch_y})
                if epoch % 5000 == 0:
                    print "Epoch", '%04d' % (epoch), "cost = ", "{:.9f}".format(avg_cost)

            print "Optimization Finished!"
            saver.save(sess, self.save_path) if self.save_path != '' else ''
            print "Accuracy:", accuracy.eval({self.x: [m[0] for m in testing_data],
                                              self.y: [n[1] for n in testing_data]})

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
