# ！usr/bin/python
# -*- coding:utf-8 -*-#
# @date:2019/1/21 10:04
# @name:cnnModel
# @author:TDYe
import tensorflow as tf
from .imageProcessing import *
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import StratifiedShuffleSplit

SYMBOL = {0: '0',
          1: '1',
          2: '2',
          3: '3',
          4: '4',
          5: '5',
          6: '6',
          7: '7',
          8: '8',
          9: '9',
          10: '+',
          11: '-',
          12: '*',
          13: '/',
          14: '(',
          15: ')'}


class TrainTest(object):
    def __init__(self):
        self.images = None
        self.labels = None
        self.offset = 0

    def next_batch(self, batch_size):
        if self.offset + batch_size <= self.images.shape[0]:
            batch_images = self.images[self.offset:self.offset + batch_size]
            batch_labels = self.labels[self.offset:self.offset + batch_size]
            self.offset = (self.offset + batch_size) % self.images.shape[0]
        else:
            new_offset = self.offset + batch_size - self.images.shape[0]
            batch_images = self.images[self.offset:-1]
            batch_labels = self.labels[self.offset:-1]
            batch_images = np.r_[batch_images, self.images[0:new_offset]]
            batch_labels = np.r_[batch_labels, self.labels[0:new_offset]]
            self.offset = new_offset
        return batch_images, batch_labels


class DigitData(object):
    def __init__(self):
        self.train = TrainTest()
        self.test = TrainTest()

    def input_data(self):
        mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
        images = np.r_[mnist.train.images, mnist.test.images]
        labels = np.r_[mnist.train.labels, mnist.test.labels]
        zeros = np.zeros((labels.shape[0], 6))
        labels = np.c_[labels, zeros]
        print("Loading the operators' dataset....")
        op_images, op_labels = get_images_labels()
        images, labels = np.r_[images, op_images], np.r_[labels, op_labels]
        print("Generating the train_data and test_data....")
        sss = StratifiedShuffleSplit(n_splits=16, test_size=0.15, random_state=23)
        for train_index, test_index in sss.split(images, labels):
            self.train.images, self.test.images = images[train_index], images[test_index]
            self.train.labels, self.test.labels = labels[train_index], labels[test_index]


class Model(object):
    def __init__(self, batch_size=100, hidden_size=1024, n_output=16):
        self.HIDDEN_SIZE = hidden_size
        self.BATCH_SIZE = batch_size
        self.N_OUTPUT = n_output
        self.N_BATCH = 0

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.10)  
        return tf.Variable(initial, name="w")

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name="b")

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def train_model(self, EPOCH=21, learning_rate=1e-4, regular_coef=5e-4, model_dir='../static/model/', model_name='model'):
        mnist_operator = DigitData()
        mnist_operator.input_data()
        self.N_BATCH = mnist_operator.train.images.shape[0] // self.BATCH_SIZE
        x = tf.placeholder(tf.float32, [None, 784], name='image_input')
        y = tf.placeholder(tf.float32, [None, self.N_OUTPUT])
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        x_image = tf.reshape(x, [-1, 28, 28, 1])
        with tf.variable_scope("conv1"):
            W_conv1 = self.weight_variable([5, 5, 1, 32])
            b_conv1 = self.bias_variable([32])
            h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = self.max_pool_2x2(h_conv1)

        with tf.variable_scope("conv2"):
            W_conv2 = self.weight_variable([5, 5, 32, 64])
            b_conv2 = self.bias_variable([64])
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = self.max_pool_2x2(h_conv2)

        with tf.variable_scope("fc1"):
            W_fc1 = self.weight_variable([7 * 7 * 64, self.HIDDEN_SIZE])
            b_fc1 = self.bias_variable([self.HIDDEN_SIZE])
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        with tf.variable_scope("fc2"):
            W_fc2 = self.weight_variable([self.HIDDEN_SIZE, self.N_OUTPUT])
            b_fc2 = self.bias_variable([self.N_OUTPUT])
            h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        regularizers = (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc1))
        prediction = tf.nn.softmax(h_fc2, name="prediction")
        predict_op = tf.argmax(prediction, 1, name="predict_op")

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
        loss_re = loss + regular_coef * regularizers

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_re)

        correct_prediction = tf.equal(predict_op, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()
        tf.add_to_collection("predict_op", predict_op)

        print("Start training....")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in tqdm(range(EPOCH * self.N_BATCH)):
                epoch = i // self.N_BATCH
                batch_xs, batch_ys = mnist_operator.train.next_batch(self.BATCH_SIZE)
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
                if epoch % 10 == 0 and (i+1) % self.N_BATCH == 0:
                    acc = []
                    for i in range(mnist_operator.test.labels.shape[0]//self.BATCH_SIZE):
                        batch_xs_test, batch_ys_test = mnist_operator.test.next_batch(self.BATCH_SIZE)
                        test_acc = sess.run(accuracy, feed_dict={x: batch_xs_test, y: batch_ys_test, keep_prob: 1.0})
                        acc.append(test_acc)
                    print()
                    print("Iter" + str(epoch) + ",Testing Accuracy = " + str(sum(acc) / len(acc)))
                    if not os.path.exists(model_dir):
                        os.mkdir(model_dir)
                    saver.save(sess, model_dir + '/' + model_name, global_step=epoch)

    def load_model(self, meta, path):
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(meta)
        saver.restore(self.sess, tf.train.latest_checkpoint(path))

    def predict(self, X):
        predict = tf.get_collection('predict_op')[0]
        graph = tf.get_default_graph()
        input_X = graph.get_operation_by_name("image_input").outputs[0]
        keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
        return self.sess.run(predict, feed_dict={input_X: X, keep_prob: 1.0})[0:]