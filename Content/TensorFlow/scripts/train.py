##################################################
## Author: RuochenLiu
## Email: ruochen.liu@columbia.edu
## Version: 1.0.0
##################################################
import os
import os.path
import time
import pickle
from PIL import Image
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tensorflow.compat.v1 as tf
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import graph_util
import preprocess

SEED = 42
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_random_seed(SEED)

MODEL_LOG_DIR = "../log/"
TRAIN_LOG_DIR = MODEL_LOG_DIR + "train/"
VAL_LOG_DIR = MODEL_LOG_DIR + "val/"

def stand(img):
    return (img - np.mean(img))/np.std(img)

def batch(X_train, y_train, batch_size):
    n = X_train.shape[0]
    index = np.random.randint(0, n-1, batch_size)
    return X_train[index], y_train[index]

def get_one_hot(targets, n_classes):
    res = np.eye(n_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[n_classes])

def load_data(batch_size):
    X = np.load("../data/X.npy")
    y = np.load("../data/y.npy")

    data_train, data_val, label_train, label_val = train_test_split(X, y, test_size=0.1)

    n_train = data_train.shape[0]/batch_size
    n_val = data_val.shape[0]/batch_size
    
    return np.split(data_train, n_train), np.split(data_val, n_val), np.split(label_train, n_train), np.split(label_val, n_val), int(n_train), int(n_val)

def conv(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def maxpool(x):
    return tf.nn.max_pool(x, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")

class Network:
    def __init__(self, session, image_size, n_out):
        self.session = session
        self.image_size = image_size
        self.h, self.w, self.num_channels = self.image_size
        self.n_out = n_out
        self.global_step = 0
        
        self.n_conv1 = 64
        self.n_conv2 = 64
        self.n_conv3 = 256
        self.n_conv4 = 256

        self.n_fc1 = 256
        self.n_fc2 = 128
        self.n_fc3 = 64
        self.n_fc4 = 32

        with tf.name_scope('Input'): 
            self.x = tf.placeholder(tf.float32, [None, self.h, self.w, self.num_channels], name="x")
            self.y = tf.placeholder(tf.float32, [None, self.n_out], name="y")
        
        with tf.name_scope('ConvolutionalLayers'):
            self.w_conv1 = tf.get_variable("w_conv1", shape=[3, 3, self.num_channels, self.n_conv1])
            self.b_conv1 = tf.get_variable("b_conv1", shape=[self.n_conv1])
            self.h_conv1 = maxpool(tf.nn.relu(tf.add(conv(self.x, self.w_conv1), self.b_conv1)))

            self.w_conv2 = tf.get_variable("w_conv2", shape=[3, 3, self.n_conv1, self.n_conv2])
            self.b_conv2 = tf.get_variable("b_conv2", shape=[self.n_conv2])
            self.h_conv2 = maxpool(tf.nn.relu(tf.add(conv(self.h_conv1, self.w_conv2), self.b_conv2)))

            self.w_conv3 = tf.get_variable("w_conv3", shape=[3, 3, self.n_conv2, self.n_conv3])
            self.b_conv3 = tf.get_variable("b_conv3", shape=[self.n_conv3])
            self.h_conv3 = maxpool(tf.nn.relu(tf.add(conv(self.h_conv2, self.w_conv3), self.b_conv3)))

            self.w_conv4 = tf.get_variable("w_conv4", shape=[3, 3, self.n_conv3, self.n_conv4])
            self.b_conv4 = tf.get_variable("b_conv4", shape=[self.n_conv4])
            self.h_conv4 = maxpool(tf.nn.relu(tf.add(conv(self.h_conv3, self.w_conv4), self.b_conv4)))
        
        with tf.name_scope('FullyConnectedLayers'):
            self.h_flat = tf.reshape(self.h_conv4, [-1, self.n_conv4*int(self.h/16)*int(self.w/16)])
            self.w_fc1 = tf.get_variable("w_fc1", shape=[self.n_conv4*int(self.h/16)*int(self.w/16), self.n_fc1])
            self.b_fc1 = tf.get_variable("b_fc1", shape=[self.n_fc1])
            self.h_fc1 = tf.nn.relu(tf.add(tf.matmul(self.h_flat, self.w_fc1), self.b_fc1))

            self.w_fc2 = tf.get_variable("w_fc2", shape=[self.n_fc1, self.n_fc2])
            self.b_fc2 = tf.get_variable("b_fc2", shape=[self.n_fc2])
            self.h_fc2 = tf.nn.relu(tf.add(tf.matmul(self.h_fc1, self.w_fc2), self.b_fc2))

            self.w_fc3 = tf.get_variable("w_fc3", shape=[self.n_fc2, self.n_fc3])
            self.b_fc3 = tf.get_variable("b_fc3", shape=[self.n_fc3])
            self.h_fc3 = tf.nn.relu(tf.add(tf.matmul(self.h_fc2, self.w_fc3), self.b_fc3))

            self.w_fc4 = tf.get_variable("w_fc4", shape=[self.n_fc3, self.n_fc4])
            self.b_fc4 = tf.get_variable("b_fc4", shape=[self.n_fc4])
            self.h_fc4 = tf.nn.relu(tf.add(tf.matmul(self.h_fc3, self.w_fc4), self.b_fc4))

            self.w_fc = tf.get_variable("w_fc", shape=[self.n_fc4, self.n_out])
            self.b_fc = tf.get_variable("b_fc", shape=[self.n_out])

        with tf.name_scope('Output'):
            self.logits = tf.add(tf.matmul(self.h_fc4, self.w_fc), self.b_fc)
            self.proba = tf.nn.softmax(self.logits, name="prob")
        
        with tf.name_scope('Train'):
            self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.lr = tf.train.exponential_decay(1e-4, self.global_step, 1000, 0.9)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y))
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
            self.train_summary = tf.summary.merge([tf.summary.scalar("train_accuracy", self.accuracy), tf.summary.scalar("train_loss", self.loss)])
            self.val_summary = tf.summary.merge([tf.summary.scalar("val_accuracy", self.accuracy)])
            self.train_writer = tf.summary.FileWriter(TRAIN_LOG_DIR, session.graph)
            self.val_writer = tf.summary.FileWriter(VAL_LOG_DIR, session.graph)

        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.session.run(self.init)


    def train(self, x_batch, y_batch):
        self.global_step += 1
        self.train_tf_summary, _ = self.session.run([self.train_summary, self.train_step], feed_dict={self.x: x_batch, self.y: y_batch})
        self.train_writer.add_summary(self.train_tf_summary, self.global_step)

    def val(self, x_batch, y_batch):
        self.val_tf_summary = self.session.run(self.val_summary, feed_dict={self.x: x_batch, self.y: y_batch})
        self.val_writer.add_summary(self.val_tf_summary, self.global_step)

    def get_acc(self, x_batch, y_batch):
        return self.session.run(self.accuracy, feed_dict={self.x: x_batch, self.y: y_batch})

    def get_loss(self, x_batch, y_batch):
        return self.session.run(self.loss, feed_dict={self.x: x_batch, self.y: y_batch})

    def get_proba(self, x_batch):
        return self.session.run(self.proba, feed_dict={self.x: x_batch})


def main():
    #preprocess.preprocess_data()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if not os.path.exists(MODEL_LOG_DIR):
        os.makedirs(MODEL_LOG_DIR)
    if not os.path.exists(TRAIN_LOG_DIR):
        os.makedirs(TRAIN_LOG_DIR)
    if not os.path.exists(VAL_LOG_DIR):
        os.makedirs(VAL_LOG_DIR)
    
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    image_size = (128, 128, 1)
    n_class = 2
    batch_size = 16
    x_train, x_val, y_train, y_val, train_steps, val_steps = load_data(batch_size)

    train_epochs = 10
    train_acc = []
    train_loss = []

    network = Network(sess, image_size, n_class)
    saver = tf.train.Saver()

    time_now = time.asctime(time.localtime(time.time()))[11:19]
    print("{0} === Start training: {1} epochs + {2} batches each\n".format(time_now, train_epochs, train_steps))
    
    for j in range(train_epochs):
        train_acc.append([])
        train_loss.append([])

        for i in range(train_steps):
            network.train(x_train[i], y_train[i])
            train_acc[-1].append(network.get_acc(x_train[i], y_train[i]))
            train_loss[-1].append(network.get_loss(x_train[i], y_train[i]))
            time_now = time.asctime(time.localtime(time.time()))[11:19]
            print("{0} --- Processing {1:.2f}% acc: {2:.2f}% loss: {3:.3f}     ".format(time_now, (i+1)/train_steps*100, np.mean(train_acc[-1])*100, np.mean(train_loss[-1])), end="\r")
        
        print("{0} *** Epoch #{1} train acc: {2:.2f}% loss: {3:.3f}     ".format(time_now, j+1, np.mean(train_acc[-1])*100, np.mean(train_loss[-1])), end="\n")

        val_acc = np.mean([network.get_acc(x_val[k], y_val[k]) for k in range(val_steps)])
        val_loss = np.mean([network.get_loss(x_val[k], y_val[k]) for k in range(val_steps)])
        time_now = time.asctime(time.localtime(time.time()))[11:19]
        print("{0} *** Epoch #{1} valid acc: {2:.2f}% loss: {3:.3f}\n".format(time_now, j+1, val_acc*100, val_loss), end="\n")

    # Graph Only
    with tf.gfile.GFile('../output/trained_model/cat_dog.pb', mode='wb') as f:
        f.write(tf.get_default_graph().as_graph_def().SerializeToString())
    # Checkpoints
    saver.save(sess, '../output/trained_model/cat_dog.ckpt')

    # Graph and constant weights
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,  # The session is used to retrieve the weights
        tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
        output_node_names=['Output/prob']  # The output node names are used to select the usefull nodes
      )

    with tf.gfile.GFile('../output/trained_model/pb_only/cat_dog.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())


    time_now = time.asctime(time.localtime(time.time()))[11:19]    
    print("{0} === Finished\n".format(time_now))
    
    sess.close()
    return 0

if __name__ == "__main__":
    main()
