#!/usr/bin/env python

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import sys

import cv2

import Image as Img
import numpy
import tensorflow as tf
import numpy as np
import rospy
import message_filters

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
SEED = 66478
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100
WORK_DIRECTORY = '/tmp/data'

def main(argv=None):
  # Trained model parameter check
  if False == os.path.isfile(WORK_DIRECTORY + '/checkpoint'):
    print('The model parameter does not exit. Please begin the model parameter (train.py)')
    sys.exit()

  conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED))
  conv1_biases = tf.Variable(tf.zeros([32]))
  conv2_weights = tf.Variable(
      tf.truncated_normal([5, 5, 32, 64],
                          stddev=0.1,
                          seed=SEED))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
  fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal(
          [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
          stddev=0.1,
          seed=SEED))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
  fc2_weights = tf.Variable(
      tf.truncated_normal([512, NUM_LABELS],
                          stddev=0.1,
                          seed=SEED))
  fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

  """The Model definition."""
  def model(data, train=False):    
    conv = tf.nn.conv2d(data, 
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape( pool,[pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

  # Saves memory and enables this to run on smaller GPUs.
  
  def eval_image(imagePath):
    imData = [y for x in imagePath for y in x] # Mat style list 28 x 28 -> line vector style list len = 784
    data = numpy.fromiter(imData, dtype=numpy.uint8).astype(numpy.float32)    
    imData = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    reshapeData = imData.reshape([1, IMAGE_SIZE , IMAGE_SIZE , 1])
    test = model(reshapeData, False)
    return test    
    
  # Create a local session to run the training.
  with tf.Session() as sess:
    saver = tf.train.Saver()
    sess = tf.Session()
    
    # load the variables from disk.
    model_path = WORK_DIRECTORY+'/mnist.ckpt'
    saver.restore(sess, model_path)

    bridge = CvBridge()
    print("Model restored")
    rospy.init_node('listener')

  # ROS Subscriber, Topic : sensor_msgs.msg Image
    def callback(data):
      result = eval_image(bridge.imgmsg_to_cv2(data))
      r = sess.run(result) 
      print('Result : ', numpy.argmax(r))

  # Recognize loop.
    while(True) : 
     rospy.Subscriber('video', Image, callback)
     rospy.spin()

    sys.stdout.flush()  

if __name__ == '__main__':
  tf.app.run()