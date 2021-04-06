import tensorflow as tf
import os


class ESPCN:

    def __init__(self, input, scale, learning_rate):
        self.LR_input = input
        self.scale = scale
        self.learning_rate = learning_rate
        self.saver = ""

    def ESPCN_model(self):
        """
        Implementation of ESPCN: https://arxiv.org/abs/1609.05158
        Returns
        ----------
        Model
        """

        scale = self.scale
        channels = 1
        bias_initializer = tf.constant_initializer(value=0.1) # 제공된 값으로 모든 것을 초기화 함
        initializer = tf.contrib.layers.xavier_initializer_conv2d() # Returns an initializer performing "Xavier" initialization for weights.
        # initializer = tf.contrib.layers.variance_scaling_initializer()

        filters = [
            tf.Variable(initializer(shape=(5, 5, channels, 64)), name="f1"),  # (f1,n1) = (5,64)    # shape=(size, size, input_channel, ouput_channel)
            tf.Variable(initializer(shape=(3, 3, 64, 32)), name="f2"),  # (f2,n2) = (3,32)
            tf.Variable(initializer(shape=(3, 3, 32, channels * (scale * scale))), name="f3")  # (f3) = (3)
        ]

        bias = [
            tf.get_variable(shape=[64], initializer=bias_initializer, name="b1"),
            tf.get_variable(shape=[32], initializer=bias_initializer, name="b2"),
            tf.get_variable(shape=[channels * (scale * scale)], initializer=bias_initializer, name="b3")  # H x W x r^2
        ]
            #tf.nn.conv2d(input, filter, strides, padding, use_cudnn_n_gpu=True, data_format='NHWC', dilations=[1,1,1,1], name=None)
        l1 = tf.nn.conv2d(self.LR_input, filters[0], [1, 1, 1, 1], padding='SAME', name="conv1") # NHWC : [batch, height, width, channels]
        l1 = l1 + bias[0]
        l1 = tf.nn.relu(l1)

        l2 = tf.nn.conv2d(l1, filters[1], [1, 1, 1, 1], padding='SAME', name="conv2")
        l2 = l2 + bias[1]
        l2 = tf.nn.relu(l2)

        l3 = tf.nn.conv2d(l2, filters[2], [1, 1, 1, 1], padding='SAME', name="conv3")
        l3 = l3 + bias[2]

        # Depth_to_space is equivalent to the pixel shuffle layer.
        # depth의 데이터를 공간적인 데이터의 블록으로 재배열
        # block_size는 input의 블록 크기와 데이터가 어떻게 이동될지를 지정함
        # 이 함수는 depth 차원의 값들을 height와 width 차원들의 공간적인 블록으로 이동시킴
        out = tf.nn.depth_to_space(l3, scale, data_format='NHWC')  # (input, block_size, data_format='NHWC', name=None)

        out = tf.nn.tanh(out, name="NHWC_output")

        out_nchw = tf.transpose(out, [0, 3, 1, 2], name="NCHW_output") # (a, perm=None, name='transpose') a를 전치, perm에 따라 차원의 순서 구성
        # out = tf.nn.relu(out, name="NHWC_output")

        self.saver = tf.train.Saver()

        return out

    def ESPCN_trainable_model(self, HR_out, HR_orig):
        psnr = tf.image.psnr(HR_out, HR_orig, max_val=1.0) # (a, b, max_val, name)
        # a: First set of images, b: Second set of images.
        # max_val: The dynamic range of the images (i.e., the difference between the maximum the and minimum allowed values).

        loss = tf.losses.mean_squared_error(HR_orig, HR_out)

        # train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
        # train_op = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        return loss, train_op, psnr