#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import h5py

L2_LAMBDA = 1e-04


def vel_omega_from_cmds(vel_left, vel_right):
    # compute duty cycle gain
    # Distance between the wheels
    baseline = 0.102
    # assuming same motor constants k for both motors
    k_r = 27.0
    k_l = 27.0
    gain = 1.0
    trim = 0.0
    radius = 0.0318

    k_r_inv = (gain + trim) / k_r
    k_l_inv = (gain - trim) / k_l

    # Commands are stored as bytes
    vel_left = vel_left.astype(float)
    vel_right = vel_right.astype(float)

    # Conversion from motor duty to motor rotation rate
    omega_r = np.divide(vel_left, k_r_inv)
    omega_l = np.divide(vel_right, k_l_inv)

    # Compute linear and angular velocity of the platform
    v = (radius * omega_r + radius * omega_l) / 2.0
    omega = (radius * omega_r - radius * omega_l) / baseline
    return v, omega


def load_data(file_path, train_or_test="training"):
    """
    Loads images and velocities from hdf files and checks for potential mismatch in the number of images and velocities
    :param file_path: path to the hdf file from which it will extract the data
           train_or_test: String specifies whether training or testset partition is loaded.
    :return: velocities, images as numpy arrays
    """
    # read dataframes
    with h5py.File(file_path, 'r') as f:
        data = f["split"][train_or_test]
        vel_left = data['vel_left'][()]
        vel_right = data['vel_right'][()]
        # v, omega = vel_omega_from_cmds(vel_left, vel_right)

        velocities = np.concatenate((vel_left[:, np.newaxis], vel_right[:, np.newaxis]), axis=1)
        images = data['images'][()]

        print('The dataset is loaded: {} images and {} omega velocities.'.format(images.shape[0], velocities.shape[0]))

    if not images.shape[0] == velocities.shape[0]:
        raise ValueError("The number of images and velocities must be the same.")

    return velocities, images


def form_model_name(batch_size, lr, optimizer, epochs):
    '''
    Creates name of model as a string, based on the defined hyperparameters used in training

    :param batch_size: batch size
    :param lr: learning rate
    :param optimizer: optimizer (e.g. GDS, Adam )
    :param epochs: number of epochs
    :return: name of model as a string
    '''

    # return "batch={},lr={},optimizer={},epochs={}_grayscale".format(batch_size, lr, optimizer, epochs)
    return "batch={},lr={},optimizer={},epochs={}".format(batch_size, lr, optimizer, epochs)


def _residual_block(x, size, dropout=True, dropout_prob=0.5, seed=None):
    residual = tf.layers.batch_normalization(x)  # TODO: check if the defaults in Tf are the same as in Keras
    residual = tf.nn.relu(residual)
    residual = tf.layers.conv2d(residual, filters=size, kernel_size=3, strides=2, padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                                kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))
    if dropout:
        residual = tf.nn.dropout(residual, dropout_prob, seed=seed)
    residual = tf.layers.batch_normalization(residual)
    residual = tf.nn.relu(residual)
    residual = tf.layers.conv2d(residual, filters=size, kernel_size=3, padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                                kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))
    if dropout:
        residual = tf.nn.dropout(residual, dropout_prob, seed=seed)

    return residual


def one_residual(x, keep_prob=0.5, seed=None):
    nn = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=2, padding='same',
                          kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                          kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))
    nn = tf.layers.max_pooling2d(nn, pool_size=3, strides=2)

    rb_1 = _residual_block(nn, 32, dropout_prob=keep_prob, seed=seed)

    nn = tf.layers.conv2d(nn, filters=32, kernel_size=1, strides=2, padding='same',
                          kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                          kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))
    nn = tf.keras.layers.add([rb_1, nn])

    nn = tf.layers.flatten(nn)

    return nn


class CNN_training:

    def __init__(self, batch, epochs, learning_rate, optimizer):

        self.batch_size = batch
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer

    def backpropagation(self):
        '''
        Executes backpropagation during training based on the defined optimizer,learning rate and loss function

        '''

        # define the optimizer
        with tf.name_scope("Optimizer"):
            if self.optimizer == "Adam":
                return tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-4).minimize(self.loss)
            elif self.optimizer == "GDS":
                return tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def loss_function(self):
        '''
        Calculates the loss during training using the predicted and true values(in this case velocities)

        '''

        # define loss function and encapsulate its scope
        with tf.name_scope("Loss"):
            return tf.reduce_mean(tf.square(self.vel_pred - self.vel_true))

    def model(self, x):
        '''
        Define model of CNN under the TensorFlow scope "ConvNet".
        The scope is used for better organization and visualization in TensorBoard

        :return: output layer
        '''

        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):

            # define the 4-d tensor expected by TensorFlow
            # [-1: arbitrary num of images, img_height, img_width, num_channels]
            x_img = tf.reshape(x, [-1, 48, 96, 1])

            hl_conv_1 = one_residual(x_img)
            # batch_normed = tf.keras.layers.BatchNormalization()(x_img, training=mode)
            # define 1st convolutional layer
            # hl_conv_1 = tf.layers.conv2d(x_img, kernel_size=5, filters=8, padding="valid",
            #                              activation=tf.nn.relu, name="conv_layer_1")

            # max_pool_1 = tf.layers.max_pooling2d(hl_conv_1, pool_size=2, strides=2)

            # define 2nd convolutional layer
            # hl_conv_2 = tf.layers.conv2d(max_pool_1, kernel_size=5, filters=16, padding="valid",
            #                              activation=tf.nn.relu, name="conv_layer_2")

            # max_pool_2 = tf.layers.max_pooling2d(hl_conv_2, pool_size=2, strides=2)

            # flatten tensor to connect it with the fully connected layers
            # conv_flat = tf.layers.flatten(max_pool_1)

            # add 1st fully connected layers to the neural network
            hl_fc_1 = tf.layers.dense(inputs=hl_conv_1, units=64, activation=tf.nn.relu, name="fc_layer_1")

            # add 1st fully connected layers to the neural network
            hl_fc_1_2 = tf.layers.dense(inputs=hl_fc_1, units=32, activation=tf.nn.relu, name="fc_layer_1_2")

            # add 2nd fully connected layers to predict the driving commands
            hl_fc_2 = tf.layers.dense(inputs=hl_fc_1_2, units=2, name="fc_layer_2")

            return hl_fc_2

    def epoch_iteration(self, data_size, x_data, y_data, mode):
        '''
        For each epoch extract batches and execute train or test step depending on the inserted mode

        :param data_size: number of velocities and images
        :param x_data: images
        :param y_data: velocities
        :param mode: 'train' or 'test' in order to define if backpropagation is executed as well or not
        :return: sum of loss at each epoch
        '''

        pred_loss = 0
        i = 0
        n_batches = 0
        while i <= data_size - 1:
            n_batches += 1
            # extract batch
            if i + self.batch_size <= data_size - 1:
                train_x = x_data[i: i + self.batch_size]
                train_y = y_data[i: i + self.batch_size]
            else:
                train_x = x_data[i:]
                train_y = y_data[i:]

            if mode == 'train':
                # train using the batch and calculate the loss
                _, c = self.sess.run([self.opt, self.loss], feed_dict={self.x: train_x, self.vel_true: train_y})

            elif mode == 'test':
                # train using the batch and calculate the loss
                c = self.sess.run(self.loss, feed_dict={self.x: train_x, self.vel_true: train_y})

            pred_loss += c
            i += self.batch_size

        return pred_loss / n_batches

    def training(self, model_name, train_velocities, train_images, test_velocities, test_images):

        # define paths to save the TensorFlow logs
        model_path = os.path.join(os.getcwd(), model_name, 'tensorflow_logs')
        logs_train_path = os.path.join(model_path, 'train')
        logs_test_path = os.path.join(model_path, 'test')
        graph_path = os.path.join(model_path, 'graph')

        # manual scalar summaries for loss tracking
        man_loss_summary = tf.Summary()
        man_loss_summary.value.add(tag='Loss', simple_value=None)

        # define placeholder variable for input images (each images is a row vector [1, 4608 = 48x96x1])
        self.x = tf.placeholder(tf.float16, shape=[None, 48 * 96], name='x')

        # define placeholder for the true omega velocities
        # [None: tensor may hold arbitrary num of velocities, number of omega predictions for each image]
        self.vel_true = tf.placeholder(tf.float16, shape=[None, 2], name="vel_true")
        self.vel_pred = self.model(self.x)

        self.loss = self.loss_function()
        self.opt = self.backpropagation()

        # initialize variables
        init = tf.global_variables_initializer()

        # Operation to save and restore all variables
        saver = tf.train.Saver()

        with tf.Session() as self.sess:

            # run initializer
            self.sess.run(init)

            # operation to write logs for Tensorboard
            tf_graph = self.sess.graph
            test_writer = tf.summary.FileWriter(logs_test_path, graph=tf.get_default_graph() )
            test_writer.add_graph(tf_graph)

            train_writer = tf.summary.FileWriter(logs_train_path, graph=tf.get_default_graph() )
            train_writer.add_graph(tf_graph)

            # IMPORTANT: this is a crucial part for compiling TensorFlow graph to a Movidius one later in the pipeline.
            # The important file to create is the 'graph.pb' which will be used to freeze the TensorFlow graph.
            # The 'graph.pbtxt' file is just the same graph in txt format in case you want to check the format of the
            # saved information.
            tf.train.write_graph(tf_graph.as_graph_def(), graph_path, 'graph.pbtxt', as_text=True)
            tf.train.write_graph(tf_graph.as_graph_def(), graph_path, 'graph.pb', as_text=False)

            for epoch in range(self.epochs):

                # run train cycle
                avg_train_loss = self.epoch_iteration(train_velocities.shape[0], train_images,
                                                      train_velocities, 'train')

                # save the training loss using the manual summaries
                man_loss_summary.value[0].simple_value = avg_train_loss
                train_writer.add_summary(man_loss_summary, epoch)

                # run test cycle
                avg_test_loss = self.epoch_iteration(test_velocities.shape[0], test_images, test_velocities, 'test')

                # save the test errors using the manual summaries
                man_loss_summary.value[0].simple_value = avg_test_loss
                test_writer.add_summary(man_loss_summary, epoch)

                # print train and test loss to monitor progress during training every 50 epochs
                if epoch % 50 == 0:
                    print("Epoch: {:04d} , train_loss = {:.6f} , "
                          "test_loss = {:.6f}".format(epoch+1, avg_train_loss, avg_test_loss))

                # save weights every 100 epochs
                if epoch % 100 == 0:
                    saver.save(self.sess, logs_train_path, epoch)

        # close summary writer
        train_writer.close()
        test_writer.close()
