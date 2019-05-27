import tensorflow as tf
import numpy as np


class cdac(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size, vocab_entity_size, vocab_pos_size, sequence_char_length, num_quantized_chars,
    embedding_size, embedding_entity_size, embedding_char_size, embedding_pos_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

            # Placeholders for input, output and dropout
            # self.input_x_entity = tf.placeholder(tf.int32, [None, sequence_length], name="input_entityvector")

            #entity input
            self.input_x_hand = tf.placeholder(tf.float32, [None, 1], name="input_x_hand")

            # intent binary input
            self.input_ib1 = tf.placeholder(tf.float32, [None, 43], name="input_ib1")
            self.input_ib2 = tf.placeholder(tf.float32, [None, 43], name="input_ib2")

            #suggested topic input
            self.input_char = tf.placeholder(tf.int32, [None, sequence_char_length], name="input_char0")
            self.input_char1 = tf.placeholder(tf.int32, [None, sequence_char_length], name="input_char1")
            self.input_char2 = tf.placeholder(tf.int32, [None, sequence_char_length], name="input_char2")

            # class topic input
            self.input_pos0 = tf.placeholder(tf.int32, [None, sequence_length], name="input_pos0")
            self.input_pos1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_pos1")
            self.input_pos2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_pos2")
            self.input_pos3 = tf.placeholder(tf.int32, [None, sequence_length], name="input_pos3")

            # class yesno intent input
            self.input_spId0 = tf.placeholder(tf.float32, [None, 609], name="input_spi0")
            self.input_spId1 = tf.placeholder(tf.float32, [None, 609], name="input_spi1")
            self.input_spId2 = tf.placeholder(tf.float32, [None, 609], name="input_spi2")
            self.input_spId3 = tf.placeholder(tf.float32, [None, 609], name="input_spi3")

            # class yesno intent input
            self.input_hub0 = tf.placeholder(tf.float32, [None, 128], name="input_hub0")
            self.input_hub1 = tf.placeholder(tf.float32, [None, 128], name="input_hub1")
            self.input_hub2 = tf.placeholder(tf.float32, [None, 128], name="input_hub2")

            # class yesno intent input
            self.input_mtp0 = tf.placeholder(tf.float32, [None, 66], name="input_mtp0")
            self.input_mtp1 = tf.placeholder(tf.float32, [None, 66], name="input_mtp1")
            self.input_mtp2 = tf.placeholder(tf.float32, [None, 66], name="input_mtp2")

            # class yesno intent input
            self.input_feat0 = tf.placeholder(tf.float32, [None, 5], name="input_feat0")
            self.input_feat1 = tf.placeholder(tf.float32, [None, 5], name="input_feat1")
            self.input_feat2 = tf.placeholder(tf.float32, [None, 5], name="input_feat2")
            self.input_feat3 = tf.placeholder(tf.float32, [None, 5], name="input_feat3")

            # utterance input
            self.input_utt1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
            self.input_utt2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
            self.input_utt3 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x3")
            self.input_utt_pred = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_pred")


            # outputs
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(l2_reg_lambda)


            ####################################################################################
            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("emb_utt_pred"):
                self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W_pred", trainable= True)

                self.place_w = tf.placeholder(tf.float32, shape=(vocab_size, embedding_size))
                self.set_W = tf.assign(self.W, self.place_w, validate_shape=False)

                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_utt_pred)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            ####################################################################################

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool_pred_-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_pred")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_pred")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="convuttpred")
                    # Apply nonlinearity

                    conv = tf.layers.batch_normalization(inputs=conv, momentum=0.997, epsilon=1e-5,center=True, scale=True)

                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="reluuttpred")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pooluttpred")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat_utt_pred = tf.reshape(self.h_pool, [-1, num_filters_total])

            ####################################################################################

            # with tf.device('/cpu:0'), tf.name_scope("embeddingc"):
            #     self.Wc = tf.Variable(tf.random_uniform([vocab_entity_size, embedding_entity_size], -1.0, 1.0), name="WW", trainable= True)
            #     self.place_wc = tf.placeholder(tf.float32, shape=(vocab_entity_size, embedding_entity_size))
            #     self.set_Wc = tf.assign(self.Wc, self.place_wc, validate_shape=False)
            #
            #     self.embedded_chars_c = tf.nn.embedding_lookup(self.Wc, self.input_x_entity)
            #     self.embedded_chars_expanded_c = tf.expand_dims(self.embedded_chars_c, -1)
            #
            # ####################################################################################
            # # Create a convolution + maxpool layer for each filter size
            # pooled_outputs_c = []
            # for i, filter_size in enumerate(filter_sizes):
            #     with tf.name_scope("convc-maxpoolc-%s" % filter_size):
            #         # Convolution Layer
            #         filter_shape = [filter_size, embedding_size, 1, num_filters]
            #         W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="Wc")
            #         b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="bc")
            #         conv = tf.nn.conv2d(
            #             self.embedded_chars_expanded_c,
            #             W,
            #             strides=[1, 1, 1, 1],
            #             padding="VALID",
            #             name="conv")
            #         conv = tf.layers.batch_normalization(inputs=conv, momentum=0.997, epsilon=1e-5, center=True, scale=True)
            #         # Apply nonlinearity
            #         h = tf.nn.relu(tf.nn.bias_add(conv, b), name="reluc")
            #         # Maxpooling over the outputs
            #         pooled = tf.nn.max_pool(
            #             h,
            #             ksize=[1, sequence_length - filter_size + 1, 1, 1],
            #             strides=[1, 1, 1, 1],
            #             padding='VALID',
            #             name="pool")
            #         pooled_outputs_c.append(pooled)
            #
            # # Combine all the pooled features
            # num_filters_total = num_filters * len(filter_sizes)
            # self.h_pool_c = tf.concat(pooled_outputs_c, 3)
            # self.h_pool_flat_c = tf.reshape(self.h_pool_c, [-1, num_filters_total])

            ##################################################################################
            ##################################################################################

            with tf.device('/cpu:0'), tf.name_scope("emb_utt-2"):
                self.W_utt_2 = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W_utt2", trainable=True)
                self.place_w_utt_2 = tf.placeholder(tf.float32, shape=(vocab_size, embedding_size))
                self.set_W_utt_2 = tf.assign(self.W_utt_2, self.place_w_utt_2, validate_shape=False)

                self.embedded_chars_utt_2 = tf.nn.embedding_lookup(self.W_utt_2, self.input_utt2)
                self.embedded_chars_expanded_utt_2 = tf.expand_dims(self.embedded_chars_utt_2, -1)

            ####################################################################################
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs_utt_2 = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("convc-maxpoolc_utt2_-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="Wutt2")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="butt2")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded_utt_2,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="convutt2")

                    conv = tf.layers.batch_normalization(inputs=conv, momentum=0.997, epsilon=1e-5, center=True, scale=True)

                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="reluutt2")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="poolutt2")
                    pooled_outputs_utt_2.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool_utt_2 = tf.concat(pooled_outputs_utt_2, 3)
            self.h_pool_flat_utt_2 = tf.reshape(self.h_pool_utt_2, [-1, num_filters_total])
            #
            # # # # ####################################################################################
            # # # # ####################################################################################
            # # # #
            with tf.device('/cpu:0'), tf.name_scope("emb_utt-1"):
                self.W_utt_1 = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W_utt1", trainable=True)
                self.place_w_utt_1 = tf.placeholder(tf.float32, shape=(vocab_size, embedding_size))
                self.set_W_utt_1 = tf.assign(self.W_utt_1, self.place_w_utt_1, validate_shape=False)

                self.embedded_chars_utt_1 = tf.nn.embedding_lookup(self.W_utt_1, self.input_utt1)
                self.embedded_chars_expanded_utt_1 = tf.expand_dims(self.embedded_chars_utt_1, -1)

            ####################################################################################

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs_utt_1 = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("convc-maxpoolc_utt1_%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="Wutt1")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="butt1")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded_utt_1,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="convutt1")
                    conv = tf.layers.batch_normalization(inputs=conv, momentum=0.997, epsilon=1e-5, center=True, scale=True)

                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="reluutt1")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="poolutt1")
                    pooled_outputs_utt_1.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool_utt_1 = tf.concat(pooled_outputs_utt_1, 3)
            self.h_pool_flat_utt_1 = tf.reshape(self.h_pool_utt_1, [-1, num_filters_total])

            ####################################################################################
            ####################################################################################
            with tf.device('/cpu:0'), tf.name_scope("emb_utt-3"):
                self.W_utt_3 = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W_utt3",
                                           trainable=True)
                self.place_w_utt_3 = tf.placeholder(tf.float32, shape=(vocab_size, embedding_size))
                self.set_W_utt_3 = tf.assign(self.W_utt_3, self.place_w_utt_3, validate_shape=False)

                self.embedded_chars_utt_3 = tf.nn.embedding_lookup(self.W_utt_3, self.input_utt3)
                self.embedded_chars_expanded_utt_3 = tf.expand_dims(self.embedded_chars_utt_3, -1)

                ####################################################################################

                # Create a convolution + maxpool layer for each filter size
            pooled_outputs_utt_3 = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("convc-maxpoolc_utt3_%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="Wutt3")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="butt3")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded_utt_3,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="convutt3")
                    conv = tf.layers.batch_normalization(inputs=conv, momentum=0.997, epsilon=1e-5, center=True,
                                                         scale=True)

                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="reluutt3")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="poolutt3")
                    pooled_outputs_utt_3.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool_utt_3 = tf.concat(pooled_outputs_utt_3, 3)
            self.h_pool_flat_utt_3 = tf.reshape(self.h_pool_utt_3, [-1, num_filters_total])

            ####################################################################################
            ####################################################################################

            with tf.device('/cpu:0'), tf.name_scope("emb_pos"):
                self.W_pos0 = tf.Variable(tf.random_uniform([vocab_pos_size, embedding_pos_size], -1.0, 1.0), name="WposW", trainable=True)
                self.place_w_pos0 = tf.placeholder(tf.float32, shape=(vocab_pos_size, embedding_pos_size))
                self.set_W_pos0 = tf.assign(self.W_pos0, self.place_w_pos0, validate_shape=False)

                self.embedded_chars_pos0 = tf.nn.embedding_lookup(self.W_pos0, self.input_pos0)
                self.embedded_chars_expanded_pos0 = tf.expand_dims(self.embedded_chars_pos0, -1)

            ###################################################################################

            ## Create a convolution + maxpool layer for each filter size
            pooled_outputs_pos = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("convc-maxpool_pos-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_pos_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="Wpos")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="bpos")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded_pos0,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="convpos")

                    conv = tf.layers.batch_normalization(inputs=conv, momentum=0.997, epsilon=1e-5, center=True, scale=True)

                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relupos")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="poolpos")
                    pooled_outputs_pos.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool_pos0 = tf.concat(pooled_outputs_pos, 3)
            self.h_pool_flat_pos0 = tf.reshape(self.h_pool_pos0, [-1, num_filters_total])

            # # ####################################################################################
            # # ####################################################################################
            #
            #
            with tf.device('/cpu:0'), tf.name_scope("emb_pos_utt1"):
                self.W_pos1 = tf.Variable(tf.random_uniform([vocab_pos_size, embedding_pos_size], -1.0, 1.0),
                                          name="Wposutt1", trainable=True)
                self.place_w_pos1 = tf.placeholder(tf.float32, shape=(vocab_pos_size, embedding_pos_size))
                self.set_W_pos1 = tf.assign(self.W_pos1, self.place_w_pos1, validate_shape=False)

                self.embedded_chars_pos1 = tf.nn.embedding_lookup(self.W_pos1, self.input_pos1)
                self.embedded_chars_expanded_pos1 = tf.expand_dims(self.embedded_chars_pos1, -1)

            # ####################################################################################
            # #
            # # Create a convolution + maxpool layer for each filter size
            pooled_outputs_pos_utt1 = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("convc-maxpool_pos_utt1-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_pos_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="Wposutt1")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="bposutt1")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded_pos1,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="convposutt1")
                    conv = tf.layers.batch_normalization(inputs=conv, momentum=0.997, epsilon=1e-5, center=True, scale=True)

                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="reluposutt1")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="poolposutt1")
                    pooled_outputs_pos_utt1.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool_pos1 = tf.concat(pooled_outputs_pos_utt1, 3)
            self.h_pool_flat_pos1 = tf.reshape(self.h_pool_pos1, [-1, num_filters_total])
            # ####################################################################################
            # ####################################################################################


            with tf.device('/cpu:0'), tf.name_scope("emb_pos_utt2"):
                self.W_pos2 = tf.Variable(tf.random_uniform([vocab_pos_size, embedding_pos_size], -1.0, 1.0),
                                          name="Wposutt2", trainable=True)
                self.place_w_pos2 = tf.placeholder(tf.float32, shape=(vocab_pos_size, embedding_pos_size))
                self.set_W_pos2 = tf.assign(self.W_pos2, self.place_w_pos2, validate_shape=False)

                self.embedded_chars_pos2 = tf.nn.embedding_lookup(self.W_pos2, self.input_pos2)
                self.embedded_chars_expanded_pos2 = tf.expand_dims(self.embedded_chars_pos2, -1)
            #
            # ####################################################################################
            # #
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs_pos_utt2 = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("convc-maxpool_pos_utt2-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_pos_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="Wposutt2")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="bposutt2")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded_pos2,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="convposutt2")
                    conv = tf.layers.batch_normalization(inputs=conv, momentum=0.997, epsilon=1e-5, center=True, scale=True)

                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="reluposutt2")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="poolposutt2")
                    pooled_outputs_pos_utt2.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool_pos2 = tf.concat(pooled_outputs_pos_utt2, 3)
            self.h_pool_flat_pos2 = tf.reshape(self.h_pool_pos2, [-1, num_filters_total])
            ####################################################################################
            ####################################################################################

            with tf.device('/cpu:0'), tf.name_scope("emb_pos_utt3"):
                self.W_pos3 = tf.Variable(tf.random_uniform([vocab_pos_size, embedding_pos_size], -1.0, 1.0),
                                          name="Wposutt3", trainable=True)
                self.place_w_pos3 = tf.placeholder(tf.float32, shape=(vocab_pos_size, embedding_pos_size))
                self.set_W_pos3 = tf.assign(self.W_pos3, self.place_w_pos3, validate_shape=False)

                self.embedded_chars_pos3 = tf.nn.embedding_lookup(self.W_pos3, self.input_pos3)
                self.embedded_chars_expanded_pos3 = tf.expand_dims(self.embedded_chars_pos3, -1)
                #
                # ####################################################################################
                # #
                # Create a convolution + maxpool layer for each filter size
            pooled_outputs_pos_utt3 = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("convc-maxpool_pos_utt2-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_pos_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="Wposutt3")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="bposutt3")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded_pos3,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="convposutt3")
                    conv = tf.layers.batch_normalization(inputs=conv, momentum=0.997, epsilon=1e-5, center=True,
                                                         scale=True)

                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="reluposutt3")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="poolposutt2")
                    pooled_outputs_pos_utt3.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool_pos3 = tf.concat(pooled_outputs_pos_utt3, 3)
            self.h_pool_flat_pos3 = tf.reshape(self.h_pool_pos3, [-1, num_filters_total])
            ####################################################################################
            ####################################################################################



            # with tf.device('/cpu:0'), tf.name_scope("emb_char"):
            #     # with tf.variable_scope('emb_char0') as scope:
            #         self.W_char0 = tf.Variable(tf.random_uniform([num_quantized_chars, embedding_char_size], -1.0, 1.0), name="Wchar0", trainable=True)
            #         # self.place_w_char0 = tf.placeholder(tf.float32, shape=(num_quantized_chars, embedding_char_size))
            #         # self.set_W_char0 = tf.assign(self.W_char0, self.place_w_char0, validate_shape=False)
            #
            #         self.embedded_chars_char0 = tf.nn.embedding_lookup(self.W_char0, self.input_char)
            #         self.embedded_chars_expanded_char0 = tf.expand_dims(self.embedded_chars_char0, -1)
            #
            # ###################################################################################
            #
            # # Create a convolution + maxpool layer for each filter size
            # pooled_outputs_char0 = []
            # for i, filter_size in enumerate(filter_sizes):
            #     with tf.name_scope("convc-maxpool_char0-%s" % filter_size):
            #         # with tf.variable_scope('conv_char0') as scope:
            #         # Convolution Layer
            #             filter_shape = [filter_size, embedding_char_size, 1, num_filters]
            #             W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="Wchar0")
            #             b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="bchar0")
            #             conv = tf.nn.conv2d(
            #                 self.embedded_chars_expanded_char0,
            #                 W,
            #                 strides=[1, 1, 1, 1],
            #                 padding="VALID",
            #                 name="convchar0")
            #             conv = tf.layers.batch_normalization(inputs=conv, momentum=0.997, epsilon=1e-5, center=True, scale=True)
            #
            #         # Apply nonlinearity
            #             h = tf.nn.relu(tf.nn.bias_add(conv, b), name="reluchar0")
            #             # Maxpooling over the outputs
            #             pooled = tf.nn.max_pool(
            #                 h,
            #                 ksize=[1, sequence_char_length - filter_size + 1, 1, 1],
            #                 strides=[1, 1, 1, 1],
            #                 padding='VALID',
            #                 name="poolchar0")
            #             pooled_outputs_char0.append(pooled)
            #
            # # Combine all the pooled features
            # num_filters_total = num_filters * len(filter_sizes)
            # self.h_pool_char0 = tf.concat(pooled_outputs_char0, 3)
            # self.h_pool_flat_char0 = tf.reshape(self.h_pool_char0, [-1, num_filters_total])
            #
            #

            # ###################################################################################
            # ###################################################################################
            #
            # with tf.device('/cpu:0'), tf.name_scope("emb_char1"):
            #     # with tf.variable_scope('emb_char0') as scope:
            #     self.W_char1 = tf.Variable(tf.random_uniform([num_quantized_chars, embedding_char_size], -1.0, 1.0),
            #                                name="Wchar1", trainable=True)
            #     # self.place_w_char0 = tf.placeholder(tf.float32, shape=(num_quantized_chars, embedding_char_size))
            #     # self.set_W_char0 = tf.assign(self.W_char0, self.place_w_char0, validate_shape=False)
            #
            #     self.embedded_chars_char1 = tf.nn.embedding_lookup(self.W_char1, self.input_char)
            #     self.embedded_chars_expanded_char1 = tf.expand_dims(self.embedded_chars_char1, -1)
            #
            #     ####################################################################################
            #
            # # Create a convolution + maxpool layer for each filter size
            # pooled_outputs_char1 = []
            # for i, filter_size in enumerate(filter_sizes):
            #     with tf.name_scope("convc-maxpool_char1-%s" % filter_size):
            #         # with tf.variable_scope('conv_char0') as scope:
            #         # Convolution Layer
            #             filter_shape = [filter_size, embedding_char_size, 1, num_filters]
            #             W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="Wchar1")
            #             b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="bchar1")
            #             conv = tf.nn.conv2d(
            #                 self.embedded_chars_expanded_char1,
            #                 W,
            #                 strides=[1, 1, 1, 1],
            #                 padding="VALID",
            #                 name="convchar1")
            #        conv = tf.layers.batch_normalization(inputs=conv, momentum=0.997, epsilon=1e-5, center=True, scale=True)

            #             # Apply nonlinearity
            #             h = tf.nn.relu(tf.nn.bias_add(conv, b), name="reluchar1")
            #             # Maxpooling over the outputs
            #             pooled = tf.nn.max_pool(
            #                 h,
            #                 ksize=[1, sequence_char_length - filter_size + 1, 1, 1],
            #                 strides=[1, 1, 1, 1],
            #                 padding='VALID',
            #                 name="poolcharo")
            #             pooled_outputs_char1.append(pooled)
            #
            # # Combine all the pooled features
            # num_filters_total = num_filters * len(filter_sizes)
            # self.h_pool_char1 = tf.concat(pooled_outputs_char1, 3)
            # self.h_pool_flat_char1 = tf.reshape(self.h_pool_char1, [-1, num_filters_total])
            #
            #
            #
            # ###################################################################################
            # ###################################################################################
            #
            # with tf.device('/cpu:0'), tf.name_scope("emb_char2"):
            #     # with tf.variable_scope('emb_char0') as scope:
            #     self.W_char2 = tf.Variable(tf.random_uniform([num_quantized_chars, embedding_char_size], -1.0, 1.0),
            #                                name="Wchar2", trainable=True)
            #     # self.place_w_char0 = tf.placeholder(tf.float32, shape=(num_quantized_chars, embedding_char_size))
            #     # self.set_W_char0 = tf.assign(self.W_char0, self.place_w_char0, validate_shape=False)
            #
            #     self.embedded_chars_char2 = tf.nn.embedding_lookup(self.W_char2, self.input_char)
            #     self.embedded_chars_expanded_char2 = tf.expand_dims(self.embedded_chars_char2, -1)
            #
            #     ####################################################################################
            #
            # # Create a convolution + maxpool layer for each filter size
            # pooled_outputs_char2 = []
            # for i, filter_size in enumerate(filter_sizes):
            #     with tf.name_scope("convc-maxpool_char2-%s" % filter_size):
            #         # with tf.variable_scope('conv_char0') as scope:
            #         # Convolution Layer
            #             filter_shape = [filter_size, embedding_char_size, 1, num_filters]
            #             W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="Wchar2")
            #             b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="bchar2")
            #             conv = tf.nn.conv2d(
            #                 self.embedded_chars_expanded_char2,
            #                 W,
            #                 strides=[1, 1, 1, 1],
            #                 padding="VALID",
            #                 name="convchar2")
            #        conv = tf.layers.batch_normalization(inputs=conv, momentum=0.997, epsilon=1e-5, center=True, scale=True)

            #             # Apply nonlinearity
            #             h = tf.nn.relu(tf.nn.bias_add(conv, b), name="reluchar2")
            #             # Maxpooling over the outputs
            #             pooled = tf.nn.max_pool(
            #                 h,
            #                 ksize=[1, sequence_char_length - filter_size + 1, 1, 1],
            #                 strides=[1, 1, 1, 1],
            #                 padding='VALID',
            #                 name="poolcharo")
            #             pooled_outputs_char2.append(pooled)
            #
            # # Combine all the pooled features
            # num_filters_total = num_filters * len(filter_sizes)
            # self.h_pool_char2 = tf.concat(pooled_outputs_char2, 3)
            # self.h_pool_flat_char2 = tf.reshape(self.h_pool_char2, [-1, num_filters_total])

            ####################################################################################
            ####################################################################################


            # text_length = self._length(self.input_char)
            # # Embedding Lookup 16
            # with tf.name_scope("embedding_char"):
            #     self.embedding_char_W = tf.get_variable(name='lookup_W',
            #                                             shape=[num_quantized_chars, embedding_char_size],
            #                                             initializer=tf.keras.initializers.he_uniform(), trainable=True)
            #     self.embedded_characters = tf.nn.embedding_lookup(self.embedding_char_W, self.input_char)
            #
            # # Create a convolution + maxpool layer for each filter size
            # with tf.name_scope("rnn_char"):
            #     with tf.variable_scope('rnn_chr') as scope:
            #         cell = self._get_cell(sequence_char_length, 'gru')
            #         cell_with_attention = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
            #         # cell_with_attention = tf.contrib.rnn.AttentionCellWrapper(cell_with_attention, 25)
            #         outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_with_attention,
            #                                                                  cell_bw=cell_with_attention,
            #                                                                  inputs=self.embedded_characters,
            #                                                                  sequence_length=text_length,
            #                                                                  dtype=tf.float32)
            #
            #         output_fw, output_bw = outputs
            #         all_outputs = tf.concat([output_fw, output_bw], 2)
            #         self.h_pool_flat_char = self.last_relevant(all_outputs, sequence_char_length)

            # ####################################################################################
            # ####################################################################################
            #
            #
            # text_length = self._length(self.input_char1)
            # # Embedding Lookup 16
            # with tf.name_scope("embedding_char1"):
            #     self.embedding_char_W = tf.get_variable(name='lookup_W1',
            #                                             shape=[num_quantized_chars, embedding_char_size],
            #                                             initializer=tf.keras.initializers.he_uniform(), trainable=True)
            #     self.embedded_characters = tf.nn.embedding_lookup(self.embedding_char_W, self.input_char1)
            #
            # # Create a convolution + maxpool layer for each filter size
            # with tf.name_scope("rnn_char1"):
            #     with tf.variable_scope('rnn_chr1') as scope:
            #         cell = self._get_cell(sequence_char_length, 'gru')
            #         cell_with_attention = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
            #         # cell_with_attention = tf.contrib.rnn.AttentionCellWrapper(cell_with_attention, 10)
            #         outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_with_attention,
            #                                                                  cell_bw=cell_with_attention,
            #                                                                  inputs=self.embedded_characters,
            #                                                                  sequence_length=text_length,
            #                                                                  dtype=tf.float32)
            #
            #         output_fw, output_bw = outputs
            #         all_outputs = tf.concat([output_fw, output_bw], 2)
            #         self.h_pool_flat_char1 = self.last_relevant(all_outputs, sequence_char_length)

            # ####################################################################################
            # ####################################################################################
            #
            #
            # text_length = self._length(self.input_char2)
            # # Embedding Lookup 16
            # with tf.name_scope("embedding_char2"):
            #     self.embedding_char_W = tf.get_variable(name='lookup_W2',
            #                                             shape=[num_quantized_chars, embedding_char_size],
            #                                             initializer=tf.keras.initializers.he_uniform(), trainable=True)
            #     self.embedded_characters = tf.nn.embedding_lookup(self.embedding_char_W, self.input_char2)
            #
            # # Create a convolution + maxpool layer for each filter size
            # with tf.name_scope("rnn_char2"):
            #     cell2 = self._get_cell(sequence_char_length, 'vanilla')
            #     cell_with_attention = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=self.dropout_keep_prob)
            #     # cell_with_attention = tf.contrib.rnn.AttentionCellWrapper(cell_with_attention, 10)
            #     outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_with_attention,
            #                                                              cell_bw=cell_with_attention,
            #                                                              inputs=self.embedded_characters,
            #                                                              sequence_length=text_length,
            #                                                              dtype=tf.float32)
            #
            #     output_fw, output_bw = outputs
            #     all_outputs = tf.concat([output_fw, output_bw], 2)
            #     self.h_pool_flat_char2 = self.last_relevant(all_outputs, sequence_char_length)

            # ####################################################################################
            # ####################################################################################
            #

            # text_length = self._length(self.input_cl0)
            # # Embedding Lookup 16
            # with tf.name_scope("embedding_pos0"):
            #     self.embedding_char_W = tf.get_variable(name='lookup_posW2',
            #                                             shape=[vocab_pos_size, embedding_pos_size],
            #                                             initializer=tf.keras.initializers.he_uniform(), trainable=True)
            #     self.embedded_pos0 = tf.nn.embedding_lookup(self.embedding_char_W, self.input_cl0)
            #
            # # Create a convolution + maxpool layer for each filter size
            # with tf.name_scope("rnn_pos0"):
            #     with tf.variable_scope('rnn_pos0') as scope:
            #         cell2 = self._get_cell(sequence_length, 'gru')
            #         cell_with_attention = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=self.dropout_keep_prob)
            #         # cell_with_attention = tf.contrib.rnn.AttentionCellWrapper(cell_with_attention, 10)
            #         outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_with_attention,
            #                                                                  cell_bw=cell_with_attention,
            #                                                                  inputs=self.embedded_pos0,
            #                                                                  sequence_length=text_length,
            #                                                                  dtype=tf.float32)
            #
            #         output_fw, output_bw = outputs
            #         all_outputs = tf.concat([output_fw, output_bw], 2)
            #         self.h_pool_flat_pos0 = self.last_relevant(all_outputs, sequence_length)


            ####################################################################################
            ####################################################################################

            # with tf.name_scope("NN"):
            #     with tf.variable_scope('FCNN') as scope:
            #
            #         W = tf.get_variable(
            #             "W",
            #             shape=[self.h_pool_flat_utt_pred.get_shape()[1], 100],
            #             initializer=tf.contrib.layers.xavier_initializer())
            #         b = tf.Variable(tf.constant(0.1, shape=[100]), name="b")
            #         l2_loss += tf.nn.l2_loss(W)
            #         l2_loss += tf.nn.l2_loss(b)
            #         self.dense = tf.nn.xw_plus_b(self.h_pool_flat_utt_pred, W, b, name="scores")

            # ####################################################################################
            # ####################################################################################

            self.h_pool2_utt3_char3 = tf.concat([self.h_pool_flat_utt_3, self.h_pool_flat_pos3, self.input_feat3, self.input_spId3], 1)

            with tf.name_scope("NN_utt3"):
                with tf.variable_scope('FCNN_utt3') as scope:
                    W = tf.get_variable(
                        "W_utt3",
                        shape=[self.h_pool2_utt3_char3.get_shape()[1], 100],
                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.constant(0.1, shape=[100]), name="b_utt3")
                    l2_loss += tf.nn.l2_loss(W)
                    l2_loss += tf.nn.l2_loss(b)
                    self.dense_utt3 = tf.nn.xw_plus_b(self.h_pool2_utt3_char3, W, b, name="scores_utt2")

            with tf.name_scope("dropout_utt3"):
                self.h_drop_utt3 = tf.nn.dropout(self.dense_utt3, self.dropout_keep_prob)

                # Final (unnormalized) scores and predictions
            with tf.name_scope("output_utt3"):
                W = tf.get_variable(
                    "W_utt3",
                    shape=[self.dense_utt3.get_shape()[1], num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_utt2")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores_utt3 = tf.nn.xw_plus_b(self.h_drop_utt3, W, b, name="scores_utt2")

            # ####################################################################################
            # ####################################################################################

            self.h_pool2_utt2_char2 = tf.concat([self.h_pool_flat_utt_2, self.h_pool_flat_pos2, self.input_feat2, self.input_spId2, self.scores_utt3], 1)

            with tf.name_scope("NN_utt2"):
                with tf.variable_scope('FCNN_utt2') as scope:

                    W = tf.get_variable(
                        "W_utt2",
                        shape=[self.h_pool2_utt2_char2.get_shape()[1], 100],
                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.constant(0.1, shape=[100]), name="b_utt2")
                    l2_loss += tf.nn.l2_loss(W)
                    l2_loss += tf.nn.l2_loss(b)
                    self.dense_utt2 = tf.nn.xw_plus_b(self.h_pool2_utt2_char2, W, b, name="scores_utt2")

            with tf.name_scope("dropout_utt2"):
                self.h_drop_utt2 = tf.nn.dropout(self.dense_utt2, self.dropout_keep_prob)

                # Final (unnormalized) scores and predictions
            with tf.name_scope("output_utt2"):
                W = tf.get_variable(
                    "W_utt2",
                    shape=[self.dense_utt2.get_shape()[1], num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_utt2")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores_utt2 = tf.nn.xw_plus_b(self.h_drop_utt2, W, b, name="scores_utt2")

            ###################################################################################
            ###################################################################################

            self.h_pool2_utt1_char1 = tf.concat([self.h_pool_flat_utt_1, self.h_pool_flat_pos1, self.input_feat1, self.input_spId1, self.scores_utt3, self.scores_utt2], 1)

            with tf.name_scope("NN_utt1"):
                with tf.variable_scope('FCNN_utt1') as scope:
                    W = tf.get_variable(
                        "W_utt1",
                        shape=[self.h_pool2_utt1_char1.get_shape()[1], 100],
                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.constant(0.1, shape=[100]), name="b_utt1")
                    l2_loss += tf.nn.l2_loss(W)
                    l2_loss += tf.nn.l2_loss(b)
                    self.dense_utt1 = tf.nn.xw_plus_b(self.h_pool2_utt1_char1, W, b, name="scores_utt1")

            with tf.name_scope("dropout_utt1"):
                self.h_drop_utt1 = tf.nn.dropout(self.dense_utt1, self.dropout_keep_prob)

                # Final (unnormalized) scores and predictions
            with tf.name_scope("output_utt1"):
                W = tf.get_variable(
                    "W_utt1",
                    shape=[self.dense_utt1.get_shape()[1], num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_utt1")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores_utt1 = tf.nn.xw_plus_b(self.h_drop_utt1, W, b, name="scores_utt1")

            # self.h_pool2_utt2_char3 = tf.concat([self.h_pool_flat_utt_3], 1)
            #
            # with tf.name_scope("NN_utt3"):
            #     with tf.variable_scope('FCNN_utt3') as scope:
            #         W = tf.get_variable(
            #             "W_utt3",
            #             shape=[self.h_pool2_utt2_char3.get_shape()[1], 100],
            #             initializer=tf.contrib.layers.xavier_initializer())
            #         b = tf.Variable(tf.constant(0.1, shape=[100]), name="b_utt3")
            #         l2_loss += tf.nn.l2_loss(W)
            #         l2_loss += tf.nn.l2_loss(b)
            #         self.dense_utt3 = tf.nn.xw_plus_b(self.h_pool2_utt2_char3, W, b, name="scores_utt3")
            #
            # with tf.name_scope("dropout_utt3"):
            #     self.h_drop_utt3 = tf.nn.dropout(self.dense_utt3, self.dropout_keep_prob)
            #
            #     # Final (unnormalized) scores and predictions
            # with tf.name_scope("output_utt3"):
            #     W = tf.get_variable(
            #         "W_utt3",
            #         shape=[self.dense_utt3.get_shape()[1], num_classes],
            #         initializer=tf.contrib.layers.xavier_initializer())
            #     b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_utt3")
            #     l2_loss += tf.nn.l2_loss(W)
            #     l2_loss += tf.nn.l2_loss(b)
            #     self.scores_utt3 = tf.nn.xw_plus_b(self.h_drop_utt3, W, b, name="scores_utt3")

            ###################################################################################
            ###################################################################################
            # with tf.name_scope("NN_utt_char0"):
            #     with tf.variable_scope('FCNN_utt_char0') as scope:
            #
            #         W = tf.get_variable(
            #             "W_utt_char0",
            #             shape=[self.h_pool_flat_char0.get_shape()[1], 50],
            #             initializer=tf.contrib.layers.xavier_initializer())
            #         b = tf.Variable(tf.constant(0.1, shape=[50]), name="b_utt_char0")
            #         l2_loss += tf.nn.l2_loss(W)
            #         l2_loss += tf.nn.l2_loss(b)
            #         self.dense_utt_char0 = tf.nn.xw_plus_b(self.h_pool_flat_char0, W, b, name="scores_utt_char0")
            #
            # with tf.name_scope("dropout_utt_char0"):
            #     self.h_drop_utt_char0 = tf.nn.dropout(self.dense_utt_char0, self.dropout_keep_prob)
            #
            #     # Final (unnormalized) scores and predictions
            # with tf.name_scope("output_utt_char0"):
            #     W = tf.get_variable(
            #         "W_utt_char0",
            #         shape=[self.dense_utt_char0.get_shape()[1], num_classes],
            #         initializer=tf.contrib.layers.xavier_initializer())
            #     b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_utt_char0")
            #     l2_loss += tf.nn.l2_loss(W)
            #     l2_loss += tf.nn.l2_loss(b)
            #     self.scores_utt_char0 = tf.nn.xw_plus_b(self.h_drop_utt_char0, W, b, name="scores_utt_char0")
            #
            #
            # # ####################################################################################
            # # ####################################################################################
            # with tf.name_scope("NN_utt_char1"):
            #     with tf.variable_scope('FCNN_utt_char1') as scope:
            #
            #         W = tf.get_variable(
            #             "W_utt_char1",
            #             shape=[self.h_pool_flat_char1.get_shape()[1], 50],
            #             initializer=tf.contrib.layers.xavier_initializer())
            #         b = tf.Variable(tf.constant(0.1, shape=[50]), name="b_utt_char1")
            #         l2_loss += tf.nn.l2_loss(W)
            #         l2_loss += tf.nn.l2_loss(b)
            #         self.dense_utt_char1 = tf.nn.xw_plus_b(self.h_pool_flat_char1, W, b, name="scores_utt_char0")
            #
            # with tf.name_scope("dropout_utt_char1"):
            #     self.h_drop_utt_char1 = tf.nn.dropout(self.dense_utt_char1, self.dropout_keep_prob)
            #
            #     # Final (unnormalized) scores and predictions
            # with tf.name_scope("output_utt_char1"):
            #     W = tf.get_variable(
            #         "W_utt_char1",
            #         shape=[self.dense_utt_char1.get_shape()[1], num_classes],
            #         initializer=tf.contrib.layers.xavier_initializer())
            #     b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_utt_char1")
            #     l2_loss += tf.nn.l2_loss(W)
            #     l2_loss += tf.nn.l2_loss(b)
            #     self.scores_utt_char1 = tf.nn.xw_plus_b(self.h_drop_utt_char1, W, b, name="scores_utt_char1")
            #
            # # ####################################################################################
            # # ####################################################################################
            #
            # with tf.name_scope("NN_utt_char2"):
            #     with tf.variable_scope('FCNN_utt_char2') as scope:
            #
            #         W = tf.get_variable(
            #             "W_utt_char2",
            #             shape=[self.h_pool_flat_char2.get_shape()[1], 50],
            #             initializer=tf.contrib.layers.xavier_initializer())
            #         b = tf.Variable(tf.constant(0.1, shape=[50]), name="b_utt_char2")
            #         l2_loss += tf.nn.l2_loss(W)
            #         l2_loss += tf.nn.l2_loss(b)
            #         self.dense_utt_char2 = tf.nn.xw_plus_b(self.h_pool_flat_char2, W, b, name="scores_utt_char2")
            #
            # with tf.name_scope("dropout_utt_char2"):
            #     self.h_drop_utt_char2 = tf.nn.dropout(self.dense_utt_char2, self.dropout_keep_prob)
            #
            #     # Final (unnormalized) scores and predictions
            # with tf.name_scope("output_utt_char2"):
            #     W = tf.get_variable(
            #         "W_utt_char2",
            #         shape=[self.dense_utt_char2.get_shape()[1], num_classes],
            #         initializer=tf.contrib.layers.xavier_initializer())
            #     b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_utt_char2")
            #     l2_loss += tf.nn.l2_loss(W)
            #     l2_loss += tf.nn.l2_loss(b)
            #     self.scores_utt_char2 = tf.nn.xw_plus_b(self.h_drop_utt_char2, W, b, name="scores_utt_char2")

            # ####################################################################################
            # ####################################################################################

            # Concatination of both features to a Tensor
            # self.h_pool2_flat_pred_ent = tf.concat([self.h_pool_flat_utt_pred, self.h_pool_flat_c], 1)

            # Concatination of both features to a Tensor
            self.h_pool2_flat = tf.concat([self.h_pool_flat_utt_pred, self.input_feat0, self.h_pool_flat_pos0, self.input_spId0, self.scores_utt1 , self.scores_utt2, self.scores_utt3,], 1)
            # Concatination of both features to a Tensor
            # self.h_pool2_flat = tf.concat([self.h_pool2_flat_utts, self.dense1], 1)

            # Concatination of both features to a Tensor
            # self.h_pool2_flat_utts_ib = tf.concat([self.h_pool2_flat_utts, self.input_ib1, self.input_ib2], 1)
            #
            # # Concatination of both features to a Tensor
            # self.h_pool2_flat_pred_ib_st = tf.concat([self.h_pool2_flat_utts_ib, self.input_st1, self.input_st2], 1)
            #
            # # Concatination of both features to a Tensor
            #self.h_pool2_flat = tf.concat([self.h_pool2_flat_utts, self.input_cl0], 1)
            #
            # # Concatination of both features to a Tensor
            # self.h_pool2_flat_pred_ib_st_clf_yn = tf.concat([self.h_pool2_flat_pred_ib_st_clf, self.input_yn1, self.input_yn2], 1)
            #
            # # Adding handcrafted features to a Tensor
            # self.h_pool2_flat = tf.concat([self.h_pool2_flat_pred_ib_st_clf_yn, self.input_x_hand], 1)

            ####################################################################################
            ####################################################################################

            with tf.variable_scope('FCNN3') as scope:

                W = tf.get_variable(
                    "W",
                    shape=[self.h_pool2_flat.get_shape()[1], 256],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[256]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.dense = tf.nn.xw_plus_b(self.h_pool2_flat, W, b, name="scores")

            ####################################################################################
            ####################################################################################

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.dense, self.dropout_keep_prob)
            # with tf.name_scope("dropout"):
            #     self.h_drop = tf.nn.dropout(self.dense, self.dropout_keep_prob)
            # with tf.name_scope("dropout"):
            #     self.h_drop = tf.nn.dropout(self.dense, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable(
                    "W",
                    shape=[self.dense.get_shape()[1], num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                self.true_labels = tf.argmax(self.input_y, 1)
                correct_predictions = tf.equal(self.predictions, self.true_labels)
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    # @staticmethod
    def _get_cell(self,hidden_size, cell_type):
        if cell_type == "vanilla":
            return tf.nn.rnn_cell.BasicRNNCell(hidden_size)
        elif cell_type == "lstm":
            return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        elif cell_type == "gru":
            return tf.nn.rnn_cell.GRUCell(hidden_size)
        else:
            print("ERROR: '" + cell_type + "' is a wrong cell type !!!")
            return None

    # Length of the sequence data
    # @staticmethod
    def _length(self, seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    # Extract the output of last cell of each sequence
    # Ex) The movie is good -> length = 4
    #     output = [ [1.314, -3.32, ..., 0.98]
    #                [0.287, -0.50, ..., 1.55]
    #                [2.194, -2.12, ..., 0.63]
    #                [1.938, -1.88, ..., 1.31]
    #                [  0.0,   0.0, ...,  0.0]
    #                ...
    #                [  0.0,   0.0, ...,  0.0] ]
    #     The output we need is 4th output of cell, so extract it.
    # @staticmethod
    def last_relevant(self, seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, input_size])
        return tf.gather(flat, index)