import tensorflow as tf

from .ops import mu_law_encode
from .mixture import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic


class WaveNetModel(object):
    def __init__(self, batch_size, dilations, filter_width, residual_channels, dilation_channels, skip_channels,
                 quantization_channels=2 ** 8, out_channels=30,
                 use_biases=False, scalar_input=False, initial_filter_width=32, global_condition_channels=None,
                 global_condition_cardinality=None, local_condition_channels=80, upsample_factor=None, train_mode=True):

        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.quantization_channels = quantization_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.scalar_input = scalar_input
        self.initial_filter_width = initial_filter_width
        self.global_condition_channels = global_condition_channels
        self.global_condition_cardinality = global_condition_cardinality
        self.local_condition_channels = local_condition_channels
        self.upsample_factor = upsample_factor
        self.train_mode = train_mode
        self.receptive_field = WaveNetModel.calculate_receptive_field(self.filter_width, self.dilations,
                                                                      self.scalar_input, self.initial_filter_width)
        self.out_channels = out_channels

        self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)

    @staticmethod
    def calculate_receptive_field(filter_width, dilations, scalar_input, initial_filter_width):
        receptive_field = (filter_width - 1) * sum(
            dilations) + 1
        if scalar_input:
            receptive_field += initial_filter_width - 1
        else:
            receptive_field += filter_width - 1
        return receptive_field

    def _create_causal_layer(self, input_batch):
        with tf.name_scope('causal_layer'):
            if self.scalar_input:
                return tf.layers.conv1d(input_batch, filters=self.residual_channels,
                                        kernel_size=self.initial_filter_width, padding='valid', dilation_rate=1,
                                        use_bias=False)
            else:
                return tf.layers.conv1d(input_batch, filters=self.residual_channels, kernel_size=self.filter_width,
                                        padding='valid', dilation_rate=1, use_bias=False)

    def _create_queue(self):
        with tf.variable_scope('queue'):
            if self.scalar_input:
                self.causal_queue = tf.Variable(
                    initial_value=tf.zeros(shape=[self.batch_size, self.initial_filter_width, 1], dtype=tf.float32),
                    name='causal_queue', trainable=False)
            else:
                self.causal_queue = tf.Variable(
                    initial_value=tf.zeros(shape=[self.batch_size, self.filter_width, self.quantization_channels],
                                           dtype=tf.float32), name='causal_queue', trainable=False)

            self.local_condition_queue = tf.Variable(
                initial_value=tf.zeros(shape=[self.batch_size, self.filter_width, self.local_condition_channels],
                                       dtype=tf.float32), name='local_condition_queue', trainable=False)

            self.dilation_queue = []
            for i, d in enumerate(self.dilations):
                q = tf.Variable(initial_value=tf.zeros(
                    shape=[self.batch_size, d * (self.filter_width - 1) + 1, self.dilation_channels], dtype=tf.float32),
                    name='dilation_queue'.format(i), trainable=False)
                self.dilation_queue.append(q)

        self.queue_initializer = tf.variables_initializer(
            self.dilation_queue + [self.causal_queue, self.local_condition_queue])

    def _create_dilation_layer(self, input_batch, layer_index, dilation, local_condition_batch, global_condition_batch,
                               output_width):
        with tf.variable_scope('dilation_layer'):
            conv_filter = tf.layers.conv1d(input_batch, filters=self.dilation_channels, kernel_size=self.filter_width,
                                           dilation_rate=dilation, padding='valid', use_bias=self.use_biases,
                                           name='conv_filter')
            conv_gate = tf.layers.conv1d(input_batch, filters=self.dilation_channels, kernel_size=self.filter_width,
                                         dilation_rate=dilation, padding='valid', use_bias=self.use_biases,
                                         name='conv_gate')

            if global_condition_batch is not None:
                conv_filter += tf.layers.conv1d(global_condition_batch, filters=self.dilation_channels, kernel_size=1,
                                                padding="same", use_bias=False, name="gc_filter")
                conv_gate += tf.layers.conv1d(global_condition_batch, filters=self.dilation_channels, kernel_size=1,
                                              padding="same", use_bias=False, name="gc_gate")

            if local_condition_batch is not None:
                local_filter = tf.layers.conv1d(local_condition_batch, filters=self.dilation_channels, kernel_size=1,
                                                padding="same", use_bias=False, name="lc_filter")
                local_gate = tf.layers.conv1d(local_condition_batch, filters=self.dilation_channels, kernel_size=1,
                                              padding="same", use_bias=False, name="lc_gate")

                local_filter = tf.slice(local_filter, [0, 0, 0], [-1, tf.shape(conv_filter)[1], -1])
                local_gate = tf.slice(local_gate, [0, 0, 0], [-1, tf.shape(conv_gate)[1], -1])

                conv_filter += local_filter
                conv_gate += local_gate

            out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

            # The 1x1 conv to produce the residual output  == FC
            transformed = tf.layers.conv1d(out, filters=self.residual_channels, kernel_size=1, padding="same",
                                           use_bias=self.use_biases, name="dense")

            # The 1x1 conv to produce the skip output
            skip_cut = tf.shape(out)[1] - output_width
            out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, self.dilation_channels])
            skip_contribution = tf.layers.conv1d(out_skip, filters=self.skip_channels, kernel_size=1, padding="same",
                                                 use_bias=self.use_biases, name="skip")

            input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]
            input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])

            return skip_contribution, input_batch + transformed

    def create_upsample(self, local_condition_batch):
        local_condition_batch = tf.expand_dims(local_condition_batch, [3])
        # local condition batch N H W C

        for i in range(len(self.upsample_factor)):
            local_condition_batch = tf.layers.conv2d_transpose(local_condition_batch, filters=1,
                                                               kernel_size=(self.upsample_factor[i], self.filter_width),
                                                               strides=(self.upsample_factor[i], 1), padding='same',
                                                               use_bias=False, name='upsample{}'.format(i))

        local_condition_batch = tf.squeeze(local_condition_batch, [3])
        return local_condition_batch

    def _create_network(self, input_batch, local_condition_batch, global_condition_batch):
        '''Construct the WaveNet network.'''

        if not self.train_mode:
            self._create_queue()

        outputs = []
        current_layer = input_batch
        if not self.train_mode:
            self.causal_queue = tf.scatter_update(self.causal_queue, tf.range(self.batch_size),
                                                  tf.concat([self.causal_queue[:, 1:, :], input_batch], axis=1))
            current_layer = self.causal_queue

            self.local_condition_queue = tf.scatter_update(self.local_condition_queue, tf.range(self.batch_size),
                                                           tf.concat([self.local_condition_queue[:, 1:, :],
                                                                      local_condition_batch], axis=1))
            local_condition_batch = self.local_condition_queue

        # Pre-process the input with a regular convolution
        current_layer = self._create_causal_layer(current_layer)

        if self.train_mode:
            output_width = tf.shape(input_batch)[
                               1] - self.receptive_field + 1
        else:
            output_width = 1

        # Add all defined dilation layers.
        with tf.variable_scope('dilated_stack'):
            for layer_index, dilation in enumerate(
                    self.dilations):
                with tf.variable_scope('layer{}'.format(layer_index)):

                    if not self.train_mode:
                        self.dilation_queue[layer_index] = tf.scatter_update(self.dilation_queue[layer_index],
                                                                             tf.range(self.batch_size), tf.concat(
                                [self.dilation_queue[layer_index][:, 1:, :], current_layer], axis=1))
                        current_layer = self.dilation_queue[layer_index]

                    output, current_layer = self._create_dilation_layer(current_layer, layer_index, dilation,
                                                                        local_condition_batch, global_condition_batch,
                                                                        output_width)
                    outputs.append(output)
        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)
            conv1 = tf.layers.conv1d(transformed1, filters=self.skip_channels, kernel_size=1, padding="same",
                                     use_bias=self.use_biases)

            transformed2 = tf.nn.relu(conv1)
            if self.scalar_input:
                conv2 = tf.layers.conv1d(transformed2, filters=self.out_channels, kernel_size=1, padding="same",
                                         use_bias=self.use_biases)
            else:
                conv2 = tf.layers.conv1d(transformed2, filters=self.quantization_channels, kernel_size=1,
                                         padding="same", use_bias=self.use_biases)

        return conv2

    def _one_hot(self, input_batch):
        '''One-hot encodes the waveform amplitudes.

        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        '''
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(input_batch, depth=self.quantization_channels,
                                 dtype=tf.float32)  # (1, ?, 1) --> (1, ?, 1, 256)
            shape = [self.batch_size, -1, self.quantization_channels]
            encoded = tf.reshape(encoded, shape)  # (1, ?, 1, 256) --> (1, ?, 256)
        return encoded

    def _embed_gc(self, global_condition):
        '''Returns embedding for global condition.
        :param global_condition: Either ID of global condition for
               tf.nn.embedding_lookup or actual embedding. The latter is
               experimental.
        :return: Embedding or None
        '''
        embedding = None
        if self.global_condition_cardinality is not None:
            # Only lookup the embedding if the global condition is presented
            # as an integer of mutually-exclusive categories ...
            embedding_table = tf.get_variable('gc_embedding',
                                              [self.global_condition_cardinality, self.global_condition_channels],
                                              dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(
                    uniform=False))
            embedding = tf.nn.embedding_lookup(embedding_table, global_condition)
        elif global_condition is not None:
            # In this case, the number of global_embedding channels must be
            # equal to the the last dimension of the global_condition tensor.
            gc_batch_rank = len(global_condition.get_shape())
            dims_match = (global_condition.get_shape()[gc_batch_rank - 1] == self.global_condition_channels)
            if not dims_match:
                raise ValueError('Shape of global_condition {} does not match global_condition_channels {}.'.format(
                    global_condition.get_shape(),
                    self.global_condition_channels))
            embedding = global_condition

        if embedding is not None:
            embedding = tf.reshape(embedding, [self.batch_size, 1, self.global_condition_channels])

        return embedding

    def predict_proba_incremental(self, waveform, upsampled_local_condition=None, global_condition=None,
                                  name='wavenet'):
        """
        local_condition: upsampled local condition
        """

        with tf.variable_scope(name):

            if self.scalar_input:
                encoded = tf.reshape(waveform, [self.batch_size, -1, 1])
            else:
                encoded = tf.one_hot(waveform, self.quantization_channels)
                encoded = tf.reshape(encoded,
                                     [self.batch_size, -1, self.quantization_channels])  # encoded shape=(N,1, 256)

            gc_embedding = self._embed_gc(global_condition)  # --> shape=(1, 1, 32)

            # local condition
            if upsampled_local_condition is not None:
                upsampled_local_condition = tf.reshape(upsampled_local_condition,
                                                       [self.batch_size, -1, self.local_condition_channels])

            raw_output = self._create_network(encoded, upsampled_local_condition,
                                              gc_embedding)

            if self.scalar_input:
                out = tf.reshape(raw_output, [self.batch_size, -1, self.out_channels])
                proba = sample_from_discretized_mix_logistic(out)
            else:
                out = tf.reshape(raw_output, [self.batch_size, self.quantization_channels])
                proba = tf.cast(tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)

            return proba

    def add_loss(self, input_batch, local_condition=None, global_condition_batch=None, l2_regularization_strength=None,
                 name='wavenet'):
        '''Creates a WaveNet network and returns the autoencoding loss.

        The variables are all scoped to the given name.
        '''
        with tf.variable_scope(name):
            encoded_input = mu_law_encode(input_batch,
                                          self.quantization_channels)

            gc_embedding = self._embed_gc(
                global_condition_batch)
            encoded = self._one_hot(encoded_input)  # (1, ?, quantization_channels=256)
            if self.scalar_input:
                network_input = tf.reshape(tf.cast(input_batch, tf.float32), [self.batch_size, -1, 1])
            else:
                network_input = encoded

            # Cut off the last sample of network input to preserve causality.
            network_input_width = tf.shape(network_input)[1] - 1
            if self.scalar_input:
                input = tf.slice(network_input, [0, 0, 0], [-1, network_input_width, 1])
            else:
                input = tf.slice(network_input, [0, 0, 0], [-1, network_input_width, self.quantization_channels])

            # local condition
            if local_condition is not None:
                local_condition = self.create_upsample(local_condition)

            raw_output = self._create_network(input, local_condition,
                                              gc_embedding)

            with tf.name_scope('loss'):
                # Cut off the samples corresponding to the receptive field
                # for the first predicted sample.
                target_output = tf.slice(network_input, [0, self.receptive_field, 0],
                                         [-1, -1, -1])

                if self.scalar_input:
                    loss = discretized_mix_logistic_loss(raw_output, target_output, num_class=2 ** 16, reduce=False)
                    reduced_loss = tf.reduce_mean(loss)
                else:
                    target_output = tf.reshape(target_output, [-1, self.quantization_channels])
                    prediction = tf.reshape(raw_output, [-1, self.quantization_channels])
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=target_output)
                    reduced_loss = tf.reduce_mean(loss)

                tf.summary.scalar('loss', reduced_loss)

                if l2_regularization_strength is None:
                    self.loss = reduced_loss
                else:
                    # L2 regularization for all trainable parameters
                    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if not ('bias' in v.name)])

                    # Add the regularization term to the loss
                    total_loss = (reduced_loss + l2_regularization_strength * l2_loss)

                    tf.summary.scalar('l2_loss', l2_loss)
                    tf.summary.scalar('total_loss', total_loss)

                    self.loss = total_loss

    def add_optimizer(self, hparams, global_step):
        '''Adds optimizer to the graph. Supposes that initialize function has already been called.
        '''
        with tf.variable_scope('optimizer'):
            hp = hparams

            learning_rate = tf.train.exponential_decay(hp.wavenet_learning_rate, global_step, hp.wavenet_decay_steps,
                                                       hp.wavenet_decay_rate)

            # Adam optimization
            self.learning_rate = learning_rate
            optimizer = tf.train.AdamOptimizer(learning_rate)

            gradients, variables = zip(
                *optimizer.compute_gradients(self.loss))  # len(tf.trainable_variables()) = len(variables)
            self.gradients = gradients

            # Gradients clipping
            if hp.wavenet_clip_gradients:
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.)
            else:
                clipped_gradients = gradients

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                adam_optimize = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step)

        # https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        # Use adam optimization process as a dependency
        with tf.control_dependencies([adam_optimize]):
            # Create the shadow variables and add ops to maintain moving averages
            # Also updates moving averages after each update step
            # This is the optimize call instead of traditional adam_optimize one.
            assert tuple(tf.trainable_variables()) == variables  # Verify all trainable variables are being averaged
            self.optimize = self.ema.apply(variables)
