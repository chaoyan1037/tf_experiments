
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib import layers


# TODO, add gamma_regularizer and beta_regularizer
def batch_norm(x, training, scope):
  with slim.arg_scope([layers.batch_norm], scope=scope, decay=0.9, center=True,
      scale=True, param_regularizers=layers.l2_regularizer(scale=1e-4),
      updates_collections=None, zero_debias_moving_mean=True):
    bn = tf.cond(training,
        lambda: layers.batch_norm(inputs=x, is_training=True),
        lambda: layers.batch_norm(inputs=x, is_training=False, reuse=True))
    
    return bn


def conv_layer(x, filters, kernel_size, stride, scope, padding='valid'):
  with tf.name_scope(scope):
    x = tf.layers.conv2d(x, filters, kernel_size, stride, 
        use_bias=False, padding=padding, name=scope,
        kernel_initializer=layers.xavier_initializer(uniform=True),
        kernel_regularizer=layers.l2_regularizer(scale=1e-4))

    return x


def dropout_layer(x, dropout_rate, is_training):
  return layers.dropout(x, keep_prob=1 - dropout_rate, is_training=is_training)




class DenseNet():
  def __init__(self, x, nb_classes, nb_blocks, stages,
      growth_k, filters, compression=0.5, dropout_rate=None, training=True):
    # define model parameters
    self.nb_classes = nb_classes
    self.nb_blocks = nb_blocks
    self.stages = stages
    self.growth_k = growth_k
    self.filters = filters
    self.compression = compression
    self.dropout_rate = dropout_rate
    # if in training step
    self.training = training


  # TODO: add dropout
  def bottleneck_layer(self, x, filters, scope):
    # [BN --> ReLU --> conv11 --> BN --> ReLU -->conv33]
    with tf.name_scope(scope):
      x = batch_norm(x, training=self.training, scope=scope + '_bn1')
      x = tf.nn.relu(x, name=scope + '_relu1')
      x = conv_layer(x, 4 * filters, 1, 1, scope=scope + '_conv1')
      if self.dropout_rate:
        x = dropout_layer(x, self.dropout_rate, self.training)

      x = batch_norm(x, training=self.training, scope=scope + '_bn2')
      x = tf.nn.relu(x, name=scope + '_relu2')
      x = conv_layer(x, filters, 3, 1, padding='same', scope=scope + '_conv2')
      if self.dropout_rate:
        x = dropout_layer(x, self.dropout_rate, self.training)

    return x 


  def transition_layer(self, x, filters, scope):
    # [BN --> conv11 --> avg_pool2]
    with tf.name_scope(scope):
      x = batch_norm(x, training=self.training, scope=scope + '_bn1')
      x = tf.nn.relu(x, name=scope + '_relu1')
      x = conv_layer(x, filters, 1, 1, scope=scope + '_conv1')
      if self.dropout_rate:
        x = dropout_layer(x, self.dropout_rate, self.training)
      x = layers.avg_pool2d(x, 2, 2)

    return x 


  def dense_block(self, x, nb_layers, db_index):
    layer_name = 'db' + str(db_index)
    with tf.name_scope(layer_name):
      for i in range(nb_layers):
        scope = layer_name + '_bt_' + str(i)
        merged_x = self.bottleneck_layer(x, self.growth_k, scope)
        x = tf.concat([x, merged_x], axis=-1)

    return x


  def build_model(self, x):
    x = conv_layer(x, self.filters, 3, 1, padding='same', scope='initial_conv')
    #x = batch_norm(x, training=self.training, scope='initial_bn')
    #x = tf.nn.relu(x, name='initial_relu')
    #x = layers.max_pool2d(x, 2, 2, scope='initial_maxpool')

    # compresion the feature
    compression = 0.8
    for i in range(self.nb_blocks - 1):
      nb_layers = self.stages[i]
      x = self.dense_block(x, nb_layers, i)
       
      nb_filters = x.get_shape().as_list()[-1]
      nb_filters = int(nb_filters * compression)
      # add transition
      x = self.transition_layer(x, nb_filters, scope='transition_' + str(i))

    # the last denseblock does not have a transition
    nb_layers = self.stages[-1]
    x = self.dense_block(x, nb_layers, self.nb_blocks - 1)
    x = batch_norm(x, training=self.training, scope='bn_final')
    x = tf.nn.relu(x, name='relu_final')
    # global average pooling
    x = tf.reduce_mean(x, [1, 2])
    x = layers.fully_connected(x, self.nb_classes, activation_fn=None,
        biases_regularizer=layers.l2_regularizer(1e-4),
        weights_regularizer=layers.l2_regularizer(1e-4))

    # return output digits
    return x
