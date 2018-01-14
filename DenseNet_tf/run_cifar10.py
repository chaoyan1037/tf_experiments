
import json
import os
import time
import keras
import numpy as np
import tensorflow as tf


#from tf.keras.datasets import cifar10
#from tf.keras.utils import np_utils
from keras.datasets import cifar10
from keras.utils import np_utils


from densenet_tf import DenseNet

img_size = 32
img_channel = 3

total_epochs = 256
batch_size = 64

init_learning_rate = 1e-1
learning_rate_decay = 1e-5

drop_out = 0.5


# evaluate predict result
def evaluate(session, X_test, Y_test):
  num_test_sample = X_test.shape[0]
  num_splits = num_test_sample // batch_size
  arr_splits = np.array_split(np.arange(num_test_sample), num_splits)

  loss = 0.0
  accuracy = 0.0
  for batch_idx in arr_splits:
    X_batch, Y_batch = X_test[batch_idx], Y_test[batch_idx]
    # prepare test input
    test_feed_dict = {
      x: X_batch,
      label: Y_batch,
      learning_rate: epoch_learning_rate,
      training_flag: False
    }

    loss_batch, correct_batch = session.run([loss_sum, correct_count],
        feed_dict=test_feed_dict)
    loss += loss_batch / num_test_sample
    accuracy += correct_batch / num_test_sample

  return loss, accuracy


# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#X_train = X_train[:1000]
#y_train = y_train[:1000]
#X_test = X_test[:200]
#y_test = y_test[:200]

nb_classes = len(np.unique(y_train))
img_dim = X_train.shape[1:]


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

mean_rgb = np.mean(X_train, axis=(0, 1, 2))

X_train -= mean_rgb
X_test -= mean_rgb




# image_size = 32, img_channels = 3, class_num = 10 in cifar10
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, img_channel])
label = tf.placeholder(tf.float32, shape=[None, nb_classes])

training_flag = tf.placeholder(tf.bool, name='is_training')

learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = DenseNet(x=x, nb_classes=nb_classes, nb_blocks=3, stages=[12, 12, 12],
    growth_k=12, filters=24, dropout_rate=0.2,
    training=training_flag).build_model(x)

loss_sum = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits)
loss = tf.reduce_mean(loss_sum)
loss_sum = tf.reduce_sum(loss_sum)


#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-4)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
    momentum=0.9, use_nesterov=True)


train = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
# count the number of correct prediction
correct_count = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(tf.global_variables())

# allocate GPU memory as needed
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
  ckpt = tf.train.get_checkpoint_state('./model')
  # if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
  #   print('restore model from ', ckpt.model_checkpoint_path)
  #   saver.restore(sess, ckpt.model_checkpoint_path)
  # else:
  sess.run(tf.global_variables_initializer())
  summary_writer = tf.summary.FileWriter('./logs', sess.graph)

  list_train_loss = []
  list_train_acc = []
  list_test_loss = []
  list_test_acc = []
  list_lr = []

  epoch_learning_rate = init_learning_rate
  for epoch in range(total_epochs):
    start_time = time.time()
    # adjust learning rate
    if epoch == int(total_epochs * 0.5) or epoch == int(total_epochs * 0.75):
      epoch_learning_rate = epoch_learning_rate / 10
    #lr = self.lr * (1. / (1. + self.decay * self.iterations))
    epoch_learning_rate *= 1. / (1. + learning_rate_decay * epoch)

    #print('lr: ', epoch_learning_rate)  
    loss_train_sum = 0.0
    correct_train = 0.0

    num_train_sample = X_train.shape[0]
    num_splits = num_train_sample // batch_size
    index_range = np.arange(num_train_sample)
    np.random.shuffle(index_range)
    arr_splits = np.array_split(index_range, num_splits)

    for batch_idx in arr_splits:
      X_batch, Y_batch = X_train[batch_idx], Y_train[batch_idx]
      # prepare training input
      train_feed_dict = {x: X_batch, label: Y_batch, training_flag: True,
          learning_rate: epoch_learning_rate}
      # run training
      _, loss_batch, correct_batch = sess.run([train, loss_sum, correct_count],
          feed_dict=train_feed_dict)

      loss_train_sum += loss_batch
      correct_train += correct_batch

    # get mean loss and accuracy
    loss_train = loss_train_sum / num_train_sample
    accuracy_train = correct_train / num_train_sample

    loss_test, acc_test = evaluate(sess, X_test, Y_test)

    line = 'epoch: {}/{}, time: {:.2f}, train acc: {:.4f}, '\
           'loss: {:.4f}, test acc: {:.4f}, loss: {:.4f}'
    print(line.format(epoch, total_epochs, time.time() - start_time, 
        accuracy_train, loss_train, acc_test, loss_test))

    list_train_loss.append(loss_train)
    list_train_acc.append(accuracy_train)
    list_test_loss.append(loss_test)
    list_test_acc.append(acc_test)
    list_lr.append(epoch_learning_rate)
    
    if epoch % 10 == 0:
      saver.save(sess=sess, save_path='./model/dense.ckpt', global_step=epoch)
      
      d_log = {}
      d_log["batch_size"] = batch_size
      d_log["drop_out"] = drop_out
      d_log["nb_epoch"] = total_epochs
      d_log["train_loss"] = list_train_loss
      d_log["train_acc"] = list_train_acc
      d_log["test_loss"] = list_test_loss
      d_log["test_acc"] = list_test_acc
      d_log["learning_rate"] = list_lr
      
      json_file = os.path.join('./logs/running.json')
      with open(json_file, 'w') as fp:
        json.dump(d_log, fp, indent=4, sort_keys=True)
