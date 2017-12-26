import tensorflow as tf
from time import time
import numpy as np
import sys
import os


def recup(folder, num):
  X_train = np.load('./'+ folder + '/Xtrain_'+str(num)+'.npy')
  X_test = np.load('./'+ folder + '/Xtest_'+str(num)+'.npy')
  y_test = np.load('./'+ folder + '/Ytest_'+str(num)+'.npy')
  y_train = np.load('./'+ folder + '/Ytrain_'+str(num)+'.npy')
  return X_train, y_train, X_test, y_test

def recupTest(folder, num):
  X_test = np.load('./'+ folder + '/Xtest_'+str(num)+'.npy')
  y_test = np.load('./'+ folder + '/Ytest_'+str(num)+'.npy')
  return X_test, y_test

def recupTrain(folder, num):
  X_train = np.load('./'+ folder + '/Xtrain_'+str(num)+'.npy')
  y_train = np.load('./'+ folder + '/Ytrain_'+str(num)+'.npy')
  return X_train, y_train

def new_weights_fc(name,shape):
    return tf.get_variable(name, shape=shape, dtype=tf.float32,
           initializer=tf.contrib.layers.xavier_initializer())
       
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length], dtype=tf.float32), dtype=tf.float32)

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def new_fc_layer(name,input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs, use_nonlinear):
    weights = new_weights_fc(name,[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases
    if use_nonlinear:
      layer = tf.nn.tanh(layer)

    return layer, weights

X_test, y_test = recupTest('dataTrain',0)

imgSize = 64
n_classes = 5
batch_size = 64


new_saver = tf.train.import_meta_graph('./final_model/best_model.meta')
graph = tf.get_default_graph()

x = graph.get_tensor_by_name("input_x:0")
y = graph.get_tensor_by_name("labels:0")
keep_prob = graph.get_tensor_by_name("dropRate:0")
layer_conv1c1 = graph.get_tensor_by_name("dropout_3/mul:0")

layer_conv1c1 = tf.stop_gradient(layer_conv1c1) # It's an identity function

layer_flat, num_features = flatten_layer(layer_conv1c1)
print(layer_flat, num_features)
layer_f, weights_f = new_fc_layer("f",input=layer_flat,
                       num_inputs=num_features,
                       num_outputs=150,
                       use_nonlinear=True)

layer_fc, weights_fc = new_fc_layer("fc",input=layer_f,
                       num_inputs=150,
                       num_outputs=50,
                       use_nonlinear=True)

layer_f1, weights_f1 = new_fc_layer("fc1",input=layer_fc,
                       num_inputs=50,
                       num_outputs=n_classes,
                       use_nonlinear=False)

pred = tf.nn.tanh(layer_f1, name='predictionBox')
print(layer_f)
rate = tf.placeholder(tf.float32, shape=[])
l_rate = 0.001#5e-4
drop_rate = 1
beta = 0.001
cost = tf.reduce_mean(tf.square(y - pred))

optimizer = tf.train.AdamOptimizer(rate).minimize(cost)
accuracy = cost
saver = tf.train.Saver()
save_dir = 'final_model_bis/'
if not os.path.exists(save_dir):
  os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_model')

hm_epochs = 150
t = time()
compteur = 0
prec = 10e100

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  new_saver.restore(sess, tf.train.latest_checkpoint('./final_model/'))
  # Now, you run this with fine-tuning data in sess.run()
  epoch = 0
  while epoch < hm_epochs:
    epoch_loss = 0
    epoch += 1
    X_train, y_train = recupTrain('dataTrain', 0)
    for name in [0]:
      #X_train, y_train = recupTrain('dataTrain', name)
      for g in range(0,len(X_train),batch_size):
        _, c = sess.run([optimizer, cost], feed_dict={keep_prob: 1, rate: l_rate, x: X_train[g:g+batch_size], y: y_train[g:g+batch_size]})
        
        sys.stdout.write('\r' + str(g) + '/' + str(len(X_train)))
        sys.stdout.flush()
        epoch_loss += c

    tempsEcoule = time() - t

    sys.stdout.write('\rEpoch : ' + str(epoch) + ' Loss : ' + str(epoch_loss) + ' Batch size : ' + str(batch_size) \
       + ' LRate : ' + str(l_rate) + ' DropRate : ' + str(drop_rate) + ' Time : ' + str(tempsEcoule))
    res2 = accuracy.eval({x:X_train[:batch_size], y:y_train[:batch_size], keep_prob: 1})
    res3 = accuracy.eval({x:X_test[:batch_size], y:y_test[:batch_size], keep_prob: 1})
    
    sys.stdout.write('\nTrain : ' + str(res2) + ' Test : ' + str(res3))
    sys.stdout.write('\n')
    t = time()

    if epoch_loss > prec:
      compteur += 1
    else:
      if compteur > 0:
        compteur -= 1
      prec = epoch_loss
      saver.save(sess=sess, save_path=save_path)
    if compteur >= 2:
      compteur = 0
      l_rate /= 1.5
      #batch_size = int(batch_size*1.5)

  res2, res = 0, 0
  for g in range(0,len(X_train),batch_size):
      res2 += accuracy.eval({x:X_train[g:g+batch_size], y:y_train[g:g+batch_size], keep_prob: 1})
  res2 /= (g/batch_size) + 1
  for g in range(0,len(X_test),batch_size):
      res += accuracy.eval({x:X_test[g:g+batch_size], y:y_test[g:g+batch_size], keep_prob: 1})
  res /= (g/batch_size) + 1
print('Epoch', epoch,'loss :',epoch_loss,'train :',res2,'test :', res)