import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
import os
import cv2

os.chdir('C:\\Users\\ASUS\\Desktop\\deeplearn')

with open('notMNIST.pickle','rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels'] 
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save

print('train:',train_dataset.shape,train_labels.shape)
print('valid:',valid_dataset.shape,valid_labels.shape)
print('test:',test_dataset.shape,test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset,labels):
    dataset = dataset.reshape((-1,image_size**2)).astype(np.float32)
    labels = (np.arange(num_labels)==labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset,train_labels = reformat(train_dataset,train_labels)
valid_dataset,valid_labels = reformat(valid_dataset,valid_labels)
test_dataset,test_labels = reformat(test_dataset,test_labels)

img = cv2.imread('photo_1.jpg')
im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
_, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
im_28 = cv2.resize(im_th, (28,28), interpolation=cv2.INTER_AREA)
im_28 = cv2.dilate(im_28, (3, 3))
im_2828 = im_28.reshape((-1,28*28)).astype(np.float32)

batch_size = 128
num_hidden_nodes1 = 1024
num_hidden_nodes2 = 100
beta_regul = 1e-3

graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    tf_im_2828 = tf.constant(im_2828)
    
    global_step = tf.Variable(0)

    # Variables.
    weights1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_hidden_nodes1], stddev=np.sqrt(2.0 / (image_size * image_size))), name='weights1')
    biases1 = tf.Variable(tf.zeros([num_hidden_nodes1]), name='biases1')
    weights2 = tf.Variable(tf.truncated_normal([num_hidden_nodes1, num_hidden_nodes2], stddev=np.sqrt(2.0 / num_hidden_nodes1)), name = 'weights2')
    biases2 = tf.Variable(tf.zeros([num_hidden_nodes2]), name = 'biases2')
    weights3 = tf.Variable(tf.truncated_normal([num_hidden_nodes2, num_labels], stddev=np.sqrt(2.0 / num_hidden_nodes2)), name = 'weights3')
    biases3 = tf.Variable(tf.zeros([num_labels]), name = 'biases3')
  
    # Training computation.
    lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
    lay2_train = tf.nn.relu(tf.matmul(lay1_train, weights2) + biases2)
    logits = tf.matmul(lay2_train, weights3) + biases3
    loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels)) + beta_regul * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(weights3))
  
    # Optimizer.
    learning_rate = tf.train.exponential_decay(0.5, global_step, 1000, 0.65, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    
    lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
    lay2_valid = tf.nn.relu(tf.matmul(lay1_valid, weights2) + biases2)
    valid_prediction = tf.nn.softmax(tf.matmul(lay2_valid, weights3) + biases3)
    
    lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
    lay2_test = tf.nn.relu(tf.matmul(lay1_test, weights2) + biases2)
    test_prediction = tf.nn.softmax(tf.matmul(lay2_test, weights3) + biases3)

    lay1_im_2828 = tf.nn.relu(tf.matmul(tf_im_2828, weights1) + biases1)
    lay2_im_2828 = tf.nn.relu(tf.matmul(lay1_im_2828, weights2) + biases2)
    im_2828_prediction = tf.nn.softmax(tf.matmul(lay2_im_2828, weights3) + biases3)

    saver = tf.train.Saver()

num_steps = 9001

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
    save_path = saver.save(session, "./95model/model.ckpt")
    print("Model saved in file: %s" % save_path)
    print("im 2828:", im_2828_prediction.eval())
