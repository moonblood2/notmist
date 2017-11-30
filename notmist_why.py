import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
import os

os.chdir('C:\\Users\\ASUS\\Desktop\\deeplearn')

with open('notMNIST.pickle','rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels =save['train_labels']
    valid_dataset =save['valid_dataset']
    valid_labels =save['valid_labels']
    test_dataset =save['test_dataset']
    test_labels =save['test_labels']

img_size = 28
num_labels = 10

def reformat(data,labels):
    data = data.reshape(-1,img_size**2).astype(np.float32)
    labels = (np.arange(num_labels)==labels[:,None]).astype(np.float32)
    return data,labels

train_dataset,train_labels = reformat(train_dataset,train_labels)
valid_dataset,valid_labels = reformat(valid_dataset,valid_labels)
test_dataset,test_labels = reformat(test_dataset,test_labels)

batch_size = 128

graph = tf.Graph()

with graph.as_default():

    tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size,img_size**2))
    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size,num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    weights = tf.Variable(tf.truncated_normal([img_size**2,num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    logits = tf.matmul(tf_train_dataset,weights)+biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels,logits=logits))

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    train_predictions = tf.nn.softmax(logits)
    valid_predictions = tf.nn.softmax(tf.matmul(tf_valid_dataset,weights)+biases)
    test_predictions = tf.nn.softmax(tf.matmul(tf_test_dataset,weights)+biases)

num_steps = 3001

def accuracy(predictions,labels):
    return 100.0*(np.sum(np.argmax(predictions,1)==np.argmax(labels,1))/labels.shape[0])

with tf.Session(graph = graph) as sess:

    tf.initialize_all_variables().run()
    print('initialized')

    for step in range(num_steps):
        offset = np.random.randint(0,high = train_labels.shape[0]-batch_size)
        batch_dataset = train_dataset[offset:(offset+batch_size)]
        batch_labels = train_labels[offset:(offset+batch_size)]
        feed_dict = {tf_train_dataset:batch_dataset, tf_train_labels:batch_labels}

        _,l,predictions = sess.run([optimizer,loss,train_predictions],feed_dict=feed_dict)

        if step%500 == 0:
            print('loss',l)
            print('train acc', accuracy(predictions,batch_labels))
            print('valid acc', accuracy(valid_predictions.eval(),valid_labels))
    print('test acc', accuracy(test_predictions.eval(),test_labels))
