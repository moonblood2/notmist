import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
import os

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

graph = tf.Graph()
with graph.as_default():

    tf_train_dataset = tf.constant(train_dataset[:10000,:])
    tf_train_labels = tf.constant(train_labels[:10000])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(valid_dataset)

    weights = tf.Variable(tf.truncated_normal([image_size**2,num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    logits = tf.matmul(tf_train_dataset,weights)+biases
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=tf_train_labels))

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

num_steps = 801

def accuracy(predicitons, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

with tf.Session(graph = graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if (step % 100 == 0):
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %.1f%%' % accuracy(predictions, train_labels[:10000, :]))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
