import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle
import time

# TODO: Load traffic signs data.
with open('train.p','rb') as f:
    data = pickle.load(f)
    features, labels = data['features'], data['labels']

# TODO: Split data into training and validation sets.
X_train,X_valid,y_train,y_valid = train_test_split(features,labels,test_size = 0.2,random_state = 0)
nb_train = X_train.shape[0]
nb_classes = 43

# TODO: Define placeholders and resize operation.
X = tf.placeholder(dtype=tf.float32,shape=(None,32,32,3))
X_normalized = tf.image.resize_images(X,size=(227,227))
y = tf.placeholder(dtype=tf.int32,shape=(None))
one_hot_y = tf.one_hot(y,depth=nb_classes)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(X_normalized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.

shape = (fc7.get_shape().as_list()[-1],nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape=shape,mean=0,stddev=0.1))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7,fc8W,fc8b)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,one_hot_y))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(one_hot_y,1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={X: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# TODO: Train and evaluate the feature extraction model.
EPOCHS = 10
BATCH_SIZE = 128
print(X_train.shape)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("Training ...")
    for epoch in range(EPOCHS):
        t = time.time()
        exp_train_X, exp_train_y = (X_train, y_train)

        # Loop over all batches
        for offset in range(0, nb_train, BATCH_SIZE):
            # get batch data
            # end = min(batch_size*(i+1),num_examples)
            end = offset + BATCH_SIZE
            batch_X, batch_y = X_train[offset:end], y_train[offset:end]

            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={X: batch_X, y: batch_y})

        print("time elapsed {} ".format(time.time()-t))
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(epoch + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    print("Training finished.")