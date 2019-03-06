import tensorflow as tf
from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.contrib.model_pruning.python.layers import layers
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# Classes/targets
label_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot'
}

# Define Placeholders
images = tf.placeholder(tf.float32, [None, 784])
labels = tf.placeholder(tf.float32, [None, 10])

# Define the model
layer1 = layers.masked_fully_connected(images, 128)
layer2 = layers.masked_fully_connected(layer1, 128)
logits = layers.masked_fully_connected(layer2, len(label_dict))


def loss_fun():
    """
    Loss function, softmax cross entropy is used
    """
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    return loss


def train(loss, global_step, lr=1e-3):
    """
    Training op. Make sure that the same global step is used for pruning!
    :param loss: the loss function
    :param global_step: the global step, needs to be the same as the one used for pruning
    :param lr: the learning rate of the optimizer
    :return:
    """
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)
    return train_op


def accuracy_op():
    """
    Accuracy op, reduces the error of wrong predictions
    :return:
    """
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def pruning_params(global_step, begin_step=0, end_step=-1, pruning_freq=60,
                   sparsity_function=1e+50, target_sparsity=.50):
    """
    Creates the pruning op
    :param global_step: the global step, needed for pruning
    :param begin_step: the global step at which to begin pruning
    :param end_step: the global step at which to end pruning
    :param pruning_freq: the frequency of global step for when to prune
    :param sparsity_function: the global step used as the end point for the gradual sparsity function
    :param target_sparsity: the target sparsity
    :return: Pruning op
    """
    pruning_hparams = pruning.get_pruning_hparams()
    #pruning_hparams.begin_pruning_step = begin_step
    #pruning_hparams.end_pruning_step = end_step
    #pruning_hparams.pruning_frequency = pruning_freq
    pruning_hparams.sparsity_function_end_step = sparsity_function
    pruning_hparams.target_sparsity = target_sparsity
    p = pruning.Pruning(pruning_hparams, global_step=global_step, sparsity=target_sparsity)
    p_op = p.conditional_mask_update_op()
    p.add_pruning_summaries()
    return p_op


def plot(y, x_label='Epochs', y_label='Accuracy', title='Pruned', combined=False):
    """
    Plots the accuracy
    :param y: the accuracy to be plotted
    :param x_label: label on the x-axis
    :param y_label: label on the y-axis
    :param title: title of the plot
    :param combined: whether the plot is for the unpruned and pruned accuracy combined
    :return:
    """
    x = [i for i in range(0, len(y))]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if combined:
        plt.axvline(x=int(len(y)/2), color='r')  # plot the line half-way through the plot
    plt.plot(x, y)
    plt.show()


def main(saver, total_epochs=250, print_freq=1, model_path_unpruned="Model_Saves/Unpruned.ckpt",
         model_path_pruned="Model_Saves/Pruned.ckpt", sparsity=.5, learning_rate=1e-3, train_mode=False):
    """
    The main method, creates all the different ops and starts the training.
    The model starts by training for total_epochs, this is considered the pre-training, then model prunes and trains
    for another total_epochs
    :param total_epochs: total epochs to run the training or pruning
    :param print_freq: determines when to print the accuracy (per #epoch)
    :param model_path_unpruned: path to the unpruned model
    :param model_path_pruned: path to the pruned model
    :param sparsity: the target sparsity of the pruning
    :param learning_rate: learning rate of the optimizer
    :param train_mode: whether or not to train the model
    :return:
    """
    acc = []

    # Import dataset
    fashion_mnist = input_data.read_data_sets('data/fashion', one_hot=True)

    # Create global step variable (needed for pruning)
    global_step = tf.train.get_or_create_global_step()
    reset_global_step_op = tf.assign(global_step, 0)

    # Operators and functions
    loss = loss_fun()
    accuracy = accuracy_op()
    train_op = train(loss, global_step, lr=learning_rate)
    prune_op = pruning_params(global_step, target_sparsity=sparsity)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if train_mode:
            for epoch in range(total_epochs):
                sess.run(train_op, feed_dict={images: fashion_mnist.train.images,
                                              labels: fashion_mnist.train.labels})

                # Save the accuracy
                acc_print = sess.run(accuracy, feed_dict={images: fashion_mnist.test.images,
                                                          labels: fashion_mnist.test.labels})
                acc.append(acc_print)

                if epoch % print_freq == 0:
                    print("(Pre) Epoch: {}, accuracy: {}".format(epoch, acc_print))

            acc_print_pre = sess.run(accuracy, feed_dict={images: fashion_mnist.test.images,
                                                          labels: fashion_mnist.test.labels})

            print("Pre-pruning accuracy: {}".format(acc_print_pre))

            # Save the model before pruning starts
            saver.save(sess, model_path_unpruned)
        else:
            # Restore unpruned model
            saver.restore(sess, model_path_unpruned)

            # Reset the global step counter and begin pruning
            sess.run(reset_global_step_op)

            # Get the accuracy before pruning
            acc_print_pre = sess.run(accuracy, feed_dict={images: fashion_mnist.test.images,
                                                          labels: fashion_mnist.test.labels})

            # Prune and train the model
            for epoch in range(total_epochs):
                sess.run(train_op, feed_dict={images: fashion_mnist.train.images,
                                              labels: fashion_mnist.train.labels})
                sess.run(prune_op)

                # Save the accuracy
                acc_print = sess.run(accuracy, feed_dict={images: fashion_mnist.test.images,
                                                          labels: fashion_mnist.test.labels})
                acc.append(acc_print)

                if epoch % print_freq == 0:
                    print("Epoch: {}, accuracy: {}".format(epoch, acc_print))

            # Saves the model after pruning
            saver.save(sess, model_path_pruned + "_{}".format(sparsity*100))

            # Print final accuracy
            acc_print_final = sess.run(accuracy, feed_dict={images: fashion_mnist.test.images,
                                                            labels: fashion_mnist.test.labels})
            print("Pre-pruning accuracy: {}, post-pruning accuracy: {}".format(acc_print_pre, acc_print_final))
            print("Final sparsity by layer, expected: {}, actually: {}"
                  .format(sparsity, sess.run(tf.contrib.model_pruning.get_weight_sparsity())))
        return acc


if __name__ == "__main__":
    model_sparsities = [.10, .25, .50, .90]  # Target sparsity used for pruning
    train_lr = 1e-4  # Learning rate used for the optimizer
    epochs = 500  # Epochs to train before and after pruning
    print_every = 50  # Frequency of prints
    path_unpruned = "Model_Saves/Unpruned.ckpt"  # Path to save the unpruned model
    path_pruned = "Model_Saves/Pruned.ckpt"  # Path to save the pruned model
    saver = tf.train.Saver()

    unpruned_acc = main(saver, total_epochs=epochs, print_freq=print_every,
                       model_path_unpruned=path_unpruned, model_path_pruned=path_pruned,
                       learning_rate=train_lr, train_mode=True)
    plot(unpruned_acc, title='Fashion MNIST Unpruned')

    for model_sparsity in model_sparsities:
        pruned_acc = main(saver, total_epochs=epochs, print_freq=print_every,
                          model_path_unpruned=path_unpruned, model_path_pruned=path_pruned, sparsity=model_sparsity,
                          learning_rate=train_lr)
        plot(pruned_acc, title='Fashion MNIST Pruned (Sparsity: {})'.format(model_sparsity))
        plot(unpruned_acc + pruned_acc, title='Fashion MNIST (sparsity: {})'.format(model_sparsity), combined=True)
