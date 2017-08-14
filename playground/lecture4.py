import os
import playground.process_data as process_data
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128  # Dimension of the word embedding vectors
SKIP_WINDOW = 1  # Context window
NUM_SAMPLED = 64  # Number of negative examples to sample
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 10000
SKIP_STEP = 2000  # How many steps to skip before reporting the loass


def word2vec(batch_gen):
    """ Build the graph for word2vec and train it."""

    # Step 1: Define the placeholders for the input and output
    with tf.name_scope('data'):
        center_words = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE], name='center_words')
        target_words = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE, 1], name='target_words')

    # Step 2: Define weights. In word2vec, it's actually the weights that we care about
    with tf.name_scope('embedding_matrix'):
        embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0), name='embed_matrix')

    # Step 3: Define the inference
    with tf.name_scope('loss'):
        embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')

        # Step 4: Construct variables for the NCE loss
        nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], stddev=1.0/(EMBED_SIZE ** 0.5)),
                                 name='nce_weight')

        nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')

        # Define loss function to be NCE loss function
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                             biases=nce_bias,
                                             labels=target_words,
                                             inputs=embed,
                                             num_sampled=NUM_SAMPLED,
                                             num_classes=VOCAB_SIZE), name='loss')

    # Step 5: Define the Optimizer
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_loss = 0.0  # We use this to calculate average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('./graphs/no_frills', sess.graph)
        for index in range(NUM_TRAIN_STEPS):
            centers, targets = next(batch_gen)
            loss_batch, _ = sess.run([loss, optimizer], feed_dict={center_words: centers, target_words: targets})
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss/SKIP_STEP))
                total_loss = 0.0
        writer.close()


def main():
    batch_gen = process_data.process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    word2vec(batch_gen)

if __name__ == '__main__':
    main()
