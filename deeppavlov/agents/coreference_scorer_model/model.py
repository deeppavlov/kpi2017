import tensorflow as tf

class MentionScorerModel():
    def __init__(self, hidden_size=512, lr=0.0005, keep_prob_input=0.5, keep_prob_dense=0.8, features_size=455):
        self.keep_prob_input = keep_prob_input
        self.keep_prob_dense = keep_prob_dense
        self.lr = lr

        self.A = tf.placeholder(dtype=tf.float32, shape=(None, features_size), name='A')
        self.B = tf.placeholder(dtype=tf.float32, shape=(None, features_size), name='B')

        self.A_features = tf.placeholder(dtype=tf.int32, shape=(None, 5), name='A_features')
        self.B_features = tf.placeholder(dtype=tf.int32, shape=(None, 5), name='B_features')

        self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
        self.keep_prob_input_ph = tf.placeholder(dtype=tf.float32, shape=(None), name='keep_prob_input_ph')
        self.keep_prob_dense_ph = tf.placeholder(dtype=tf.float32, shape=(None), name='keep_prob_dense_ph')
        self.roc_auc = tf.placeholder(dtype=tf.float32, shape=(None), name='roc_auc_ph')

        A_do = tf.nn.dropout(self.A, keep_prob=self.keep_prob_input_ph)
        B_do = tf.nn.dropout(self.B, keep_prob=self.keep_prob_input_ph)
        #inputs = tf.concat([A_do, B_do, A_do * B_do], axis=1)
        inputs = tf.concat([A_do, B_do], axis=1)
        
        dense_1 = tf.layers.dense(inputs, units=hidden_size, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
        dense_1_do = tf.nn.dropout(dense_1, keep_prob=self.keep_prob_dense_ph)
        dense_2 = tf.layers.dense(dense_1_do, units=hidden_size, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
        dense_2_do = tf.nn.dropout(dense_2, keep_prob=self.keep_prob_dense_ph)
        self.logits = tf.layers.dense(dense_2_do, units=2, kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.pred = tf.nn.softmax(self.logits)

        lables_ohe = tf.one_hot(self.labels, depth=2)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=lables_ohe, logits=self.logits)
        self.loss = tf.reduce_mean(cross_entropy)
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.loss_test_summary = tf.summary.scalar("loss_test", self.loss)
        self.roc_auc_summary = tf.summary.scalar("roc_auc", self.roc_auc)
        
        self.train_op = tf.contrib.layers.optimize_loss(loss=self.loss, global_step=tf.contrib.framework.get_global_step(),
            learning_rate=self.lr, optimizer='Adam')

    def train_batch(self, session, A, B, labels):
        feed_dict = {
            self.A: A,
            self.B: B,
            self.labels: labels,
            self.keep_prob_input_ph: self.keep_prob_input,
            self.keep_prob_dense_ph: self.keep_prob_dense,
        }
        loss, loss_sum, logits, _ = session.run(
            [self.loss, self.loss_summary, self.logits, self.train_op],
            feed_dict=feed_dict)
        return loss, loss_sum, logits

    def test_batch(self, session, A, B, labels):
        feed_dict = {
            self.A: A,
            self.B: B,
            self.labels: labels,
            self.keep_prob_input_ph: 1.0,
            self.keep_prob_dense_ph: 1.0,
        }
        loss, loss_sum, logits, pred = session.run([self.loss, self.loss_test_summary, self.logits, self.pred],
            feed_dict=feed_dict)

        return loss, loss_sum, logits, pred