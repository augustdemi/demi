""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import sys
import tensorflow as tf

try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)

from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, normalize

FLAGS = flags.FLAGS


class MAML:
    def __init__(self, dim_input=1, dim_output=1):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.classification = False
        self.weight_dim = 300
        if FLAGS.datasource == 'disfa':
            self.loss_func = xent
            self.classification = True
            self.forward = self.forward_fc
            self.construct_weights = self.getWeightVar
        else:
            raise ValueError('Unrecognized data source.')

    def construct_model(self, input_tensors=None, prefix='metatrain_'):
        # a: training data for inner gradient, b: test data for meta gradient
        if input_tensors is None:
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
        else:
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            num_updates = FLAGS.num_updates  # TODO max(self.test_num_updates, FLAGS.num_updates)
            outputbs = [[]] * num_updates
            lossesb = [[]] * num_updates
            accuraciesb = [[]] * num_updates

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                inputa = tf.reshape(inputa, [int(inputa.shape[0]), int(inputa.shape[1]), 1])
                inputb = tf.reshape(inputb, [int(inputb.shape[0]), int(inputb.shape[1]), 1])

                labela = tf.cast(labela, tf.float32)
                labela = tf.reshape(labela, [int(labela.shape[0]), 1, int(labela.shape[1])])
                labelb = tf.cast(labelb, tf.float32)
                labelb = tf.reshape(labelb, [int(labelb.shape[0]), 1, int(labelb.shape[1])])
                task_outputbs, task_lossesb, task_labelbs = [], [], []
                all_w, all_b = [], []

                if self.classification:
                    task_accuraciesb = []

                task_outputa = self.forward(inputa, weights,
                                            reuse=reuse)  # only reuse on the first iter: <<<previously meta-updated weight * input a>>>

                task_lossa = self.loss_func(task_outputa, labela)  # 2,1

                grads = tf.gradients(task_lossa, list(weights.values()))  # 2000,1,2
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(
                    zip(weights.keys(), [weights[key] - self.update_lr * gradients[key] for key in weights.keys()]))
                output = self.forward(inputb, fast_weights, reuse=True)  # <<< fast weight * inputb >>>
                task_outputbs.append(output)
                task_labelbs.append(labelb)
                task_lossesb.append(self.loss_func(output, labelb))
                all_w.append(fast_weights['w1'])
                all_b.append(fast_weights['b1'])

                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)
                    # compute gradients based on the previous fast weights
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    # fast_weights are updated
                    fast_weights = dict(zip(fast_weights.keys(),
                                            [fast_weights[key] - self.update_lr * gradients[key] for key in
                                             fast_weights.keys()]))
                    output = self.forward(inputb, fast_weights, reuse=True)  # (2,1,2) = (2*k, # of au, onehot label)
                    task_outputbs.append(output)
                    task_labelbs.append(labelb)
                    task_lossesb.append(self.loss_func(output, labelb))
                    all_w.append(fast_weights['w1'])
                    all_b.append(fast_weights['b1'])

                task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1),
                                                             tf.argmax(labela, 1))
                for j in range(num_updates):
                    task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1),
                                                                        tf.argmax(labelb, 1)))

                task_output = [task_outputa, task_outputbs, labela, task_labelbs, task_lossa, task_lossesb,
                               task_accuracya, task_accuraciesb, all_w, all_b,
                               [fast_weights['w1'], fast_weights['b1']]]
                return task_output

            out_dtype = [tf.float32, [tf.float32] * num_updates, tf.float32, [tf.float32] * num_updates, tf.float32,
                         [tf.float32] * num_updates, tf.float32, [tf.float32] * num_updates,
                         [tf.float32] * num_updates, [tf.float32] * num_updates,
                         [tf.float32, tf.float32]]

            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb),
                               dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            # In result, outa has shape (1,2,1,2) = (num.of.task, 2*k, num.of.au, one-hot label)
            outputas, outputbs, res_labela, res_labelbs, lossesa, lossesb, accuraciesa, accuraciesb, all_w, all_b, fast_weights = result

        ## Performance & Optimization
        if 'train' in prefix:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            w = weights['w1']
            b = weights['b1']
            bb = sess.run(b)
            ww = sess.run(w)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

            print(bb.shape)
            print(ww.shape)
            print(sess.run(tf.reshape(self.inputa, [int(self.inputa.shape[0]), int(self.inputa.shape[1]), 1])).shape)
            task_outputa = self.forward(
                tf.reshape(self.inputa, [int(self.inputa.shape[0]), int(self.inputa.shape[1]), 1]), weights)
            print(sess.run(task_outputa).shape)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

            self.lossesa = lossesa
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j
                                                  in range(num_updates)]
            # after the map_fn
            self.outputas, self.outputbs, self.all_w, self.all_b, self.fast_weights = outputas, outputbs, all_w, all_b, fast_weights
            if self.classification:
                self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.total_accuracies2 = total_accuracies2 = [
                    tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
                self.result1 = [outputas, res_labela]
                self.result2 = [outputbs, res_labelbs]
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)
            # self.pretrain_op = tf.train.AdadeltaOptimizer(1.0).minimize(total_loss1)

            if FLAGS.metatrain_iterations > 0:
                self.metatrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(
                    self.total_losses2[FLAGS.num_updates - 1])
                # self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates - 1])
                # self.metatrain_op = optimizer.apply_gradients(gvs)
                # self.metatrain_op = tf.train.AdadeltaOptimizer(1.0).minimize(self.total_losses2[FLAGS.num_updates - 1])
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size)
                                                          for j in range(num_updates)]
            self.outputas, self.outputbs = outputas, outputbs
            self.res_labela, self.res_labelbs = res_labela, res_labelbs
            if self.classification:
                self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(
                    FLAGS.meta_batch_size)
                self.metaval_total_accuracies2 = total_accuracies2 = [
                    tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
                self.metaval_result1 = [outputas, res_labela]
                self.metaval_result2 = [outputbs, res_labelbs]

        ## Summaries
        tf.summary.scalar(prefix + 'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix + 'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):
            tf.summary.scalar(prefix + 'Post-update loss, step ' + str(j + 1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix + 'Post-update accuracy, step ' + str(j + 1), total_accuracies2[j])

    def forward_fc(self, inp, weights, reuse=False):
        var_w = weights['w1'][None, ::]
        # add dimension for features
        var_b = weights['b1'][None, ::]
        # add dimension for output and class
        var_x = inp[:, :, None]

        # matrix multiplication with dropout
        z = tf.reduce_sum(var_w * var_x, 1) + var_b
        # normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
        score = tf.nn.softmax(z)
        return score

    def getWeightVar(self):
        tf.set_random_seed(FLAGS.weight_seed)
        # w1 = tf.Variable(tf.truncated_normal([self.weight_dim, 1, 2], stddev=0.01), name="w1")
        # b1 = tf.Variable(tf.zeros([1, 2]), name="b1")
        dtype = tf.float32
        w1 = tf.get_variable("w1", [self.weight_dim, 1, 2],
                             initializer=tf.contrib.layers.xavier_initializer(dtype=dtype))
        b1 = tf.get_variable("b1", [1, 2], initializer=tf.zeros_initializer())
        weight_tensor = {"w1": w1, "b1": b1}
        return weight_tensor
