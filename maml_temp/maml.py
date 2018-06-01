""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)

from tensorflow.python.platform import flags
from maml_temp.utils import mse, xent, conv_block, normalize
from maml_temp.vae_model import VAE

FLAGS = flags.FLAGS

class MAML:
    def __init__(self, dim_input=1, dim_output=1):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.classification = False
        if FLAGS.datasource == 'disfa':
            self.loss_func = xent
            self.classification = True
            self.loss_func = xent
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
            num_updates = FLAGS.num_updates # TODO max(self.test_num_updates, FLAGS.num_updates)
            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            accuraciesb = [[]]*num_updates

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                inputa = tf.reshape(inputa, [int(inputa.shape[0]), int(inputa.shape[1]),1])
                inputb = tf.reshape(inputb, [int(inputb.shape[0]), int(inputb.shape[1]),1])

                labela= tf.cast(labela, tf.float32)
                labela = tf.reshape(labela, [int(labela.shape[0]), 1, int(labela.shape[1])])
                labelb= tf.cast(labelb, tf.float32)
                labelb = tf.reshape(labelb, [int(labelb.shape[0]), 1, int(labelb.shape[1])])
                task_outputbs, task_lossesb, task_labelbs = [], [], []

                if self.classification:
                    task_accuraciesb = []

                task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter

                task_lossa = self.loss_func(task_outputa, labela) # 2,1

                grads = tf.gradients(task_lossa, list(weights.values())) # 2000,1,2
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gradients[key] for key in weights.keys()]))
                output = self.forward(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)
                task_labelbs.append(labelb)
                task_lossesb.append(self.loss_func(output, labelb))

                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in fast_weights.keys()]))
                    output = self.forward(inputb, fast_weights, reuse=True)
                    task_outputbs.append(output)
                    task_labelbs.append(labelb)
                    task_lossesb.append(self.loss_func(output, labelb))

                task_output = [task_outputa, task_outputbs, labela, task_labelbs, task_lossa, task_lossesb]

                if self.classification:
                    task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(labela, 1))
                    for j in range(num_updates):
                        task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), tf.argmax(labelb, 1)))
                    task_output.extend([task_accuracya, task_accuraciesb])

                return task_output

            # if FLAGS.norm is not 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
            unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)


            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]

            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)

            outputas, outputbs, res_labela, res_labelbs, lossesa, lossesb, accuraciesa, accuraciesb = result

        ## Performance & Optimization
        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs
            if self.classification:
                self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)

            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates-1])

                self.metatrain_op = optimizer.apply_gradients(gvs)
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            self.outputas, self.outputbs = outputas, outputbs
            self.res_labela, self.res_labelbs = res_labela, res_labelbs
            if self.classification:
                self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
                self.metaval_result1 = [outputas, res_labela]
                self.metaval_result2= [outputbs, res_labelbs]


        ## Summaries
        tf.summary.scalar(prefix+'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix+'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), total_accuracies2[j])


    def forward_fc(self, inp, weights, reuse=False):
        var_w = weights['w1'][ None, ::]
        # add dimension for features
        var_b = weights['b1'][ None, ::]
        # add dimension for output and class
        var_x = inp[ :, :, None]

        # matrix multiplication with dropout
        z = tf.reduce_sum( var_w*var_x , 1) + var_b
        score = tf.nn.softmax(z)
        return score



    def getWeightVar(self):
        w1 = tf.get_variable("w1",[2000,1,2])
        b1 = tf.get_variable("b1",[1,2])
        weight_tensor = {"w1": w1, "b1":b1}
        return weight_tensor

