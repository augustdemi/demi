""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np

try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)

from tensorflow.python.platform import flags
from utils import mse, xent, scaling

FLAGS = flags.FLAGS


class MAML:
    def __init__(self, dim_input=1, dim_output=1):
        """ must call construct_model() after initializing MAML! """

        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = FLAGS.meta_lr
        self.classification = False
        self.weight_dim = 300
        self.total_num_au = 8
        self.num_classes = 2
        self.LAMBDA2 = FLAGS.lambda2
        self.au_idx = -1
        self.loss_func = xent
        self.loss_func2 = mse
        self.scaling = scaling
        self.classification = True
        self.forward = self.forward_fc
        self.construct_weights = self.getWeightVar

    def construct_model(self, input_tensors=None):
        # a: training data for inner gradient, b: test data for meta gradient
        self.inputa = input_tensors['inputa']
        self.inputb = input_tensors['inputb']
        self.labela = input_tensors['labela']
        self.labelb = input_tensors['labelb']
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []

            num_updates = FLAGS.num_updates  # TODO max(self.test_num_updates, FLAGS.num_updates)

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp  # input = (NK,latent_dim)  label = (NK, num of au, N), N = num of class
                inputa = tf.reshape(inputa, [int(inputa.shape[0]), int(inputa.shape[1]), 1])  # (NK,2000,1)
                inputb = tf.reshape(inputb, [int(inputb.shape[0]), int(inputb.shape[1]), 1])
                inputa = tf.cast(inputa, tf.float32)
                inputb = tf.cast(inputb, tf.float32)

                labela_ce = tf.one_hot(labela, self.num_classes)  # (NK,8,2)
                labela_ce = tf.cast(labela_ce, tf.float32)[:, self.au_idx, :]  # (NK,1)
                labela_ce = tf.reshape(labela_ce, [int(labela_ce.shape[0]), 1, self.num_classes])  # (NK,1,N)

                labelb_ce = tf.one_hot(labelb, self.num_classes)  # (NK,2)
                labelb_ce = tf.cast(labelb_ce, tf.float32)[:, self.au_idx, :]
                labelb_ce = tf.reshape(labelb_ce, [int(labelb_ce.shape[0]), 1, self.num_classes])  # (NK,1,N)

                this_w = weights['w1'][:, self.au_idx, :]  # weights['w1'] = (300, 8,2)    this_w = (300,2)
                this_b = weights['b1'][self.au_idx, :]
                this_w = tf.reshape(this_w, [int(this_w.shape[0]), 1, int(this_w.shape[1])])  # (300,1,2)
                this_b = tf.reshape(this_b, [1, int(this_b.shape[0])])
                this_weight = {'w1': this_w, 'b1': this_b}
                # only reuse on the first iter: <<<previously meta-updated weight * input a>>>
                task_outputa = self.forward(inputa, this_weight)  # (NK, 1, 2)

                ##### for cross-entropy loss ####
                task_ce_lossa = self.loss_func(task_outputa, labela_ce)[:, 0]  # 2,1

                ##### for co-occur loss ####
                def predict_other_au(i, input, label):
                    other_w = weights['w1'][:, i, :]  # weights['w1'] = (300, 8,2)    this_w = (300,2)
                    other_b = weights['b1'][i, :]
                    other_w = tf.reshape(other_w, [int(other_w.shape[0]), 1, int(other_w.shape[1])])  # (300,1,2)
                    other_b = tf.reshape(other_b, [1, int(other_b.shape[0])])
                    other_weight = {'w1': other_w, 'b1': other_b}
                    pred_other_au = self.forward(input, other_weight)
                    pred_other_au = tf.nn.softmax(pred_other_au)
                    pred_other_au = pred_other_au[:, 0, 1]
                    label_other_au = tf.cast(label[:, i], tf.float32)
                    return [pred_other_au, label_other_au]

                pred_this_au = tf.nn.softmax(task_outputa)
                pred_this_au = pred_this_au[:, 0, 1]
                labela_this_au = tf.cast(labela[:, self.au_idx], tf.float32)  # (NK,)

                task_co_lossa = []
                for i in range(self.total_num_au):
                    pred_other_au, label_other_au = predict_other_au(i, inputa, labela)
                    loss = self.loss_func2(pred_this_au * pred_other_au,
                                           labela_this_au * label_other_au)  # (num of samples=NK,1=num of au,2=N)
                    # losses 는 현재 주어진 subject이, between 현재 주어진 au and 다른 모든 au간 이룬 loss들의 모임.
                    task_co_lossa.append(loss)
                task_co_lossa = tf.reduce_sum(task_co_lossa, 0) / self.total_num_au
                task_lossa = task_ce_lossa + self.LAMBDA2 * task_co_lossa

                grads = tf.gradients(task_lossa, list(this_weight.values()))  # 300,1,2
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(this_weight.keys(), grads))
                fast_weights = dict(
                    zip(this_weight.keys(),
                        [this_weight[key] - self.update_lr * gradients[key] for key in this_weight.keys()]))

                ### for more inner update ###
                for j in range(num_updates - 1):

                    task_outputa = self.forward(inputa, fast_weights)
                    task_ce_loss = self.loss_func(task_outputa, labela_ce)[:, 0]
                    ### co-occur ###
                    task_co_loss = []
                    pred_this_au = tf.nn.softmax(task_outputa)
                    pred_this_au = pred_this_au[:, 0, 1]

                    for i in range(self.total_num_au):
                        pred_other_au, label_other_au = predict_other_au(i, inputa, labela)
                        task_co_loss.append(self.loss_func2(pred_this_au * pred_other_au,
                                                            labela_this_au * label_other_au))

                    task_co_loss = tf.reduce_sum(task_co_loss, 0) / self.total_num_au
                    task_loss = task_ce_loss + self.LAMBDA2 * task_co_loss
                    # compute gradients based on the previous fast weights
                    grads = tf.gradients(task_loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    # fast_weights are updated
                    fast_weights = dict(zip(fast_weights.keys(),
                                            [fast_weights[key] - self.update_lr * gradients[key] for key in
                                             fast_weights.keys()]))

                #### make evaluation with inputb ####
                outputb = self.forward(inputb, fast_weights)  # (2,1,2) = (2*k, # of au, onehot label)
                task_ce_lossb = self.loss_func(outputb, labelb_ce)[:, 0]
                ### for co-occur loss ###
                labelb_this_au = tf.cast(labelb[:, self.au_idx], tf.float32)  # (NK,)
                predb_this_au = tf.nn.softmax(outputb)
                predb_this_au = predb_this_au[:, 0, 1]
                task_co_lossb = []
                for i in range(self.total_num_au):
                    predb_other_au, labelb_other_au = predict_other_au(i, inputb, labelb)
                    task_co_lossb.append(self.loss_func2(predb_this_au * predb_other_au,
                                                         labelb_this_au * labelb_other_au))
                    # test_other_au.append(labelb_other_au)
                # test_other_au = tf.convert_to_tensor(test_other_au)
                task_co_lossb = tf.reduce_sum(task_co_lossb, 0) / self.total_num_au
                task_total = task_ce_lossb + self.LAMBDA2 * task_co_lossb
                ### return output ###
                task_output = [fast_weights['w1'], fast_weights['b1'], task_ce_lossb, task_co_lossb, task_total,
                               predb_this_au]
                return task_output

            out_dtype_task_metalearn = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
            ##### inputa를 모든 au에 대해 다 받아온후 여기서 8등분해줘야함. 8등분 된 인풋별로 다음 for loop을 하나씩 걸쳐 매트릭스 건져냄
            batch = FLAGS.meta_batch_size
            self.task_ce_losses = []
            self.task_co_losses = []
            self.task_total_losses = []
            self.fast_weight_w = []
            self.fast_weight_b = []
            for i in range(self.total_num_au):
                self.au_idx = i
                inputa = tf.slice(self.inputa, [i * batch, 0, 0], [batch, -1,
                                                                   -1])  ##(aus*subjects, 2K, latent_dim)로부터 AU별로 #subjects 잘라냄 => (subjects, 2K, latent_dim)
                inputb = tf.slice(self.inputb, [i * batch, 0, 0], [batch, -1, -1])
                labela = tf.slice(self.labela, [i * batch, 0, 0], [batch, -1,
                                                                   -1])  # (aus*subjects, 2K, au, 2)로부터 AU별로 #subjects 잘라냄 => (subjects, 2K, au, 2)
                labelb = tf.slice(self.labelb, [i * batch, 0, 0], [batch, -1, -1])
                print("=========================================================")
                print("used inputa shape in au_idx {} : {}".format(i, inputa.shape))
                print("used labela shape in au_idx {} : {}".format(i, labela.shape))
                print("=========================================================")
                fast_weight_w, fast_weight_b, ce_lossesb, co_lossesb, total_lossesb, predict_b = tf.map_fn(
                    task_metalearn,
                    elems=(inputa, inputb, labela, labelb),
                    dtype=out_dtype_task_metalearn,
                    parallel_iterations=FLAGS.meta_batch_size)
                try:
                    self.fast_weight_w = np.append(self.fast_weight_w, fast_weight_w, axis=2)
                    self.fast_weight_b = np.append(self.fast_weight_b, fast_weight_b, axis=2)
                except:
                    self.fast_weight_w = fast_weight_w # 14 * 300*1*2
                    self.fast_weight_b = fast_weight_b
                self.task_ce_losses.append(ce_lossesb)
                self.task_co_losses.append(co_lossesb)  # 8*14
                self.task_total_losses.append(total_lossesb)  # 8*14
        # 8*14 --> 8*1 (make each 1*14 into 1*1)
        self.total_losses = [tf.reduce_sum(self.task_total_losses[k]) / tf.to_float(FLAGS.meta_batch_size) for k in
                             range(self.total_num_au)]

        # self.val_loss =

        ## Performance & Optimization
        tf.summary.scalar('CE_AU1', tf.reduce_sum(self.task_ce_losses[0]) / tf.to_float(FLAGS.meta_batch_size))
        tf.summary.scalar('CE_AU2', tf.reduce_sum(self.task_ce_losses[1]) / tf.to_float(FLAGS.meta_batch_size))
        tf.summary.scalar('CE_AU4', tf.reduce_sum(self.task_ce_losses[2]) / tf.to_float(FLAGS.meta_batch_size))
        tf.summary.scalar('CE_AU6', tf.reduce_sum(self.task_ce_losses[3]) / tf.to_float(FLAGS.meta_batch_size))
        tf.summary.scalar('CE_AU9', tf.reduce_sum(self.task_ce_losses[4]) / tf.to_float(FLAGS.meta_batch_size))
        tf.summary.scalar('CE_AU12', tf.reduce_sum(self.task_ce_losses[5]) / tf.to_float(FLAGS.meta_batch_size))
        tf.summary.scalar('CE_AU25', tf.reduce_sum(self.task_ce_losses[6]) / tf.to_float(FLAGS.meta_batch_size))
        tf.summary.scalar('CE_AU26', tf.reduce_sum(self.task_ce_losses[7]) / tf.to_float(FLAGS.meta_batch_size))
        tf.summary.scalar('CE_total', tf.reduce_sum(
            [tf.reduce_sum(self.task_ce_losses[k]) / tf.to_float(FLAGS.meta_batch_size) for k in
             range(self.total_num_au)]) / tf.to_float(self.total_num_au))

        tf.summary.scalar('co_occur_AU1', tf.reduce_sum(self.task_co_losses[0]) / tf.to_float(FLAGS.meta_batch_size))
        tf.summary.scalar('co_occur_AU2', tf.reduce_sum(self.task_co_losses[1]) / tf.to_float(FLAGS.meta_batch_size))
        tf.summary.scalar('co_occur_AU4', tf.reduce_sum(self.task_co_losses[2]) / tf.to_float(FLAGS.meta_batch_size))
        tf.summary.scalar('co_occur_AU6', tf.reduce_sum(self.task_co_losses[3]) / tf.to_float(FLAGS.meta_batch_size))
        tf.summary.scalar('co_occur_AU9', tf.reduce_sum(self.task_co_losses[4]) / tf.to_float(FLAGS.meta_batch_size))
        tf.summary.scalar('co_occur_AU12', tf.reduce_sum(self.task_co_losses[5]) / tf.to_float(FLAGS.meta_batch_size))
        tf.summary.scalar('co_occur_AU25', tf.reduce_sum(self.task_co_losses[6]) / tf.to_float(FLAGS.meta_batch_size))
        tf.summary.scalar('co_occur_AU26', tf.reduce_sum(self.task_co_losses[7]) / tf.to_float(FLAGS.meta_batch_size))
        tf.summary.scalar('co_occur_total', tf.reduce_sum(
            [tf.reduce_sum(self.task_co_losses[k]) / tf.to_float(FLAGS.meta_batch_size) for k in
             range(self.total_num_au)]) / tf.to_float(self.total_num_au))

        tf.summary.scalar('co+ce_AU1', self.total_losses[0])
        tf.summary.scalar('co+ce_AU2', self.total_losses[1])
        tf.summary.scalar('co+ce_AU4', self.total_losses[2])
        tf.summary.scalar('co+ce_AU6', self.total_losses[3])
        tf.summary.scalar('co+ce_AU9', self.total_losses[4])
        tf.summary.scalar('co+ce_AU12', self.total_losses[5])
        tf.summary.scalar('co+ce_AU25', self.total_losses[6])
        tf.summary.scalar('co+ce_AU26', self.total_losses[7])
        tf.summary.scalar('co+ce_total', tf.reduce_sum(self.total_losses) / tf.to_float(self.total_num_au))

        if FLAGS.opti.startswith('adadelta'):
            print('------------- optimized with ADADELTA')
            # self.metatrain_op0 = tf.train.AdadeltaOptimizer(1.0).minimize(self.total_losses[0])
            # self.metatrain_op1 = tf.train.AdadeltaOptimizer(1.0).minimize(self.total_losses[1])
            # self.metatrain_op2 = tf.train.AdadeltaOptimizer(1.0).minimize(self.total_losses[2])
            # self.metatrain_op3 = tf.train.AdadeltaOptimizer(1.0).minimize(self.total_losses[3])
            # self.metatrain_op4 = tf.train.AdadeltaOptimizer(1.0).minimize(self.total_losses[4])
            # self.metatrain_op5 = tf.train.AdadeltaOptimizer(1.0).minimize(self.total_losses[5])
            # self.metatrain_op6 = tf.train.AdadeltaOptimizer(1.0).minimize(self.total_losses[6])
            # self.metatrain_op7 = tf.train.AdadeltaOptimizer(1.0).minimize(self.total_losses[7])
            total_loss = tf.reduce_sum(self.total_losses) / self.total_num_au
            self.metatrain_op = tf.train.AdadeltaOptimizer(1.0).minimize(total_loss)
        elif FLAGS.opti.startswith('adam'):
            print('------------- optimized with ADAM - lr: ', self.meta_lr)
            # self.metatrain_op0 = tf.train.AdamOptimizer(self.meta_lr).minimize(self.total_losses[0])
            # self.metatrain_op1 = tf.train.AdamOptimizer(self.meta_lr).minimize(self.total_losses[1])
            # self.metatrain_op2 = tf.train.AdamOptimizer(self.meta_lr).minimize(self.total_losses[2])
            # self.metatrain_op3 = tf.train.AdamOptimizer(self.meta_lr).minimize(self.total_losses[3])
            # self.metatrain_op4 = tf.train.AdamOptimizer(self.meta_lr).minimize(self.total_losses[4])
            # self.metatrain_op5 = tf.train.AdamOptimizer(self.meta_lr).minimize(self.total_losses[5])
            # self.metatrain_op6 = tf.train.AdamOptimizer(self.meta_lr).minimize(self.total_losses[6])
            # self.metatrain_op7 = tf.train.AdamOptimizer(self.meta_lr).minimize(self.total_losses[7])
            total_loss = tf.reduce_sum(self.total_losses) / self.total_num_au
            self.metatrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss)

        else:
            print('------------- optimizer should be adam or adadelta but given: ', FLAGS.opti)
        self.train_op = self.metatrain_op
        # self.train_op = tf.group(self.metatrain_op0, self.metatrain_op1, self.metatrain_op2, self.metatrain_op3,
        #                          self.metatrain_op4, self.metatrain_op5, self.metatrain_op6, self.metatrain_op7)

    def forward_fc(self, inp, weights):
        var_w = weights['w1'][None, ::]
        # add dimension for features
        var_b = weights['b1'][None, ::]
        # add dimension for output and class
        var_x = inp[:, :, None]

        # matrix multiplication with dropout

        z = tf.reduce_sum(var_w * var_x, 1) + var_b
        # normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
        # score = tf.nn.softmax(z)
        return z

    def getWeightVar(self):
        tf.set_random_seed(FLAGS.weight_seed)
        # w1 = tf.Variable(tf.truncated_normal([self.weight_dim, 1, 2], stddev=0.01), name="w1")
        # b1 = tf.Variable(tf.zeros([1, 2]), name="b1")
        dtype = tf.float32
        w1 = tf.get_variable("w1", [self.weight_dim, self.total_num_au, 2],
                             initializer=tf.contrib.layers.xavier_initializer(dtype=dtype))
        b1 = tf.get_variable("b1", [self.total_num_au, 2], initializer=tf.zeros_initializer())
        weight_tensor = {"w1": w1, "b1": b1}
        return weight_tensor
