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
        self.total_num_au = 8
        self.num_classes = 2
        self.LAMBDA1 = FLAGS.lambda1
        self.LAMBDA2 = FLAGS.lambda2
        self.au_idx = -1
        if FLAGS.datasource == 'disfa':
            self.loss_func = xent
            self.loss_func2 = mse
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

            num_updates = FLAGS.num_updates  # TODO max(self.test_num_updates, FLAGS.num_updates)

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp  # input = (NK,latent_dim)  label = (NK, num of au, N), N = num of class
                inputa = tf.reshape(inputa, [int(inputa.shape[0]), int(inputa.shape[1]), 1])  # (NK,2000,1)
                inputb = tf.reshape(inputb, [int(inputb.shape[0]), int(inputb.shape[1]), 1])

                labela = tf.one_hot(labela, self.num_classes)  # (NK,8,2)
                labela = tf.cast(labela, tf.float32)[:, self.au_idx, :]  # (NK,1)
                labela = tf.reshape(labela, [int(labela.shape[0]), 1, self.num_classes])  # (NK,1,N)

                labelb = tf.one_hot(labelb, self.num_classes)  # (NK,2)
                labelb = tf.cast(labelb, tf.float32)[:, self.au_idx, :]
                labelb = tf.reshape(labelb, [int(labelb.shape[0]), 1, self.num_classes])  # (NK,1,N)




                this_w = weights['w1'][:, self.au_idx, :] # weights['w1'] = (300, 8,2)    this_w = (300,2)
                this_b = weights['b1'][self.au_idx, :]
                this_w = tf.reshape(this_w, [int(this_w.shape[0]), 1, int(this_w.shape[1])]) # (300,1,2)
                this_b = tf.reshape(this_b, [1, int(this_b.shape[0])])
                this_weight = {'w1': this_w, 'b1': this_b}
                # only reuse on the first iter: <<<previously meta-updated weight * input a>>>
                task_outputa = self.forward(inputa, this_weight, reuse=reuse)  # (NK, 1, 2)
                # ///////////////////////////////////////////////////////////////////////////
                task_lossa1 = self.loss_func(task_outputa, labela)  # 2,1
                task_lossa = task_lossa1
                # ///////////////////////////////////////////////////////////////////////////

                grads = tf.gradients(task_lossa, list(this_weight.values()))  # 2000,1,2
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(this_weight.keys(), grads))
                fast_weights = dict(
                    zip(this_weight.keys(),
                        [this_weight[key] - self.update_lr * gradients[key] for key in this_weight.keys()]))
                for j in range(num_updates - 1):
                    task_outputa = self.forward(inputa, this_weight, reuse=reuse)  # (NK, 1, 2)
                    # ///////////////////////////////////////////////////////////////////////////
                    loss1 = self.loss_func(task_outputa, labela)
                    loss = loss1
                    # ///////////////////////////////////////////////////////////////////////////

                    # compute gradients based on the previous fast weights
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    # fast_weights are updated
                    fast_weights = dict(zip(fast_weights.keys(),
                                            [fast_weights[key] - self.update_lr * gradients[key] for key in
                                             fast_weights.keys()]))

                outputb = self.forward(inputb, fast_weights, reuse=True)  # (2,1,2) = (2*k, # of au, onehot label)
                # outputb = outputb[:, 0,1] # choose the prob. of ON intensity from the softmax result to compare it with label '1'
                task_lossb = self.loss_func(outputb, labelb)
                task_output = [fast_weights['w1'], fast_weights['b1'], task_lossb]
                return task_output

            out_dtype_task_metalearn = [tf.float32, tf.float32, tf.float32]
            ##### inputa를 모든 au에 대해 다 받아온후 여기서 8등분해줘야함. 8등분 된 인풋별로 다음 for loop을 하나씩 걸쳐 매트릭스 건져냄
            batch = FLAGS.meta_batch_size
            # 매트릭스의 각 행은 각 au별 정보.
            w_matrix = []
            b_matrix = []
            ce_losses_of_inputb = []

            for i in range(self.total_num_au):
                self.au_idx = i
                inputa = tf.slice(self.inputa, [i * batch, 0, 0], [batch, -1, -1])  ##(aus*subjects, 2K, latent_dim)로부터 AU별로 #subjects 잘라냄 => (subjects, 2K, latent_dim)
                inputb = tf.slice(self.inputb, [i * batch, 0, 0], [batch, -1, -1])
                labela = tf.slice(self.labela, [i * batch, 0, 0], [batch, -1,
                                                                   -1])  # (aus*subjects, 2K, au, 2)로부터 AU별로 #subjects 잘라냄 => (subjects, 2K, au, 2)
                labelb = tf.slice(self.labelb, [i * batch, 0, 0], [batch, -1, -1])
                fast_weight_w, fast_weight_b, lossesb = tf.map_fn(task_metalearn,
                                                                  elems=(inputa, inputb, labela, labelb),
                                                                  dtype=out_dtype_task_metalearn,
                                                                  parallel_iterations=FLAGS.meta_batch_size)

                w_matrix.append(fast_weight_w)  # w_matrix = 8*14*(300*1*2)
                b_matrix.append(fast_weight_b)  # b_matrix = 8*14*(1*2)
                sum_loss_subjects = tf.reduce_sum(lossesb) / tf.to_float(FLAGS.meta_batch_size)  # lossesb = (14,NK,1)
                ce_losses_of_inputb.append(sum_loss_subjects)  # 8*14
            import numpy as np

            self.w_mat = tf.stack(w_matrix) # w_matrix = 8*14*(300*1*2)
            self.b_mat = tf.stack(b_matrix)# b_matrix = 8*14*(1*2)
            self.ce_losses = ce_losses_of_inputb

            def task_co_occur_loss(inp, reuse=True):
                inputb, labelb, this_au_subject_w, this_subject_ws, this_au_subject_b, this_subject_bs = inp  # this_au_subject_weight = (300,1,2)  this_subject_weights=(8,300,1,2)
                inputb = tf.reshape(inputb, [int(inputb.shape[0]), int(inputb.shape[1]), 1])
                labelb = tf.cast(labelb, tf.float32)

                this_au_subject_weight = {'w1': this_au_subject_w, 'b1': this_au_subject_b}

                losses = []
                for i in range(self.total_num_au):
                    other_au_subject_weight = {'w1': this_subject_ws[i], 'b1': this_subject_bs[i]}
                    pred_this_au = self.forward(inputb, this_au_subject_weight,
                                                reuse=reuse)  # (NK,1,2)
                    pred_this_au = tf.nn.softmax(pred_this_au)
                    pred_this_au = pred_this_au[:, 0,
                                   1]  ### choose the prob. of ON intensity from the softmax result to compare it with label '1' # (NK,)

                    pred_other_au = self.forward(inputb, other_au_subject_weight, reuse=reuse)
                    pred_other_au = tf.nn.softmax(pred_other_au)
                    pred_other_au = pred_other_au[:, 0, 1]

                    label_this_au = labelb[:, self.au_idx]  # (NK,)
                    label_other_au = labelb[:, i]

                    # sample 갯수만큼이 reduced sum된 per au and per subject의 loss가 생김
                    loss = self.loss_func2((-1 + 2 * pred_this_au) * (-1 + 2 * pred_other_au),
                                           (-1 + 2 * label_this_au) * (
                                           -1 + 2 * label_other_au))  # (num of samples=NK,1=num of au,2=N)

                    losses.append(loss)  # losses 는 현재 주어진 subject이, between 현재 주어진 au and 다른 모든 au간 이룬 loss들의 모임.

                task_output = [losses]
                return task_output

            out_dtype_task_occur_result = [[tf.float32] * self.total_num_au]
            # 매트릭스의 각 row = 각 au별로 au_global을 구해야함. 이때는 au_global간 크로스는 없지만, 매트릭스 전체가 모든 au마다 다쓰임
            # 로스를 포룹안에서 구하지 않고, 대신 이미주어져있는 inputb와 포룹으로 부터구한 매트릭스로 여기서부터 loss를 구하기시작

            all_co_occur_losses = []
            for i in range(self.total_num_au):
                self.au_idx = i
                inputb = tf.slice(self.inputb, [i * batch, 0, 0], [batch, -1, -1])
                labelb = tf.slice(self.labelb, [i * batch, 0, 0], [batch, -1, -1])
                this_au_weights = self.w_mat[i]  # 14*(300*1*2)
                transposed_w_mat = tf.transpose(self.w_mat, (1, 0, 2, 3, 4))  # 14*8*(300*1*2)
                this_au_biases = self.b_mat[i]  # 14*(300*1*2)
                transposed_b_mat = tf.transpose(self.b_mat, (1, 0, 2, 3))  # 14*8*(300*1*2)
                # 이번 포룹의 au에서, 14개의 subjects가 서로다른 aus와 이뤘던 co occur 에러. 따라서 14개의 값
                per_au_losses = tf.map_fn(task_co_occur_loss, elems=(
                inputb, labelb, this_au_weights, transposed_w_mat, this_au_biases, transposed_b_mat),
                                          dtype=out_dtype_task_occur_result, parallel_iterations=FLAGS.meta_batch_size)
                all_co_occur_losses.append(per_au_losses)  # 모든 au가 모든 au와 이루는 loss들. 모든 subjects에 대해. 다라서 8*14개



        ## Performance & Optimization
        # ce_loss = 8*14
        self.total_losses1 = [tf.reduce_sum(self.ce_losses[j]) / tf.to_float(FLAGS.meta_batch_size) for j in
                              range(self.total_num_au)]



        # 8*14개의 all_co_occur_losses에서, 각 au별 loss를 구함 by subject을 통틀어 합해버림으로써
        self.total_losses2 = [tf.reduce_sum(all_co_occur_losses[j]) / tf.to_float(FLAGS.meta_batch_size) for j in
                              range(self.total_num_au)]
        tf.summary.scalar('cross_entropy_0', self.total_losses1[0])
        tf.summary.scalar('cross_entropy_0', self.total_losses1[1])
        tf.summary.scalar('cross_entropy_0', self.total_losses1[2])
        tf.summary.scalar('cross_entropy_0', self.total_losses1[3])
        tf.summary.scalar('cross_entropy_0', self.total_losses1[4])
        tf.summary.scalar('cross_entropy_5', self.total_losses1[5])
        tf.summary.scalar('cross_entropy_5', self.total_losses1[6])
        tf.summary.scalar('cross_entropy_7', self.total_losses1[7])
        tf.summary.scalar('cross_entropy_total', tf.reduce_sum(self.total_losses1))
        tf.summary.scalar('co_occur_0', self.total_losses2[0])
        tf.summary.scalar('co_occur_0', self.total_losses2[1])
        tf.summary.scalar('co_occur_5', self.total_losses2[2])
        tf.summary.scalar('co_occur_7', self.total_losses2[3])
        tf.summary.scalar('co_occur_7', self.total_losses2[4])
        tf.summary.scalar('co_occur_5', self.total_losses2[5])
        tf.summary.scalar('co_occur_5', self.total_losses2[6])
        tf.summary.scalar('co_occur_7', self.total_losses2[7])
        tf.summary.scalar('co_occur_total', tf.reduce_sum(self.total_losses2))

        self.metatrain_op0 = tf.train.AdadeltaOptimizer(1.0).minimize(
            self.total_losses1[0] + self.LAMBDA2 * self.total_losses2[0])
        self.metatrain_op1 = tf.train.AdadeltaOptimizer(1.0).minimize(
            self.total_losses1[1] + self.LAMBDA2 * self.total_losses2[1])
        self.metatrain_op2 = tf.train.AdadeltaOptimizer(1.0).minimize(
            self.total_losses1[2] + self.LAMBDA2 * self.total_losses2[2])
        self.metatrain_op3 = tf.train.AdadeltaOptimizer(1.0).minimize(
            self.total_losses1[3] + self.LAMBDA2 * self.total_losses2[3])
        self.metatrain_op4 = tf.train.AdadeltaOptimizer(1.0).minimize(
            self.total_losses1[4] + self.LAMBDA2 * self.total_losses2[4])
        self.metatrain_op5 = tf.train.AdadeltaOptimizer(1.0).minimize(
            self.total_losses1[5] + self.LAMBDA2 * self.total_losses2[5])
        self.metatrain_op6 = tf.train.AdadeltaOptimizer(1.0).minimize(
            self.total_losses1[6] + self.LAMBDA2 * self.total_losses2[6])
        self.metatrain_op7 = tf.train.AdadeltaOptimizer(1.0).minimize(
            self.total_losses1[7] + self.LAMBDA2 * self.total_losses2[7])
        self.train_op = tf.group(self.metatrain_op0, self.metatrain_op1,self.metatrain_op2, self.metatrain_op3,self.metatrain_op4,self.metatrain_op5,self.metatrain_op6, self.metatrain_op7)

    def forward_fc(self, inp, weights, reuse=False):
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
