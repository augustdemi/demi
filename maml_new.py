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
        self.LAMBDA1 = 1
        self.au_idx = -1
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

            num_updates = FLAGS.num_updates  # TODO max(self.test_num_updates, FLAGS.num_updates)


            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp  # input = (NK,2000) label = (NK, N), N = num of class
                inputa = tf.reshape(inputa, [int(inputa.shape[0]), int(inputa.shape[1]), 1])  # (NK,2000,1)
                inputb = tf.reshape(inputb, [int(inputb.shape[0]), int(inputb.shape[1]), 1])

                labela = tf.cast(labela, tf.float32)
                labela = tf.reshape(labela, [int(labela.shape[0]), 1, int(labela.shape[1])])  # (NK,1,N)
                labelb = tf.cast(labelb, tf.float32)
                labelb = tf.reshape(labelb, [int(labelb.shape[0]), 1, int(labelb.shape[1])])

                this_weight = {'w1': weights['w1'][:, self.au_idx, :], 'b1': weights['b1'][self.au_idx, :]}
                # only reuse on the first iter: <<<previously meta-updated weight * input a>>>
                task_outputa = self.forward(inputa, this_weight, reuse=reuse)
                # ///////////////////////////////////////////////////////////////////////////
                task_lossa1 = self.loss_func(task_outputa, labela)  # 2,1
                pairwise_weight_avg = tf.reduce_sum(weights['w1'], 0) / self.total_num_au
                task_lossa2 = tf.nn.l2_loss(this_weight['w1'] - pairwise_weight_avg)
                task_lossa = task_lossa1 + self.LAMBDA1 * task_lossa2
                # ///////////////////////////////////////////////////////////////////////////

                grads = tf.gradients(task_lossa, list(this_weight.values()))  # 2000,1,2
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(this_weight.keys(), grads))
                fast_weights = dict(
                    zip(this_weight.keys(),
                        [this_weight[key] - self.update_lr * gradients[key] for key in this_weight.keys()]))
                for j in range(num_updates - 1):
                    # ///////////////////////////////////////////////////////////////////////////
                    loss1 = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)
                    pairwise_weight_avg = tf.reduce_sum(weights['w1'], 0) / self.total_num_au
                    loss2 = tf.nn.l2_loss(this_weight['w1'] - pairwise_weight_avg)
                    loss = loss1 + self.LAMBDA1 * loss2
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
                task_lossb = self.loss_func(outputb, labelb)
                task_output = [fast_weights['w1'], fast_weights['b1'], task_lossb]
                return task_output

            out_dtype_task_metalearn = [tf.float32, tf.float32, tf.float32]
            ##### inputa를 모든 au에 대해 다 받아온후 여기서 8등분해줘야함. 8등분 된 인풋별로 다음 for loop을 하나씩 걸쳐 매트릭스 건져냄
            # 매트릭스의 각 행은 각 au별 정보.
            w_matrix = []
            b_matrix = []
            ce_losses_of_inputb = []
            for i in range(8):
                batch = self.num_classes * FLAGS.update_batch_size
                inputa = tf.slice(self.inputa, [i * batch, 0, 0], [batch, -1, -1])  ##(NK,2000,1)로부터 AU별로 잘라냄
                inputb = tf.slice(self.inputb, [i * batch, 0, 0], [batch, -1, -1])  ##(NK,2000,1)로부터 AU별로 잘라냄
                labela = tf.slice(self.labela, [i * batch, 0, 0], [batch, -1, -1])  # (NK,1,N)로부터 AU별로 잘라냄
                labelb = tf.slice(self.labelb, [i * batch, 0, 0], [batch, -1, -1])  # (NK,1,N)로부터 AU별로 잘라냄
                self.au_idx = i
                sess = tf.Session()
                sess.run(tf.global_variables_initializer())

                this_weight = {'w1': weights['w1'][:, self.au_idx, :], 'b1': weights['b1'][self.au_idx, :]}
                w = this_weight['w1']
                b = this_weight['b1']
                bb = sess.run(b)
                ww = sess.run(ww)
                print(bb.shape)
                print(ww.shape)
                print('==============================')

                fast_weight_w, fast_weight_b, lossesb = tf.map_fn(task_metalearn,
                                                                  elems=(inputa, inputb, labela, labelb),
                                                                  dtype=out_dtype_task_metalearn,
                                                                  parallel_iterations=FLAGS.meta_batch_size)

                w_matrix.append(fast_weight_w)  # w_matrix = 8*14*(300*1)
                b_matrix.append(fast_weight_b)  # b_matrix = 8*14*2
                sum_loss_subjects = tf.reduce_sum(lossesb) / tf.to_float(FLAGS.meta_batch_size)  # lossesb = (14,NK,1)
                ce_losses_of_inputb.append(sum_loss_subjects)  # 8*14
            self.w_mat = w_matrix
            self.b_mat = b_matrix
            self.ce_losses = ce_losses_of_inputb

        # def task_occur_result(inp, reuse=True):
        #     inputb, labelb, au_idx, sub_idx = inp  # input = (NK,2000) label = (NK, N), N = num of class, this_w = 1*14(one au*subjects)
        #     inputb = tf.reshape(inputb, [int(inputb.shape[0]), int(inputb.shape[1]), 1])
        #     labelb = tf.cast(labelb, tf.float32)
        #     labelb = tf.reshape(labelb, [int(labelb.shape[0]), 1, int(labelb.shape[1])])
        #
        #     task_outputa = self.forward(inputa, weights, reuse=reuse)
        #     # ///////////////////////////////////////////////////////////////////////////
        #     task_lossa1 = self.loss_func(task_outputa, labela)  # 2,1 // loss 1 추가되어야
        #     # ///////////////////////////////////////////////////////////////////////////
        #     task_output = [fast_weights['w1'], fast_weights['b1']]
        #     return task_output

        out_dtype_task_occur_result = [tf.float32, tf.float32, tf.float32]
        # 매트릭스의 각 row = 각 au별로 au_global을 구해야함. 이때는 au_global간 크로스는 없지만, 매트릭스 전체가 모든 au마다 다쓰임
        # 로스를 포룹안에서 구하지 않고, 대신 이미주어져있는 inputb와 포룹으로 부터구한 매트릭스로 여기서부터 loss를 구하기시작
        sub_idx = tf.constant(list(range(FLAGS.meta_batch_size)), dtype=tf.int64)
        # for i in range(8):
        #     batch = self.total_num_au * self.num_classes * FLAGS.update_batch_size
        #     inputb = tf.slice(self.inputb, [i*batch, 0, 0], [(i+1)*batch, -1, -1])  ##(NK,2000,1)로부터 AU별로 잘라냄
        #     labelb = tf.slice(self.labelb, [i*batch, 0, 0], [(i+1)*batch, -1, -1])  #(NK,1,N)로부터 AU별로 잘라냄
        #     au_idx =  tf.constant([i] * FLAGS.meta_batch_size, dtype=tf.int64)
        #     result= tf.map_fn(task_occur_result, elems=(inputb, labelb, au_idx, sub_idx),
        #                                              dtype=out_dtype_task_occur_result, parallel_iterations=FLAGS.meta_batch_size)
        #     output = self.forward(inputb, w_matrix[i], reuse=True)  # (2,1,2) = (2*k, # of au, onehot label)
        #     self.loss_func(output, labelb)


        ## Performance & Optimization
        self.lossesa = lossesa  # (meta_batch_size, NK, 1)
        self.total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
        # ce_loss = 8*14
        self.total_losses2 = [tf.reduce_sum(self.ce_losses[j]) / tf.to_float(FLAGS.meta_batch_size) for j in
                              range(self.total_num_au)]

        # after the map_fn
        # def optimize_all_au():
        #     for i in range(8):
        #         tf.train.AdadeltaOptimizer(1.0).minimize(self.total_losses2[i])

        self.metatrain_op = tf.train.AdadeltaOptimizer(1.0).minimize(self.total_losses2[0])
        self.metatrain_op = tf.train.AdadeltaOptimizer(1.0).minimize(self.total_losses2[1])

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
        w1 = tf.get_variable("w1", [self.weight_dim, self.total_num_au, 2],
                             initializer=tf.contrib.layers.xavier_initializer(dtype=dtype))
        b1 = tf.get_variable("b1", [self.total_num_au, 2], initializer=tf.zeros_initializer())
        weight_tensor = {"w1": w1, "b1": b1}
        return weight_tensor
