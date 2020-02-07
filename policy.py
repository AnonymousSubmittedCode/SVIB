import datetime
import os
import sys

import gym
import joblib
import tensorflow as tf
import numpy as np
import argparse
import os.path as osp
import time
from baselines import logger
from runner import Runner_explore
# from baselines.a2c.a2c import Runner
from baselines.a2c.utils import conv, lstm, lnlstm, conv_to_fc, fc, ortho_init, cat_entropy, find_trainable_variables, Scheduler, make_path, batch_to_seq, seq_to_batch
from baselines.common import set_global_seeds, tf_util
from baselines.common.cmd_util import make_atari_env
from baselines.common.distributions import make_pdtype
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat
from gym.wrappers import Monitor
from tensorflow import sqrt
from tensorflow.contrib.solvers.python.ops.util import l2norm
from utils import grad_clip, batch_norm, explained_variance, conv_unequ_size, deconv_unequ_size, deconv, noise_and_argmax, dense_layers, generate_noise, tf_normalize, mse
ds = tf.contrib.distributions

class LstmPolicyEm(object):
    '''
    HRL LSTM policy with em algorithm , note that the goal in this policy is the abstraction of state
    '''
    def __init__(self, sess, ob_space, ac_space, nbatch, master_ts, worker_ts, cell=256, reuse=tf.AUTO_REUSE, use_vae=True, phase='vae'):

        nenv = nbatch // (master_ts * worker_ts)
        nh, nw, nc = ob_space.shape
        nact = ac_space.n
        self.use_vae = use_vae
        self.nenv = nenv

        wX = tf.placeholder(tf.uint8, [nbatch, nh, nw, nc])#worker obs
        wS = tf.placeholder(tf.float32, [nenv * master_ts, cell * 2])#worker initial state
        wM = tf.placeholder(tf.float32, [nbatch])#worker mask
        G = tf.placeholder(tf.float32, [nbatch, cell])#goal set by master

        mX = tf.placeholder(tf.uint8, [nenv * master_ts, nh, nw, nc])#master obs
        mS = tf.placeholder(tf.float32, [nenv, cell * 2])#master initial state
        mM = tf.placeholder(tf.float32, [nenv * master_ts]) #master mask
        mA = tf.placeholder(tf.float32, [nenv * master_ts, cell]) #master's action

        Z = tf.placeholder(tf.float32, [None, cell])

        istraining = tf.placeholder(tf.bool, shape=[])

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu
            wX_ = tf.cast(wX, tf.float32) / 255.
            mX_ = tf.cast(mX, tf.float32) / 255.

            def vae_encoder(X, scope = 'vae_encoder', reuse=reuse):
                #we just replace vae with auto_encoder temporally_
                with tf.variable_scope(scope, reuse=reuse):
                    h1 = activ(conv(X, 'c1', nf=16, rf=8, stride=4,init_scale=np.sqrt(2)))  # h, w = 20, 20
                    h2 = activ(conv(h1, 'c2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2)))  # h, w = 9, 9
                    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=2, init_scale=np.sqrt(2)))  # h, w = 4, 4
                    h_encoder = 4.4*tf.nn.tanh(conv(h3, 'c4', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
                return h_encoder

            def encoder(X, scope = 'ordinary_encoder', reuse=reuse):
                with tf.variable_scope(scope, reuse=reuse):
                    h1 = activ(conv(X, 'c1', nf=16, rf=8, stride=4,init_scale=np.sqrt(2)))  # h, w = 20, 20
                    h2 = activ(conv(h1, 'c2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2)))  # h, w = 9, 9
                    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=2, init_scale=np.sqrt(2)))  # h, w = 4, 4
                    h_encoder = activ(conv(h3, 'c4', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))) # h, w = 2, 2
                return h_encoder

            def decoder(X, scope='vae_decoder', reuse=reuse):
                with tf.variable_scope(scope, reuse=reuse):
                    whd4 = activ(deconv(X, 'cd4', nf=64, rf=3, stride=1, output_shape=[nbatch, 4, 4, 64],
                                        init_scale=np.sqrt(2)))
                    whd3 = activ(deconv(whd4, 'cd3', nf=32, rf=3, stride=2, output_shape=[nbatch, 9, 9, 32],
                                        init_scale=np.sqrt(2)))
                    whd2 = activ(deconv(whd3, 'cd2', nf=16, rf=4, stride=2, output_shape=[nbatch, 20, 20, 16],
                                        init_scale=np.sqrt(2)))
                    wh_decoder = activ(deconv(whd2, 'cd1', nf=nc, rf=8, stride=4, output_shape=[nbatch, nh, nw, nc],
                                              init_scale=np.sqrt(2)))
                return wh_decoder

            if self.use_vae:
                wh_encoder = vae_encoder(wX_)
                wh = tf.stop_gradient(conv_to_fc(wh_encoder))
                mh_encoder = vae_encoder(mX_)
                mh = tf.stop_gradient(conv_to_fc(mh_encoder))
                #in order to train vae, we just need wh_encoder and wh_decoder
                wh_decoder = decoder(wh_encoder)
                self.encoder = wh_encoder
                self.decoder = wh_decoder
            else:
                wh_encoder = encoder(wX_)
                wh = conv_to_fc(wh_encoder)
                mh_encoder = encoder(mX_)
                mh = conv_to_fc(mh_encoder)

            with tf.variable_scope('master_module'):
                with tf.variable_scope('comm'):
                    mhc = activ(fc(mh, 'mhc', nh=cell))
                    if use_vae:
                        mhc = dense_layers(mhc, cell=cell, name='mhc', activ=activ, num_layers=3)
                    mhc = batch_to_seq(mhc, nenv, master_ts)#mh common
                    mms = batch_to_seq(mM, nenv, master_ts)
                    mh_, msnew = lstm(mhc, mms, mS, 'master_lstm', nh = cell)
                    mh_ = seq_to_batch(mh_)
                with tf.variable_scope('m_policy'):
                    ma = 4.4 * tf.nn.tanh(fc(mh_, 'ma', nh=cell))#deterministic action of manager
                    ma += tf.random_normal(shape=ma.get_shape(), stddev=0.05)  # add some Gaussian noise since manager is a deterministic policy
                with tf.variable_scope('m_value'):
                    mmh = tf.get_variable('mmh',  [cell, cell], initializer=ortho_init(np.sqrt(2)))
                    mma = tf.get_variable('mma', [cell, cell], initializer=ortho_init(np.sqrt(2)))
                    mb = tf.get_variable("mb", [cell], initializer=tf.constant_initializer(0.0))
                    #concat state embedding and action
                    mas = activ(tf.matmul(mh_, mmh) + tf.matmul(mA, mma) + mb)
                    mvf = fc(mas, 'mv', 1)#actually this should be Q function

            with tf.variable_scope('worker_module'):
                with tf.variable_scope('comm'):
                    w_wh = tf.get_variable('w_wh',  [cell, cell], initializer=ortho_init(np.sqrt(2)))
                    w_G = tf.get_variable('w_G', [cell, cell], initializer=ortho_init(np.sqrt(2)))
                    w_b = tf.get_variable("w_b", [cell], initializer=tf.constant_initializer(0.0))
                    whc = activ(tf.matmul(wh, w_wh) + tf.matmul(G, w_G) + w_b)#wh common
                    if use_vae:
                        whc = dense_layers(whc, cell=cell, name='whc', activ=activ, num_layers=3)
                    whc = batch_to_seq(whc, nenv * master_ts, worker_ts)#shape = [nenv * master_ts, worker_ts, wcell]
                    wms = batch_to_seq(wM, nenv * master_ts, worker_ts)
                    wh_, wsnew = lstm(whc, wms, wS, 'worker_lstm', nh = cell)
                    wh_ = seq_to_batch(wh_)#shape = [nbatch, wcell]
                with tf.variable_scope('w_policy'):
                    phi = 4.4 * tf.nn.tanh(fc(wh_, 'phi', nh=int(nact*cell), init_scale=np.sqrt(2.)))
                    phi = tf.reshape(phi, [-1, nact, cell])
                    G_ = tf.tile(tf.expand_dims(G, axis=1), [1, nact, 1])
                    wpi = - tf.reduce_mean(tf.square(G_ - phi), axis=-1)
                with tf.variable_scope('w_value'):
                    wvf = fc(wh_, 'wv', 1)

        a0 = noise_and_argmax(logits = wpi)
        wp0 = tf.nn.softmax(wpi) * tf.one_hot(a0, depth=nact)
        logp0 = tf.log(tf.reduce_sum(wp0, axis=-1))

        wv0 = wvf[:, 0]
        mv0 = mvf[:, 0]

        self.w_initial_state= np.zeros((nenv * master_ts, cell*2), dtype=np.float32)
        self.m_initial_state = np.zeros((nenv, cell*2), dtype=np.float32)

        def step(**kwargs):
            ma0, msnew_ = sess.run([ma, msnew], feed_dict={mX: kwargs['wobs'], mS: kwargs['mstate'], mM: kwargs['mmask'],
                                                           istraining: False})
            mv0_ = sess.run(mv0, feed_dict={mX: kwargs['mobs'], mS: kwargs['mstate'], mM: kwargs['mmask'], mA: ma0,
                                            istraining: False})
            goal_mask = np.expand_dims(np.asarray(kwargs['goal_mask'], dtype=np.bool), axis=-1)
            ma0 = kwargs['origin_goal'] * (1. - goal_mask) + ma0 * goal_mask
            a0_, wv0_, wsnew_, logp0_ = sess.run([a0, wv0, wsnew, logp0],
                                                 feed_dict={wX: kwargs['wobs'], wS: kwargs['wstate'], wM: kwargs['wmask'], G: ma0, istraining: False})
            return ma0, mv0_, msnew_, a0_, wv0_, wsnew_, logp0_

        def single_layer_step(obs, state, mask):
            return sess.run([a0, wv0, wsnew, logp0],
                            {wX: obs, wS: state, wM: mask, G: np.zeros(shape=(nenv, cell), dtype=state.dtype),
                             istraining: False})


        def wvalue(wobs, wstate, wmask, goal):
            return sess.run(wv0, {wX: wobs, wS: wstate, wM: wmask, G: goal, istraining: False})

        def mvalue(mobs, mstate, mmask):
            maction = sess.run(ma, feed_dict={mX: mobs, mS: mstate, mM: mmask, istraining: False})
            return sess.run(mv0, feed_dict={mX: mobs, mS: mstate, mM: mmask, mA: maction, istraining: False})

        def get_mh(mobs):
            return sess.run(mh, feed_dict = {mX:mobs, istraining:False})

        def get_wh(wobs):
            return sess.run(wh, feed_dict = {wX:wobs, istraining:False})

        self.wX = wX
        self.wM = wM
        self.wS = wS
        self.G = G

        self.mX = mX
        self.mM = mM
        self.mS = mS
        self.mA = mA
        self.Z = Z

        self.istraining = istraining

        self.wX_ = wX_
        self.mX_ = mX_
        self.ma = ma
        self.wpi = wpi
        self.mvf = mvf
        self.wvf = wvf
        self.mh = mh
        self.wh = wh
        # self.msigma = tf.exp(0.5*mlog_sigma)
        # self.normal_dist = normal_dist

        self.step = step
        self.single_layer_step = single_layer_step
        self.wvalue = wvalue
        self.mvalue = mvalue
        self.get_mh = get_mh
        self.get_wh = get_wh

class LstmPolicyTpg(object):
    '''
    LSTM policy with transition policy gradient, note that the goal in this policy is the abstraction of state
    '''
    def __init__(self, sess, ob_space, ac_space, nbatch, master_ts, worker_ts, cell=256, reuse=tf.AUTO_REUSE,
                 use_vae = True, phase = 'vae'):

        nenv = nbatch // (master_ts * worker_ts)
        self.nenv = nenv
        self.use_vae = use_vae

        nh, nw, nc = ob_space.shape
        nact = ac_space.n

        wX = tf.placeholder(tf.uint8, [nbatch, nh, nw, nc])#worker obs
        wS = tf.placeholder(tf.float32, [nenv * master_ts, cell * 2])#worker initial state
        wM = tf.placeholder(tf.float32, [nbatch])#worker mask
        G = tf.placeholder(tf.float32, [nbatch, cell])#goal set by master

        mX = tf.placeholder(tf.uint8, [nenv * master_ts, nh, nw, nc])#master obs
        mS = tf.placeholder(tf.float32, [nenv, cell * 2])#master initial state
        mM = tf.placeholder(tf.float32, [nenv * master_ts]) #master mask
        mA = tf.placeholder(tf.float32, [nenv * master_ts, cell]) #master's action

        Z = tf.placeholder(tf.float32, [None, cell])

        istraining = tf.placeholder(tf.bool, shape=[])

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu
            wX_ = tf.cast(wX, tf.float32) / 255.
            mX_ = tf.cast(mX, tf.float32) / 255.
            def vae_encoder(X, scope = 'vae_encoder', reuse=reuse):
                '''
                we just replace vae with auto-encoder temporally
                :param X:
                :param scope:
                :param reuse:
                :return:
                '''
                with tf.variable_scope(scope, reuse=reuse):
                    # X_ = tf.layers.batch_normalization(X, training=istraining, name='bn1')
                    h1 = activ(conv(X, 'c1', nf=16, rf=8, stride=4,init_scale=np.sqrt(2)))  # h, w = 20, 20
                    # h1 = tf.layers.batch_normalization(h1, training=istraining, name='bn2')
                    h2 = activ(conv(h1, 'c2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2)))  # h, w = 9, 9
                    # h2 = tf.layers.batch_normalization(h2, training=istraining, name= 'bn3')
                    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=2, init_scale=np.sqrt(2)))  # h, w = 4, 4
                    # h3 = tf.layers.batch_normalization(h3, training=istraining, name='bn4')
                    # mean = 4.4 * tf.nn.tanh(conv(h3, 'c4', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))) # h, w = 2, 2
                    # log_dev = 4.4 * tf.nn.tanh(conv(h3, 'c5', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))) # h, w = 2, 2
                    # vae_normal_dist = tf.distributions.Normal(loc=mean, scale=tf.exp(0.5 * log_dev))
                    # h_encoder = tf.squeeze(vae_normal_dist.sample(1), axis=0)
                    h_encoder = 4.4*tf.nn.tanh(conv(h3, 'c4', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
                return h_encoder

            def encoder(X, scope = 'ordinary_encoder', reuse=reuse):
                with tf.variable_scope(scope, reuse=reuse):
                    # X_ = tf.layers.batch_normalization(X/255., training=istraining, name='bn1')
                    h1 = activ(conv(X, 'c1', nf=16, rf=8, stride=4,init_scale=np.sqrt(2)))  # h, w = 20, 20
                    h2 = activ(conv(h1, 'c2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2)))  # h, w = 9, 9
                    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=2, init_scale=np.sqrt(2)))  # h, w = 4, 4
                    h_encoder = activ(conv(h3, 'c4', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))) # h, w = 2, 2
                return h_encoder

            def decoder(X, scope='vae_decoder', reuse=reuse):
                with tf.variable_scope(scope, reuse=reuse):
                    whd4 = activ(deconv(X, 'cd4', nf=64, rf=3, stride=1, output_shape=[nbatch, 4, 4, 64],
                                        init_scale=np.sqrt(2)))
                    whd3 = activ(deconv(whd4, 'cd3', nf=32, rf=3, stride=2, output_shape=[nbatch, 9, 9, 32],
                                        init_scale=np.sqrt(2)))
                    whd2 = activ(deconv(whd3, 'cd2', nf=16, rf=4, stride=2, output_shape=[nbatch, 20, 20, 16],
                                        init_scale=np.sqrt(2)))
                    wh_decoder = activ(deconv(whd2, 'cd1', nf=nc, rf=8, stride=4, output_shape=[nbatch, nh, nw, nc],
                                              init_scale=np.sqrt(2)))
                return wh_decoder

            if self.use_vae:
                wh_encoder = vae_encoder(wX_)
                wh = tf.stop_gradient(conv_to_fc(wh_encoder))
                mh_encoder = vae_encoder(mX_)
                mh = tf.stop_gradient(conv_to_fc(mh_encoder))
                #in order to train vae, we just need wh_encoder and wh_decoder
                wh_decoder = decoder(wh_encoder)
                self.encoder = wh_encoder
                self.decoder = wh_decoder
            else:
                wh_encoder = encoder(wX_)
                wh = conv_to_fc(wh_encoder)
                mh_encoder = encoder(mX_)
                mh = conv_to_fc(mh_encoder)

            with tf.variable_scope('master_module'):
                with tf.variable_scope('comm'):
                    mhc = batch_to_seq(mh, nenv, master_ts)#mh common
                    mms = batch_to_seq(mM, nenv, master_ts)
                    mh_, msnew = lstm(mhc, mms, mS, 'master_lstm', nh = cell)
                    mh_ = seq_to_batch(mh_)
                with tf.variable_scope('m_policy'):
                    '''
                    during the process of praticing, this can also be treated just as a goal generator
                    '''
                    mhp1_ = activ(fc(mh_, 'mpc1', nh=cell, init_scale=np.sqrt(2)))
                    # mhp1_ = activ(tf.layers.batch_normalization(mhp1_, training=istraining, name='mpbn1'))
                    mhp2_ = activ(fc(mhp1_, 'mpc2', nh=cell, init_scale=np.sqrt(2)))
                    # mhp2_ = activ(tf.layers.batch_normalization(mhp2_, training=istraining, name='mpbn2'))
                    mhp3_ = activ(fc(mhp2_, 'mpc3', nh=cell, init_scale=np.sqrt(2)))
                    # mhp3_ = activ(tf.layers.batch_normalization(mhp3_, training=istraining, name='mpbn3'))
                    ma = 4.4 * tf.nn.tanh(fc(mhp3_, 'ma', nh=cell, init_scale=np.sqrt(1.)))#deterministic action of manager
                    ma += tf.random_normal(shape=ma.get_shape(),
                                           stddev=0.05)  # add some Gaussian noise since manager is a deterministic policy
                with tf.variable_scope('m_value'):
                    mmh = tf.get_variable('mmh',  [cell, cell], initializer=ortho_init(np.sqrt(2)))
                    mma = tf.get_variable('mma', [cell, cell], initializer=ortho_init(np.sqrt(2)))
                    mb = tf.get_variable("mb", [cell], initializer=tf.constant_initializer(0.0))

                    mhv1_ = activ(fc(activ(tf.matmul(mh_, mmh) + tf.matmul(mA, mma) + mb), 'mvc1', nh=cell, init_scale=np.sqrt(2)))
                    # mhv1_ = activ(tf.layers.batch_normalization(mhv1_, training=istraining, name='mvbn1'))
                    mhv2_ = activ(fc(mhv1_, 'mvc2', nh=cell, init_scale=np.sqrt(2)))
                    # mhv2_ = activ(tf.layers.batch_normalization(mhv2_, training=istraining, name='mvbn2'))
                    mhv3_ = activ(fc(mhv2_, 'mvc3', nh=cell, init_scale=np.sqrt(2)))
                    # mhv3_ = activ(tf.layers.batch_normalization(mhv3_, training=istraining, name='mvbn3'))
                    mvf = fc(mhv3_, 'mv', 1)#actually this should be Q function
            if use_vae:
                '''
                if use_vae is True, we use GAN to generate the goal
                '''
                def generator(Z, cell=cell, scope='generator', reuse=reuse):
                    with tf.variable_scope(scope, reuse=reuse):
                        mg1 = activ(fc(Z, 'mg1', nh=cell, init_scale=np.sqrt(2)))
                        mg2 = activ(fc(mg1, 'mg2', nh=2*cell, init_scale=np.sqrt(2)))
                        # mg2 = activ(tf.layers.batch_normalization(mg2, training=istraining, name='mgbn1'))
                        mg3 = activ(fc(mg2, 'mg3', nh=cell, init_scale=np.sqrt(2)))
                        # mg3 = activ(tf.layers.batch_normalization(mg3, training=istraining, name='mgbn2'))
                        mg4 = activ(fc(mg3, 'mg4', nh=cell, init_scale=np.sqrt(2)))
                        # mg4 = activ(tf.layers.batch_normalization(mg4, training=istraining, name='mgbn3'))
                        mg = fc(mg4, 'mg', nh=cell, init_scale=np.sqrt(2))
                    return mg

                mg = generator(Z)

                def discriminator(X, cell=cell, scope='discriminator', reuse=reuse):
                    with tf.variable_scope(scope, reuse=reuse):
                        md1 = activ(fc(X, 'md1', nh=cell, init_scale=np.sqrt(2)))
                        md2 = activ(fc(md1, 'md2', nh=cell, init_scale=np.sqrt(2)))
                        # md2 = activ(tf.layers.batch_normalization(md2, training=istraining, name='mdbn1'))
                        md3 = activ(fc(md2, 'md3', nh=cell, init_scale=np.sqrt(2)))
                        # md3 = activ(tf.layers.batch_normalization(md3, training=istraining, name='mdbn2'))
                        md4 = activ(fc(md3, 'md4', nh=2*cell, init_scale=np.sqrt(2)))
                        # md4 = activ(tf.layers.batch_normalization(md4, training=istraining, name='mdbn3'))
                        md = fc(md4, 'md', nh=cell, init_scale=np.sqrt(2))
                    return md

                G_loss = tf.reduce_mean(tf.reduce_sum(discriminator(mg), axis=1))
                         # tf.maximum(0.2 * tf.reduce_mean(mse(ma, mh)), 5.)  # loss of generator, feed mhs to this loss
                if phase == 'vae':
                    eps = tf.random_uniform(shape=[wh.get_shape()[0].value, 1], minval=0., maxval=1.)
                    x_inter = eps * mg + (1. - eps) * wh
                    lip_grad = tf.gradients(discriminator(x_inter), [x_inter])[0]
                    grad_pen = tf.reduce_mean(tf.reduce_sum(mse(lip_grad, 1.), axis=1))
                    D_loss = tf.reduce_mean(tf.reduce_sum(discriminator(wh) - discriminator(mg), axis=1)) + 0.5*grad_pen
                else:
                    eps = tf.random_uniform(shape=[mh.get_shape()[0].value, 1], minval=0, maxval=1.)
                    x_inter = eps * mg + (1. - eps) * mh
                    lip_grad = tf.gradients(discriminator(x_inter), [x_inter])[0]
                    grad_pen = tf.reduce_mean(tf.reduce_sum(mse(lip_grad, 1.), axis=1))
                    D_loss = tf.reduce_mean(tf.reduce_sum(discriminator(mh) - discriminator(mg), axis=1)) + 0.5*grad_pen

                self.G_loss = G_loss
                self.D_loss = D_loss

            with tf.variable_scope('worker_module'):
                with tf.variable_scope('comm'):
                    w_wh = tf.get_variable('w_wh',  [cell, cell], initializer=ortho_init(np.sqrt(2)))
                    w_G = tf.get_variable('w_G', [cell, cell], initializer=ortho_init(np.sqrt(2)))
                    b = tf.get_variable("b", [cell], initializer=tf.constant_initializer(0.0))
                    whc = activ(tf.matmul(wh, w_wh) + tf.matmul(G, w_G) + b)#wh common
                    if use_vae:
                        whc = dense_layers(whc, cell=cell, name='whc', activ=activ, num_layers=3)
                    whc = batch_to_seq(whc, nenv * master_ts, worker_ts)#shape = [nenv * master_ts, worker_ts, wcell]
                    wms = batch_to_seq(wM, nenv * master_ts, worker_ts)
                    wh_, wsnew = lstm(whc, wms, wS, 'worker_lstm', nh = cell)
                    wh_ = seq_to_batch(wh_)#shape = [nbatch, wcell]
                with tf.variable_scope('w_policy'):
                    # whp1_ = activ(fc(wh_, 'wpc1', nh=cell, init_scale=0.01))
                    # whp1_ = activ(tf.layers.batch_normalization(whp1_, training=istraining, name='wpbn1'))
                    # whp2_ = activ(fc(whp1_, 'wpc2', nh=cell, init_scale=0.01))
                    # whp2_ = activ(tf.layers.batch_normalization(whp2_, training=istraining, name='wpbn2'))
                    # whp3_ = activ(fc(whp2_, 'wpc3', nh=cell//2, init_scale=0.01))
                    # whp3_ = activ(tf.layers.batch_normalization(whp3_, training=istraining, name='wpbn3'))
                    wpi = fc(wh_, 'wpi', nact)#in step mode, shape = [nenv,  nact]
                with tf.variable_scope('w_value'):
                    # whv1_ = activ(fc(wh_, 'wvc1', nh=cell, init_scale=np.sqrt(2)))
                    # whv1_ = activ(tf.layers.batch_normalization(whv1_, training=istraining, name='wvbn1'))
                    # whv2_ = activ(fc(whv1_, 'wvc2', nh=cell, init_scale=np.sqrt(2)))
                    # whv2_ = activ(tf.layers.batch_normalization(whv2_, training=istraining, name='wvbn2'))
                    # whv3_ = activ(fc(whv2_, 'wvc3', nh=cell//2, init_scale=np.sqrt(2)))
                    # whv3_ = activ(tf.layers.batch_normalization(whv3_, training=istraining, name='wvbn3'))
                    wvf = fc(wh_, 'wv', 1)

        a0 = noise_and_argmax(logits = wpi)
        neglogp0 = tf.reduce_sum(tf.nn.softmax(wpi) * tf.one_hot(a0, depth=nact), axis=-1)
        # self.wpdtype = make_pdtype(ac_space)
        # self.wpd = self.wpdtype.pdfromflat(4.*wpi)
        # a0 = self.wpd.sample()
        # neglogp0 = self.wpd.neglogp(a0)
        wv0 = wvf[:, 0]
        mv0 = mvf[:, 0]
        # normal_dist = tf.distributions.Normal(loc = mmu, scale = tf.exp(0.5*mlog_sigma))
        # goal0 = tf.squeeze(normal_dist.sample(1), axis = 0)
        # print(goal0.get_shape())
        # neglogpg0 = - normal_dist.log_prob(goal0)

        self.w_initial_state = np.zeros((nenv * master_ts, cell*2), dtype=np.float32)
        self.m_initial_state = np.zeros((nenv, cell*2), dtype=np.float32)

        if self.use_vae:

            def step(**kwargs):
                ma0, msnew_ = sess.run([ma, msnew],
                                       feed_dict={mh: kwargs['mhs'], mS: kwargs['mstate'], mM: kwargs['mmask'],
                                                  istraining: False})
                mv0_ = sess.run(mv0, feed_dict={mh: kwargs['mhs'], mS: kwargs['mstate'], mM: kwargs['mmask'], mA: ma0,
                                                istraining: False})
                goal_mask = np.asarray(kwargs['goal_mask'], dtype=np.bool)
                ma0 = kwargs['origin_goal'] * np.expand_dims(1. - goal_mask, axis=1) + ma0 * np.expand_dims(goal_mask, axis=1)
                a0_, wv0_, wsnew_, neglogp0_ = sess.run([a0, wv0, wsnew, neglogp0],
                                                        feed_dict={wh: kwargs['whs'], wS: kwargs['wstate'],
                                                                   wM: kwargs['wmask'], G: ma0, istraining: False})
                return ma0, mv0_, msnew_, a0_, wv0_, wsnew_, neglogp0_

            def single_layer_step(whs, state, mask):
                return sess.run([a0, wv0, wsnew, neglogp0], {wh: whs, wS: state, wM: mask, G: np.zeros(shape=whs.shape, dtype=whs.dtype), istraining: False})

            def practice(whs, state, mask, goal):
                return sess.run([a0, wv0, wsnew, neglogp0], {wh: whs, wS: state, wM: mask, G: goal, istraining: False})

            self.wpdtype = make_pdtype(ac_space)
            uniform_dist = tf.ones(shape=wpi.get_shape(), dtype=tf.float32) / float(nact)
            a = self.wpdtype.pdfromflat(uniform_dist).sample()
            def exploration(obs, phase):
                if phase == 'train_vae':
                    return sess.run([a, wh_decoder], feed_dict = {wX:obs})
                else:
                    #phase should be 'train_gan
                    a_, mg_ = sess.run([a, mg], feed_dict={Z:np.random.uniform(-4.4, 4.4, size=[len(obs), cell])})
                    return [a_, sess.run(wh_decoder, feed_dict = {wh_encoder:mg_.reshape(-1, 2, 2, 64)})]

            def wvalue(whs, wstate, wmask, goal):
                return sess.run(wv0, {wh: whs, wS: wstate, wM: wmask, G: goal, istraining: False})

            def mvalue(mhs, mstate, mmask):
                maction = sess.run(ma, feed_dict={mh: mhs, mS: mstate, mM: mmask, istraining: False})
                return sess.run(mv0, feed_dict={mh: mhs, mS: mstate, mM: mmask, mA: maction, istraining: False})

            def goal_generator():
                z = np.random.uniform(-4.4, 4.4, size=(nenv, cell))
                goal = sess.run(mg, feed_dict = {Z:z, istraining:False})
                # goal = np.expand_dims(np.asarray(goal_mask, np.bool), axis=1)*goal +\
                #        np.expand_dims(1.- np.asarray(goal_mask, np.bool), 1)*orig_goal
                # mean, var = generator
                # mean, var = np.expand_dims(mean, axis=0), np.expand_dims(var, axis=0)
                # shape_ = list(mean.shape)
                # shape_[0] = self.nenv
                # shape_ = tuple(shape_)
                return goal

            def embedding_decoder(h):
                '''
                :param goal: shape is [bs, 256]
                :return: a [bs, nh, nw, nc] tensor
                '''
                h_ = h.reshape(-1, 2, 2, 64)
                return sess.run(wh_decoder, feed_dict={wh_encoder: h_})

            self.exploration = exploration
            self.practice = practice
            self.goal_generator = goal_generator
            self.embedding_decoder = embedding_decoder

        else:

            def step(**kwargs):
                ma0, msnew_ = sess.run([ma, msnew],
                                       feed_dict={mX: kwargs['wobs'], mS: kwargs['mstate'], mM: kwargs['mmask'],
                                                  istraining: False})
                mv0_ = sess.run(mv0, feed_dict={mX: kwargs['mobs'], mS: kwargs['mstate'], mM: kwargs['mmask'], mA: ma0,
                                                istraining: False})
                goal_mask = np.asarray(kwargs['goal_mask'], dtype=np.bool)
                ma0 = kwargs['origin_goal'] * np.expand_dims(1. - goal_mask, axis=1) + ma0 * np.expand_dims(goal_mask,
                                                                                                            axis=1)
                a0_, wv0_, wsnew_, neglogp0_ = sess.run([a0, wv0, wsnew, neglogp0],
                                                        feed_dict={wX: kwargs['wobs'], wS: kwargs['wstate'],
                                                                   wM: kwargs['wmask'], G: ma0, istraining: False})
                return ma0, mv0_, msnew_, a0_, wv0_, wsnew_, neglogp0_

            def single_layer_step(obs, state, mask):
                return sess.run([a0, wv0, wsnew, neglogp0],
                                {wX: obs, wS: state, wM: mask, G: np.zeros(shape=(nenv, cell), dtype=state.dtype),
                                 istraining: False})


            def wvalue(wobs, wstate, wmask, goal):
                return sess.run(wv0, {wX: wobs, wS: wstate, wM: wmask, G: goal, istraining: False})

            def mvalue(mobs, mstate, mmask):
                maction = sess.run(ma, feed_dict={mX: mobs, mS: mstate, mM: mmask, istraining: False})
                return sess.run(mv0, feed_dict={mX: mobs, mS: mstate, mM: mmask, mA: maction, istraining: False})


        def get_mh(mobs):
            return sess.run(mh, feed_dict = {mX:mobs, istraining:False})

        def get_wh(wobs):
            return sess.run(wh, feed_dict = {wX:wobs, istraining:False})

        self.wX = wX
        self.wM = wM
        self.wS = wS
        self.G = G

        self.mX = mX
        self.mM = mM
        self.mS = mS
        self.mA = mA
        self.Z = Z

        self.istraining = istraining

        self.wX_ = wX_
        self.mX_ = mX_
        self.ma = ma
        self.wpi = wpi
        self.mvf = mvf
        self.wvf = wvf
        self.mh = mh
        self.wh = wh
        # self.msigma = tf.exp(0.5*mlog_sigma)
        # self.normal_dist = normal_dist

        self.step = step
        self.single_layer_step = single_layer_step
        self.wvalue = wvalue
        self.mvalue = mvalue
        self.get_mh = get_mh
        self.get_wh = get_wh

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, master_ts, worker_ts, cell=36, reuse=tf.AUTO_REUSE):

        nenv = nbatch // (master_ts * worker_ts)
        self.nenv = nenv

        nh, nw, nc = ob_space.shape
        nact = ac_space.n

        wX = tf.placeholder(tf.float32, [nbatch, nh, nw, nc])#worker obs
        wS = tf.placeholder(tf.float32, [nenv * master_ts, cell * 2])#worker initial state
        wM = tf.placeholder(tf.float32, [nbatch])#worker mask
        G = tf.placeholder(tf.float32, [nbatch, cell])#goal set by master

        mX = tf.placeholder(tf.float32, [nenv * master_ts, nh, nw, nc])#master obs
        mS = tf.placeholder(tf.float32, [nenv, cell * 2])#master initial state
        mM = tf.placeholder(tf.float32, [nenv * master_ts]) #master mask

        istraining = tf.placeholder(tf.bool, shape=[])

        with tf.variable_scope("model", reuse=reuse):
            with tf.variable_scope('comm_encoder'):
                def activ(X):
                    return tf.nn.leaky_relu(X, alpha=0.2)
                wX_std = wX / 255.
                wX_ = tf.layers.batch_normalization(wX_std, training=istraining, name='bn1')
                wh1 = activ(conv_unequ_size(wX_, 'c1', nf=16, rf=[20, 8], stride=[5, 4],init_scale=np.sqrt(2)))  # h, w = 39, 39
                wh1 = tf.layers.batch_normalization(wh1, training=istraining, name='bn2')
                wh2 = activ(conv(wh1, 'c2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2)))  # h, w = 18, 18
                wh2 = tf.layers.batch_normalization(wh2, training=istraining, name='bn3')
                wh3 = activ(conv(wh2, 'c3', nf=64, rf=3, stride=2, pad='SAME', init_scale=np.sqrt(2)))  # h, w = 9, 9
                wh3 = tf.layers.batch_normalization(wh3, training=istraining, name='bn4')
                wmean = conv(wh3, 'c4', nf=4, rf=3, stride=3, init_scale=np.sqrt(2)) # h, w = 3, 3, shape = [bs, 3, 3, 3]
                wlog_dev = 4.4 * tf.nn.tanh(conv(wh3, 'c5', nf=4, rf=3, stride=3, init_scale=np.sqrt(2))) # h, w = 3, 3, shape = [bs, 3, 3, 3]
                wh_encoder = 4.4 * tf.nn.tanh(wmean + tf.exp(0.5 * wlog_dev) * tf.random_normal(shape=wmean.get_shape()))
                wh = tf.stop_gradient(conv_to_fc(wh_encoder))# shape = [bs, 27]

                mX_std = mX / 255.
                mX_ = tf.layers.batch_normalization(mX_std, training=istraining, name='bn1')
                mh1 = activ(conv_unequ_size(mX_, 'c1', nf=16, rf=[20, 8], stride=[5, 4], init_scale=np.sqrt(2)))# h, w = 39, 39
                mh1 = tf.layers.batch_normalization(mh1, training=istraining, name='bn2')
                mh2 = activ(conv(mh1, 'c2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2)))  # h, w = 18, 18
                mh2 = tf.layers.batch_normalization(mh2, training=istraining, name='bn3')
                mh3 = activ(conv(mh2, 'c3', nf=64, rf=3, stride=2, pad='SAME', init_scale=np.sqrt(2)))  # h, w = 9, 9
                mh3 = tf.layers.batch_normalization(mh3, training=istraining, name='bn4')
                mmean = conv(mh3, 'c4', nf=4, rf=3, stride=3, init_scale=np.sqrt(2)) # h, w = 3, 3, shape = [bs, 3, 3, 3]
                mlog_dev = 4.4 * tf.nn.tanh(conv(mh3, 'c5', nf=4, rf=3, stride=3, init_scale=np.sqrt(2))) # h, w = 3, 3, shape = [bs, 3, 3, 3]
                mh_encoder = 4.4 * tf.nn.tanh(mmean + tf.exp(0.5 * mlog_dev) * tf.random_normal(shape=mmean.get_shape()))
                mh = tf.stop_gradient(conv_to_fc(mh_encoder))# shape = [bs, 27]

            with tf.variable_scope('comm_decoder'):
                whd4 = activ(deconv(wh_encoder, 'cd4', nf=64, rf=3, stride=3, output_shape=[nbatch, 9, 9, 64], init_scale=np.sqrt(2)))
                whd3 = activ(deconv(whd4, 'cd3', nf=32, rf=3, stride=2, output_shape=[nbatch, 18, 18, 32], pad='SAME', init_scale=np.sqrt(2)))
                whd2 = activ(deconv(whd3, 'cd2', nf=16, rf=4, stride=2, output_shape=[nbatch, 39, 39, 16], init_scale=np.sqrt(2)))
                wh_decoder = 1.1 * tf.nn.sigmoid(deconv_unequ_size(whd2, 'cd1', nf=nc, rf=[20, 8], stride=[5,4], output_shape=[nbatch, nh, nw, nc], init_scale=np.sqrt(2)))

            with tf.variable_scope('master_module'):
                with tf.variable_scope('comm'):
                    mhc = batch_to_seq(mh, nenv, master_ts)#mh common
                    mms = batch_to_seq(mM, nenv, master_ts)
                    mh_, msnew = lnlstm(mhc, mms, mS, 'master_lstm', nh = cell)
                    mh_ = seq_to_batch(mh_)
                with tf.variable_scope('m_policy'):
                    with tf.variable_scope('comm'):
                        mhp1_ = fc(mh_, 'mpc1', nh=cell, init_scale=np.sqrt(2))
                        mhp1_ = activ(tf.layers.batch_normalization(mhp1_, training=istraining, name='mpbn1'))
                        mhp2_ = fc(mhp1_, 'mpc2', nh=cell, init_scale=np.sqrt(2))
                        mhp2_ = activ(tf.layers.batch_normalization(mhp2_, training=istraining, name='mpbn2'))
                        mhp3_ = fc(mhp2_, 'mpc3', nh=cell, init_scale=np.sqrt(2))
                        mhp3_ = activ(tf.layers.batch_normalization(mhp3_, training=istraining, name='mpbn3'))
                    with tf.variable_scope('mu'):
                        mmu = 4.4 * tf.nn.tanh(fc(mhp3_, 'mmu', cell))#in step mode, shape = [nenv, mcell]
                    with tf.variable_scope('log_sigma'):
                        mlog_sigma = 4.4 * tf.nn.tanh(fc(mhp3_, 'mlog_sigma', cell))
                with tf.variable_scope('m_value'):
                    mhv1_ = fc(mh_, 'mvc1', nh=cell, init_scale=np.sqrt(2))
                    mhv1_ = activ(tf.layers.batch_normalization(mhv1_, training=istraining, name='mvbn1'))
                    mhv2_ = fc(mhv1_, 'mvc2', nh=cell, init_scale=np.sqrt(2))
                    mhv2_ = activ(tf.layers.batch_normalization(mhv2_, training=istraining, name='mvbn2'))
                    mhv3_ = fc(mhv2_, 'mvc3', nh=cell, init_scale=np.sqrt(2))
                    mhv3_ = activ(tf.layers.batch_normalization(mhv3_, training=istraining, name='mvbn3'))
                    mvf = fc(mhv3_, 'mv', 1)

            def discriminator(X):
                with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
                    md1 = tf.nn.tanh(fc(X, 'md1', nh=cell, init_scale=np.sqrt(2)))
                    md2 = fc(md1, 'md2', nh=cell, init_scale=np.sqrt(2))
                    md2 = activ(tf.layers.batch_normalization(md2, training=istraining, name='mdbn1'))
                    md3 = fc(md2, 'md3', nh=cell, init_scale=np.sqrt(2))
                    md3 = activ(tf.layers.batch_normalization(md3, training=istraining, name='mdbn2'))
                    md4 = fc(md3, 'md4', nh=cell, init_scale=np.sqrt(2))
                    md4 = activ(tf.layers.batch_normalization(md4, training=istraining, name='mdbn3'))
                    md = fc(md4, 'md', nh=cell, init_scale=np.sqrt(2))
                return md

            G_loss = tf.reduce_mean(discriminator(mmu)) - \
                     tf.maximum(0.2 * tf.reduce_mean(mse(mmu, mh)), 5.)  # loss of generator, feed mhs to this loss
            eps = tf.random_uniform([self.nenv * master_ts, 1], minval=0., maxval=1.)
            x_inter = eps * mmu + (1. - eps) * mh
            lip_grad = tf.gradients(discriminator(x_inter), [x_inter])[0]
            grad_pen = 2. * tf.reduce_mean(mse(lip_grad, 1))
            D_loss = tf.reduce_mean(discriminator(mh) - discriminator(mmu)) + grad_pen

            with tf.variable_scope('worker_module'):
                with tf.variable_scope('comm'):
                    ir = - tf.stop_gradient(tf.reduce_mean(tf.square(wh - G), axis=1))

                    w_wh = tf.get_variable('w_wh',  [cell, cell], initializer=ortho_init(np.sqrt(2)))
                    w_G = tf.get_variable('w_G', [cell, cell], initializer=ortho_init(np.sqrt(2)))
                    b = tf.get_variable("b", [cell], initializer=tf.constant_initializer(0.0))
                    whc = activ(tf.matmul(wh, w_wh) + tf.matmul(G, w_G) + b)#wh common

                    whc = batch_to_seq(whc, nenv * master_ts, worker_ts)#shape = [nenv * master_ts, worker_ts, wcell]
                    wms = batch_to_seq(wM, nenv * master_ts, worker_ts)
                    wh_, wsnew = lnlstm(whc, wms, wS, 'worker_lstm', nh = cell)
                    wh_ = seq_to_batch(wh_)#shape = [nbatch, wcell]
                with tf.variable_scope('w_policy'):
                    whp1_ = fc(wh_, 'wpc1', nh=cell, init_scale=np.sqrt(2))
                    whp1_ = activ(tf.layers.batch_normalization(whp1_, training=istraining, name='wpbn1'))
                    whp2_ = fc(whp1_, 'wpc2', nh=cell, init_scale=np.sqrt(2))
                    whp2_ = activ(tf.layers.batch_normalization(whp2_, training=istraining, name='wpbn2'))
                    whp3_ = fc(whp2_, 'wpc3', nh=cell, init_scale=np.sqrt(2))
                    whp3_ = activ(tf.layers.batch_normalization(whp3_, training=istraining, name='wpbn3'))
                    wpi = fc(whp3_, 'wpi', nact)#in step mode, shape = [nenv,  nact]
                with tf.variable_scope('w_value'):
                    whv1_ = fc(wh_, 'wvc1', nh=cell, init_scale=np.sqrt(2))
                    whv1_ = activ(tf.layers.batch_normalization(whv1_, training=istraining, name='wvbn1'))
                    whv2_ = fc(whv1_, 'wvc2', nh=cell, init_scale=np.sqrt(2))
                    whv2_ = activ(tf.layers.batch_normalization(whv2_, training=istraining, name='wvbn2'))
                    whv3_ = fc(whv2_, 'wvc3', nh=cell, init_scale=np.sqrt(2))
                    whv3_ = activ(tf.layers.batch_normalization(whv3_, training=istraining, name='wvbn3'))
                    wvf = fc(whv3_, 'wv', 1)

        self.wpdtype = make_pdtype(ac_space)
        self.wpd = self.wpdtype.pdfromflat(wpi)

        wv0 = wvf[:, 0]
        a0 = self.wpd.sample()
        neglogp0 = self.wpd.neglogp(a0)
        mv0 = mvf[:, 0]
        normal_dist = tf.distributions.Normal(loc = mmu, scale = tf.exp(0.5*mlog_sigma))
        goal0 = tf.squeeze(normal_dist.sample(1), axis = 0)
        # print(goal0.get_shape())
        neglogpg0 = - normal_dist.log_prob(goal0)

        self.w_initial_state = np.zeros((nenv * master_ts, cell*2), dtype=np.float32)
        self.m_initial_state = np.zeros((nenv, cell*2), dtype=np.float32)

        #get intrinsic reward for worker, based on next obs and goal
        def get_ir(whs, goal):
            return sess.run(ir, feed_dict = {wh : whs, G : goal})#shape = [nenv] in step mode

        #def step(wobs, wstate, wmask, goal, mobs, mstate, mmask, set_goal = False):
        def step(**kwargs):
            goal0_, mv0_, msnew_, neglogpg0_ = sess.run([goal0, mv0, msnew, neglogpg0], feed_dict = {mh : kwargs['mhs'], mS : kwargs['mstate'], mM : kwargs['mmask'], istraining : False})
            goal_mask = np.asarray(kwargs['goal_mask'], dtype = np.bool)
            goal0_ = kwargs['origin_goal'] * np.expand_dims(1.-goal_mask, axis=1) + goal0_ * np.expand_dims(goal_mask, axis=1)
            a0_, wv0_, wsnew_, neglogp0_ = sess.run([a0, wv0, wsnew, neglogp0], feed_dict = {wh : kwargs['whs'], wS : kwargs['wstate'], wM : kwargs['wmask'], G : goal0_, istraining : False})
            return goal0_, mv0_, msnew_, neglogpg0_, a0_, wv0_, wsnew_, neglogp0_

        def practice(obs, state, mask, goal):
            return sess.run([a0, wv0, wsnew, neglogp0], {wX: obs, wS: state, wM: mask, G: goal, istraining: False})

        uniform_dist = tf.ones(shape=wpi.get_shape(), dtype=tf.float32) / float(nact)
        a = self.wpdtype.pdfromflat(uniform_dist).sample()
        def exploration(obs):
            return sess.run([a, wh_encoder], feed_dict={wX : obs, istraining : False})

        def wvalue(whs, wstate, wmask, goal):
            return sess.run(wv0, {wh : whs, wS : wstate, wM : wmask, G : goal, istraining : False})

        def mvalue(mhs, mstate, mmask):
            return sess.run(mv0, {mh : mhs, mS : mstate, mM : mmask, istraining : False})

        def goal_generator(generator):
            mean, var = generator
            mean, var = np.expand_dims(mean, axis=0), np.expand_dims(var, axis=0)
            shape_ = list(mean.shape)
            shape_[0] = self.nenv
            shape_ = tuple(shape_)
            return mean + np.sqrt(var) * np.random.normal(size=shape_)

        def goal_decoder(goal):
            '''
            :param goal: shape is [bs, 3, 3, 3]
            :return: a [bs, nh, nw, nc] tensor
            '''
            return sess.run(wh_decoder, feed_dict = {wh_encoder : goal})

        def get_mh(mobs):
            return sess.run(mh, feed_dict = {mX:mobs, istraining:False})

        def get_wh(wobs):
            return sess.run(wh, feed_dict = {wX:wobs, istraining:False})

        self.wX = wX
        self.wM = wM
        self.wS = wS
        self.G = G

        self.mX = mX
        self.mM = mM
        self.mS = mS

        self.istraining = istraining

        self.wX_std = wX_std
        self.wmean = wmean
        self.wlog_dev = wlog_dev
        self.encoder = wh_encoder
        self.decoder = wh_decoder
        self.wpi = wpi
        self.wvf = wvf
        self.mmu = mmu #mean of goal assuming that goal is a Gaussain
        self.mh = mh
        self.wh = wh
        self.discriminator = discriminator #dicriminator for WGAN-GP
        self.G_loss = G_loss
        self.D_loss = D_loss
        self.msigma = tf.exp(0.5*mlog_sigma)
        self.normal_dist = normal_dist
        self.mvf = mvf

        self.get_ir = get_ir
        self.step = step
        self.exploration = exploration
        self.practice = practice
        self.wvalue = wvalue
        self.mvalue = mvalue
        self.goal_generator = goal_generator
        self.goal_decoder = goal_decoder
        self.get_mh = get_mh
        self.get_wh = get_wh

class CnnPolicyDpg(object):
    '''
    Cnn policy with deterministic policy gradient in high level, note that the goal in this policy is the abstraction of state
    '''
    def __init__(self, sess, ob_space, ac_space, nbatch, master_ts, worker_ts, cell=256, reuse=tf.AUTO_REUSE,
                 use_vae = True, phase = 'vae'):

        nenv = nbatch // (master_ts * worker_ts)
        self.nenv = nenv
        self.use_vae = use_vae

        nh, nw, nc = ob_space.shape
        nact = ac_space.n

        wX = tf.placeholder(tf.uint8, [nbatch, nh, nw, nc])#worker obs
        G = tf.placeholder(tf.float32, [nbatch, cell])#goal set by master
        mX = tf.placeholder(tf.uint8, [nenv * master_ts, nh, nw, nc])#master obs
        mA = tf.placeholder(tf.float32, [nenv * master_ts, cell]) #master's action
        Z = tf.placeholder(tf.float32, [None, cell])
        istraining = tf.placeholder(tf.bool, shape=[])

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu
            wX_ = tf.cast(wX, tf.float32) / 255.
            mX_ = tf.cast(mX, tf.float32) / 255.
            def vae_encoder(X, scope = 'vae_encoder', reuse=reuse):
                with tf.variable_scope(scope, reuse=reuse):
                    h1 = activ(conv(X, 'c1', nf=16, rf=8, stride=4,init_scale=np.sqrt(2)))  # h, w = 20, 20
                    h2 = activ(conv(h1, 'c2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2)))  # h, w = 9, 9
                    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=2, init_scale=np.sqrt(2)))  # h, w = 4, 4
                    h_encoder = 4.4*tf.nn.tanh(conv(h3, 'c4', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))) #h, w = 2, 2
                return h_encoder

            def encoder(X, scope = 'ordinary_encoder', reuse=reuse):
                with tf.variable_scope(scope, reuse=reuse):
                    h1 = activ(conv(X, 'c1', nf=16, rf=8, stride=4,init_scale=np.sqrt(2)))  # h, w = 20, 20
                    h2 = activ(conv(h1, 'c2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2)))  # h, w = 9, 9
                    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=2, init_scale=np.sqrt(2)))  # h, w = 4, 4
                    h_encoder = activ(conv(h3, 'c4', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))) # h, w = 2, 2
                return h_encoder

            def decoder(X, scope='vae_decoder', reuse=reuse):
                with tf.variable_scope(scope, reuse=reuse):
                    whd4 = activ(deconv(X, 'cd4', nf=64, rf=3, stride=1, output_shape=[nbatch, 4, 4, 64], init_scale=np.sqrt(2)))
                    whd3 = activ(deconv(whd4, 'cd3', nf=32, rf=3, stride=2, output_shape=[nbatch, 9, 9, 32], init_scale=np.sqrt(2)))
                    whd2 = activ(deconv(whd3, 'cd2', nf=16, rf=4, stride=2, output_shape=[nbatch, 20, 20, 16], init_scale=np.sqrt(2)))
                    wh_decoder = activ(deconv(whd2, 'cd1', nf=nc, rf=8, stride=4, output_shape=[nbatch, nh, nw, nc], init_scale=np.sqrt(2)))
                return wh_decoder

            if self.use_vae:
                wh_encoder = vae_encoder(wX_)
                wh = tf.stop_gradient(conv_to_fc(wh_encoder))
                mh_encoder = vae_encoder(mX_)
                mh = tf.stop_gradient(conv_to_fc(mh_encoder))
                #in order to train vae, we just need wh_encoder and wh_decoder
                wh_decoder = decoder(wh_encoder)
                self.encoder = wh_encoder
                self.decoder = wh_decoder
            else:
                wh_encoder = encoder(wX_)
                wh = conv_to_fc(wh_encoder)
                mh_encoder = encoder(mX_)
                mh = conv_to_fc(mh_encoder)

            with tf.variable_scope('master_module'):
                with tf.variable_scope('comm'):
                    mh_ = activ(fc(mh, 'mh_', nh=cell, init_scale=np.sqrt(2)))
                with tf.variable_scope('m_policy'):
                    ma0 = 4.4 * tf.nn.tanh(fc(mh_, 'ma', nh=cell, init_scale=np.sqrt(1.)))#deterministic action of manager
                    ma0 += tf.random_normal(shape=ma0.get_shape(), stddev=0.05)# add some Gaussian noise since manager is a deterministic policy
                with tf.variable_scope('m_value'):
                    mmh = tf.get_variable('mmh',  [cell, cell], initializer=ortho_init(np.sqrt(2)))
                    mma = tf.get_variable('mma', [cell, cell], initializer=ortho_init(np.sqrt(2)))
                    mb = tf.get_variable("mb", [cell], initializer=tf.constant_initializer(0.0))
                    mvf = fc(activ(tf.matmul(mh_, mmh) + tf.matmul(mA, mma) + mb), 'mv', nh=1, init_scale=np.sqrt(2.))
            if use_vae:
                '''
                if use_vae is True, we use GAN to generate the goal
                '''
                def generator(Z, cell=cell, scope='generator', reuse=reuse):
                    with tf.variable_scope(scope, reuse=reuse):
                        g1 = activ(fc(Z, 'g1', nh=cell, init_scale=np.sqrt(2)))
                        g2 = activ(fc(g1, 'g2', nh=cell, init_scale=np.sqrt(2)))
                        g = 4.4*tf.nn.tanh(fc(g2, 'g', nh=cell, init_scale=np.sqrt(2)))
                    return g

                g = generator(Z)

                def discriminator(X, cell=cell, scope='discriminator', reuse=reuse):
                    with tf.variable_scope(scope, reuse=reuse):
                        d1 = activ(fc(X, 'd1', nh=cell, init_scale=np.sqrt(2)))
                        d2 = activ(fc(d1, 'd2', nh=cell, init_scale=np.sqrt(2)))
                        d = fc(d2, 'd', nh=1, init_scale=np.sqrt(2))
                    return d

                G_loss = tf.reduce_mean(discriminator(g))
                eps = tf.random_uniform(shape=[wh.get_shape()[0].value, 1], minval=0., maxval=1.)
                x_inter = eps * g + (1. - eps) * wh
                lip_grad = tf.gradients(discriminator(x_inter), [x_inter])[0]
                grad_norm = tf.sqrt(tf.reduce_sum((lip_grad)**2, axis=1))
                grad_pen = tf.reduce_mean(activ(grad_norm - 1.))
                D_loss = tf.reduce_mean(discriminator(wh)) - tf.reduce_mean(discriminator(g)) + 10.*grad_pen

                self.G_loss = G_loss
                self.D_loss = D_loss
                self.generator = generator
                self.discriminator = discriminator

            with tf.variable_scope('worker_module'):
                with tf.variable_scope('comm'):
                    w_wh = tf.get_variable('w_wh',  [cell, cell], initializer=ortho_init(np.sqrt(2)))
                    w_G = tf.get_variable('w_G', [cell, cell], initializer=ortho_init(np.sqrt(2)))
                    b = tf.get_variable("b", [cell], initializer=tf.constant_initializer(0.0))
                    wh_ = activ(tf.matmul(wh, w_wh) + tf.matmul(G, w_G) + b)#wh common
                    if use_vae:
                        wh_ = dense_layers(wh_, cell=cell, name='wh_', activ=activ, num_layers=3)
                with tf.variable_scope('w_policy'):
                    wpi = fc(wh_, 'wpi', nact)
                with tf.variable_scope('w_value'):
                    wvf = fc(wh_, 'wv', 1)
        #
        # wa0 = noise_and_argmax(logits = wpi)
        # wneglogp0 = tf.reduce_sum(tf.nn.softmax(wpi) * tf.one_hot(wa0, depth=nact), axis=-1)
        self.wpdtype = make_pdtype(ac_space)
        self.wpd = self.wpdtype.pdfromflat(wpi)
        wa0 = self.wpd.sample()
        wneglogp0 = self.wpd.neglogp(wa0)
        wv0 = wvf[:, 0]
        mv0 = mvf[:, 0]
        # normal_dist = tf.distributions.Normal(loc = mmu, scale = tf.exp(0.5*mlog_sigma))
        # goal0 = tf.squeeze(normal_dist.sample(1), axis = 0)
        # print(goal0.get_shape())
        # neglogpg0 = - normal_dist.log_prob(goal0)

        self.w_initial_state, self.m_initial_state = None, None

        if self.use_vae:

            def step(**kwargs):
                ma = sess.run(ma0, feed_dict={mh: kwargs['mhs'], istraining: False})
                mv = sess.run(mv0, feed_dict={mh: kwargs['mhs'], mA: ma, istraining: False})
                goal_mask = np.asarray(kwargs['goal_mask'], dtype=np.bool)
                ma = kwargs['origin_goal']*np.expand_dims(1.-goal_mask,axis=1) + ma*np.expand_dims(goal_mask,axis=1)
                wa, wv, wneglogp = sess.run([wa0, wv0, wneglogp0], feed_dict={wh: kwargs['whs'], G: ma, istraining: False})
                return ma, mv, self.m_initial_state, wa, wv, self.w_initial_state, wneglogp

            def single_layer_step(whs, *_args, **_kwargs):
                wa, wv, wneglogp = sess.run([wa0, wv0, wneglogp0], {wh: whs, G: np.zeros(shape=whs.shape, dtype=whs.dtype), istraining: False})
                return wa, wv, self.w_initial_state, wneglogp

            def practice(whs, goal, *_args, **_kwargs):
                wa, wv, wneglogp = sess.run([wa0, wv0, wneglogp0], {wh: whs, G: goal, istraining: False})
                return wa, wv, self.w_initial_state, wneglogp

            uniform_dist = tf.ones(shape=wpi.get_shape(), dtype=tf.float32) / float(nact)
            a_uniform = tf.squeeze(tf.multinomial(uniform_dist, num_samples=1), axis=-1)
            def exploration(obs, phase):
                if phase == 'train_vae':
                    return sess.run([a_uniform , wh_decoder], feed_dict = {wX:obs})
                elif phase == 'train_gan':
                    a_, g_ = sess.run([a_uniform, g], feed_dict={Z:np.random.uniform(-4.4, 4.4, size=[len(obs), cell])})
                    return [a_, sess.run(wh_decoder, feed_dict = {wh_encoder:g_.reshape(-1, 2, 2, 64)})]
                return None

            def wvalue(whs, goal, *_args, **_kwargs):
                return sess.run(wv0, {wh: whs, G: goal, istraining: False})

            def mvalue(mhs, *_args, **_kwargs):
                ma = sess.run(ma0, feed_dict={mh: mhs, istraining: False})
                return sess.run(mv0, feed_dict={mh: mhs, mA: ma, istraining: False})

            def goal_generator():
                z = np.random.uniform(-4.4, 4.4, size=(nenv, cell))
                g_ = sess.run(g, feed_dict = {Z:z, istraining:False})
                return g_

            def embedding_decoder(h):
                '''
                :param goal: shape is [bs, 256]
                :return: a [bs, nh, nw, nc] tensor
                '''
                h_ = h.reshape(-1, 2, 2, 64)
                return sess.run(wh_decoder, feed_dict={wh_encoder: h_})

            self.exploration = exploration
            self.practice = practice
            self.goal_generator = goal_generator
            self.embedding_decoder = embedding_decoder

        else:

            def step(**kwargs):
                ma = sess.run(ma0, feed_dict={mX: kwargs['wobs'], istraining: False})
                mv = sess.run(mv0, feed_dict={mX: kwargs['mobs'], mA: ma, istraining: False})
                goal_mask = np.asarray(kwargs['goal_mask'], dtype=np.bool)
                ma = kwargs['origin_goal']*np.expand_dims(1.-goal_mask,axis=1) + ma*np.expand_dims(goal_mask,axis=1)
                wa, wv, wneglogp = sess.run([wa0, wv0, wneglogp0], feed_dict={wX: kwargs['wobs'], G: ma, istraining: False})
                return ma, mv, self.m_initial_state, wa, wv, self.w_initial_state, wneglogp

            def single_layer_step(obs, *_args, **_kwargs):
                wa, wv, wneglogp = sess.run([wa0, wv0, wneglogp0], {wX: obs, G: np.zeros(shape=(nenv, cell), dtype=np.float32), istraining: False})
                return wa, wv, self.w_initial_state, wneglogp

            def wvalue(wobs, goal, *_args, **_kwargs):
                return sess.run(wv0, {wX: wobs, G: goal, istraining: False})

            def mvalue(mobs, *_args, **_kwargs):
                ma = sess.run(ma0, feed_dict={mX: mobs, istraining: False})
                return sess.run(mv0, feed_dict={mX: mobs, mA: ma, istraining: False})


        def get_mh(mobs):
            return sess.run(mh, feed_dict = {mX:mobs, istraining:False})

        def get_wh(wobs):
            return sess.run(wh, feed_dict = {wX:wobs, istraining:False})

        self.wX = wX
        self.G = G

        self.mX = mX
        self.mA = mA
        self.Z = Z

        self.istraining = istraining

        self.wX_ = wX_
        self.mX_ = mX_
        self.ma = ma0
        self.wpi = wpi
        self.mvf = mvf
        self.wvf = wvf
        self.mh = mh
        self.wh = wh

        self.step = step
        self.single_layer_step = single_layer_step
        self.wvalue = wvalue
        self.mvalue = mvalue
        self.get_mh = get_mh
        self.get_wh = get_wh

class CnnPolicyIB(object):
    '''
    cnn policy with information bottleneck
    '''
    def __init__(self, sess, ob_space, ac_space, nbatch, master_ts, worker_ts, cell=256, reuse=tf.AUTO_REUSE):

        nenv = nbatch // (master_ts * worker_ts)
        self.nenv = nenv
        nh, nw, nc = ob_space.shape
        nact = ac_space.n

        channels = 1
        stddev = 0.1
        wX = tf.placeholder(tf.uint8, [nbatch, nh, nw, nc])#worker obs
        istraining = tf.placeholder(tf.bool, shape=[])
        noise = tf.random_normal(shape=[nbatch, 20, 20, channels], stddev=stddev)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu
            wX_ = tf.cast(wX, tf.float32) / 255.

            def encoder(X, noise_, scope = 'ordinary_encoder', reuse=reuse):
                with tf.variable_scope(scope, reuse=reuse):
                    h1 = activ(conv(X, 'c1', nf=16, rf=8, stride=4,init_scale=np.sqrt(2)))  # h, w = 20, 20
                    h1 = tf.concat([h1, noise_], axis=-1)
                    h2 = activ(conv(h1, 'c2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2))) # h, w = 9, 9
                    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=2, init_scale=np.sqrt(2)))  # h, w = 4, 4
                    h4 = activ(conv(h3, 'c4', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))  # h, w = 2, 2
                    return conv_to_fc(h4)
            wh = encoder(wX_, noise)

            def pv_net(wh, scope = 'worker_module', reuse=reuse):
                with tf.variable_scope(scope, reuse=reuse):
                    with tf.variable_scope('comm'):
                        wh_ = activ(fc(wh, 'wh_', nh=cell, init_scale=np.sqrt(2.)))
                    with tf.variable_scope('w_policy'):
                        wpi = fc(wh_, 'wpi', nact, init_scale=0.01) # shape = [nbatch, M, nact]
                    with tf.variable_scope('w_value'):
                        wvf = fc(wh_, 'wv', 1) #shape = [nbatch, M, 1]
                return wpi, wvf
            wpi, wvf = pv_net(wh)

            def T(X, H, scope = 'T', reuse=reuse):
                with tf.variable_scope(scope, reuse=reuse):
                    with tf.variable_scope('update_params'):
                        tx1 = tf.nn.elu(conv(X, 'tx1', nf=16, rf=8, stride=4, init_scale=np.sqrt(2)))
                        tx2 = tf.nn.elu(conv(tx1, 'tx2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2)))
                        tx3 = tf.nn.elu(conv(tx2, 'tx3', nf=64, rf=3, stride=2, init_scale=np.sqrt(2)))
                        tx3_flatten = conv_to_fc(tx3)
                        w_tx = tf.get_variable('w_tx', [1024, cell], initializer=ortho_init(np.sqrt(2)))
                        w_th = tf.get_variable('w_th', [cell, cell], initializer=ortho_init(np.sqrt(2.)))
                        w_tb = tf.get_variable('w_tb', [cell], initializer=tf.constant_initializer(0.0))
                        t1 = tf.nn.elu(tf.matmul(tx3_flatten, w_tx) + tf.matmul(H, w_th) + w_tb)
                        t2 = tf.nn.elu(tf.layers.batch_normalization(fc(t1, 't2', nh=cell), training=istraining, name='tbn2'))
                        t3 = fc(t2, 't3', nh=1)
                    with tf.variable_scope('orig_params'):
                        otx1 = tf.nn.elu(conv(X, 'otx1', nf=16, rf=8, stride=4, init_scale=np.sqrt(2)))
                        otx2 = tf.nn.elu(conv(otx1, 'otx2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2)))
                        otx3 = tf.nn.elu(conv(otx2, 'otx3', nf=64, rf=3, stride=2, init_scale=np.sqrt(2)))
                        otx3_flatten = conv_to_fc(otx3)
                        ow_tx = tf.get_variable('ow_tx', [1024, cell], initializer=ortho_init(np.sqrt(2)))
                        ow_th = tf.get_variable('ow_th', [cell, cell], initializer=ortho_init(np.sqrt(2.)))
                        ow_tb = tf.get_variable('ow_tb', [cell], initializer=tf.constant_initializer(0.0))
                        ot1 = tf.nn.elu(tf.matmul(otx3_flatten, ow_tx) + tf.matmul(H, ow_th) + ow_tb)
                        ot2 = tf.nn.elu(tf.layers.batch_normalization(fc(ot1, 'ot2', nh=cell), training=istraining, name='otbn2'))
                        ot3 = fc(ot2, 'ot3', nh=1)
                return t3
            idx = tf.reshape(tf.range(start=0, limit=wh.get_shape()[0].value, dtype=tf.int32), [-1, 1])
            idx_shuffle = tf.random_shuffle(idx)
            wh_shuffle = tf.gather_nd(wh, indices=idx_shuffle)
            joint_T, marginal_T = T(wX_, wh), tf.exp(T(wX_, wh_shuffle))
            mi_xh_loss = tf.reduce_mean(joint_T) - tf.log(tf.reduce_mean(marginal_T))#mutual information between X and wh
            self.T_value = tf.reduce_mean(joint_T)

        self.wpdtype = make_pdtype(ac_space)
        self.wpd = self.wpdtype.pdfromflat(wpi)
        wa0 = self.wpd.sample()
        wneglogp0 = self.wpd.neglogp(wa0)
        wv0 = wvf[:, 0]

        self.w_initial_state = None

        def step(obs, *_args, **_kwargs):
            n = sess.run(noise)
            wa, wv, wneglogp, whs = sess.run([wa0, wv0, wneglogp0, wh], {wX: obs, istraining: False, noise:n})
            return n, wa, wv, self.w_initial_state, wneglogp, whs

        def wvalue(wobs, *_args, **_kwargs):
            # noise = generate_noise(size=(nbatch, 20, 20, 1))
            return sess.run(wv0, {wX: wobs, istraining: False})

        def get_wh(wobs, noises):
            # noise = generate_noise(size=(nbatch, 20, 20, 1))
            return sess.run(wh, feed_dict = {wX:wobs, istraining:False, noise:noises})

        def get_noise():
            return sess.run(noise)

        self.wX = wX
        self.istraining = istraining
        self.noise = noise
        self.wX_ = wX_

        self.wpi = wpi
        self.wvf = wvf
        self.wh = wh
        # self.mi = mi
        self.mi_xh_loss = mi_xh_loss

        self.step = step
        self.wvalue = wvalue
        self.get_wh = get_wh
        self.get_noise = get_noise

class CnnPolicyVaeA2C(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, master_ts, worker_ts, cell=256, reuse=tf.AUTO_REUSE):

        nenv = nbatch // (master_ts * worker_ts)
        self.nenv = nenv
        nh, nw, nc = ob_space.shape
        nact = ac_space.n

        wX = tf.placeholder(tf.uint8, [nbatch, nh, nw, nc])#worker obs
        istraining = tf.placeholder(tf.bool, shape=[])

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu
            wX_ = tf.cast(wX, tf.float32) / 255.

            def encoder(X, scope = 'ordinary_encoder', reuse=reuse):
                with tf.variable_scope(scope, reuse=reuse):
                    h1 = activ(conv(X, 'c1', nf=16, rf=8, stride=4,init_scale=np.sqrt(2)))  # h, w = 20, 20
                    h2 = activ(conv(h1, 'c2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2)))  # h, w = 9, 9
                    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=2, init_scale=np.sqrt(2)))  # h, w = 4, 4
                    h4 = activ(conv(h3, 'c4', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))) # h, w = 2, 2
                    return conv_to_fc(h4)

            def decoder(X, scope='vae_decoder', reuse=reuse):
                with tf.variable_scope(scope, reuse=reuse):
                    X_ = tf.reshape(X, [-1, 2, 2 ,64])
                    whd4 = activ(deconv(X_, 'cd4', nf=64, rf=3, stride=1, output_shape=[nbatch, 4, 4, 64], init_scale=np.sqrt(2)))
                    whd3 = activ(deconv(whd4, 'cd3', nf=32, rf=3, stride=2, output_shape=[nbatch, 9, 9, 32], init_scale=np.sqrt(2)))
                    whd2 = activ(deconv(whd3, 'cd2', nf=16, rf=4, stride=2, output_shape=[nbatch, 20, 20, 16], init_scale=np.sqrt(2)))
                    wh_decoder = activ(deconv(whd2, 'cd1', nf=nc, rf=8, stride=4, output_shape=[nbatch, nh, nw, nc], init_scale=np.sqrt(2)))
                    return wh_decoder

            wh_encoder = encoder(wX_)
            wh_decoder = decoder(wh_encoder)
            self.encoder = wh_encoder
            self.decoder = wh_decoder

            with tf.variable_scope('worker_module'):
                with tf.variable_scope('w_policy'):
                    wpi = fc(wh_encoder, 'wpi', nact)
                with tf.variable_scope('w_value'):
                    wvf = fc(wh_encoder, 'wv', 1)

        self.wpdtype = make_pdtype(ac_space)
        self.wpd = self.wpdtype.pdfromflat(wpi)
        wa0 = self.wpd.sample()
        wneglogp0 = self.wpd.neglogp(wa0)
        wv0 = wvf[:, 0]

        self.w_initial_state = None

        def step(algo, *_args, **_kwargs):
            if algo == 'vae_a2c':
                wa, wv, wneglogp, whe, whd = sess.run([wa0, wv0, wneglogp0, wh_encoder, wh_decoder], {wh_encoder: _kwargs['whs'], istraining: False})
                return wa, wv, self.w_initial_state, wneglogp
            else:
                wa, wv, wneglogp, whe = sess.run([wa0, wv0, wneglogp0, wh_encoder, wh_decoder], {wX: _kwargs['wobs'], istraining: False})
                return wa, wv, self.w_initial_state, wneglogp

        uniform_dist = tf.ones(shape=wpi.get_shape(), dtype=tf.float32) / float(nact)
        a_uniform = tf.squeeze(tf.multinomial(uniform_dist, num_samples=1), axis=-1)
        def exploration(obs):
            return sess.run([a_uniform , wh_decoder], feed_dict = {wX: obs, istraining: False})

        def wvalue(algo, *_args, **_kwargs):
            if algo == 'vae_a2c':
                return sess.run(wv0, {wh_encoder: _kwargs['whs'], istraining: False})
            else:
                return sess.run(wv0, {wX: _kwargs['wobs'], istraining: False})

        def embedding(wobs):
            return sess.run(wh_encoder, feed_dict={wX: wobs, istraining:False})

        def embedding_decoder(whs):
            return sess.run(wh_decoder, feed_dict={wh_encoder: whs, istraining: False})

        self.wX = wX
        self.istraining = istraining
        self.wX_ = wX_
        self.wpi = wpi
        self.wvf = wvf
        self.step = step
        self.wvalue = wvalue
        self.exploration = exploration
        self.embedding = embedding
        self.embedding_decoder = embedding_decoder

class CnnPolicy(object):
    '''
    cnn policy
    '''
    def __init__(self, sess, ob_space, ac_space, nbatch, master_ts, worker_ts, cell=256,
                 reuse=tf.AUTO_REUSE, model='step_model', algo='regular'):

        nenv = nbatch // (master_ts * worker_ts)
        self.nenv = nenv
        nh, nw, nc = ob_space.shape
        nact = ac_space.n

        wX = tf.placeholder(tf.uint8, [nbatch, nh, nw, nc])#worker obs

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu
            wX_ = tf.cast(wX, tf.float32) / 255.

            def encoder(X, scope = 'ordinary_encoder', reuse=reuse):
                with tf.variable_scope(scope, reuse=reuse):
                    h1 = activ(conv(X, 'c1', nf=16, rf=8, stride=4,init_scale=np.sqrt(2)))  # h, w = 20, 20
                    h2 = activ(conv(h1, 'c2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2))) # h, w = 9, 9
                    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=2, init_scale=np.sqrt(2)))  # h, w = 4, 4
                    h4 = 0
                    encoding = 0
                    try:
                        if algo == 'regular':
                            h4 = activ(conv(h3, 'c4', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))  # h, w = 2, 2
                        elif algo == 'VIB':
                            '''
                            implement VIB here, there should be two variables generated by h3: mu and rho,
                            you can refer to the following pseudo code:
                                mu = activ(conv(h3, 'mu', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
                                rho = activ(conv(h3, 'rho', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
                                encoding = ds.NormalWithSoftplusScale(mu, rho - 5.0)
                                h4 = encoding.sample()
                            '''
                            mu = activ(conv(h3, 'mu', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
                            rho = activ(conv(h3, 'rho', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
                            encoding = ds.NormalWithSoftplusScale(mu, rho - 5.0)
                            h4 = encoding.sample()
                            # pass
                        else:
                            raise Exception('Algorithm not exists')
                    except Exception as e:
                        print(e)
                    return conv_to_fc(h4), encoding

            wh, encoding = encoder(wX_)

            def pv_net(wh, scope = 'worker_module', reuse=reuse):
                with tf.variable_scope(scope, reuse=reuse):
                    with tf.variable_scope('comm'):
                        wh_ = activ(fc(wh, 'wh_', nh=cell, init_scale=np.sqrt(2.)))
                    with tf.variable_scope('w_policy'):
                        wpi = fc(wh_, 'wpi', nact, init_scale=0.01) # shape = [nbatch, M, nact]
                    with tf.variable_scope('w_value'):
                        wvf = fc(wh_, 'wv', 1) #shape = [nbatch, M, 1]
                return wpi, wvf

            wpi, wvf = pv_net(wh)

        self.wpdtype = make_pdtype(ac_space)
        self.wpd = self.wpdtype.pdfromflat(wpi)
        wa0 = self.wpd.sample()
        wneglogp0 = self.wpd.neglogp(wa0)
        wv0 = wvf[:, 0]

        self.w_initial_state = None

        def step(wobs, *_args, **_kwargs):
            wa, wv, wneglogp = sess.run([wa0, wv0, wneglogp0], {wX: wobs})
            return wa, wv, self.w_initial_state, wneglogp

        def wvalue(wobs, *_args, **_kwargs):
            return sess.run(wv0, {wX: wobs})

        def get_wh(wobs, *_args, **_kwargs):
            return sess.run(wh, feed_dict = {wX:wobs})

        self.wX = wX
        self.wX_ = wX_
        self.wpi = wpi
        self.wvf = wvf
        self.wh, self.encoding = wh, encoding

        self.step = step
        self.wvalue = wvalue
        self.get_wh = get_wh

class CnnPolicySVIB(object):
    '''
    cnn policy with stein variational information bottleneck
    '''

    def __init__(self, sess, ob_space, ac_space, nbatch, master_ts, worker_ts, cell=256,
                 reuse=tf.AUTO_REUSE, M=32, model='step_model', algo='use_svib_uniform'):

        nenv = nbatch // (master_ts * worker_ts)
        self.nenv = nenv
        nh, nw, nc = ob_space.shape
        nact = ac_space.n

        channels = 1
        stddev = 0.1
        wX = tf.placeholder(tf.uint8, [nbatch, nh, nw, nc])#worker obs
        istraining = tf.placeholder(tf.bool, shape=[])
        NOISE_KEEP = tf.placeholder(tf.float32, shape=[nbatch, 20, 20, channels])
        noise = tf.random_normal(shape=[nbatch, 20, 20, channels], stddev=stddev)
        noise_expand = tf.random_normal(shape=[nbatch*(M-1), 20, 20, channels], stddev=stddev)
        noise_total = tf.concat([tf.expand_dims(NOISE_KEEP, axis=1), tf.reshape(noise_expand, [nbatch, M-1, 20, 20, channels])], axis=1)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu
            wX_ = tf.cast(wX, tf.float32) / 255.

            def encoder(X, noise_, scope = 'ordinary_encoder', reuse=reuse):
                with tf.variable_scope(scope, reuse=reuse):
                    h1 = activ(conv(X, 'c1', nf=16, rf=8, stride=4,init_scale=np.sqrt(2)))  # h, w = 20, 20
                    if algo != 'regular':
                        h1 = tf.concat([h1, noise_], axis=-1)
                    h2 = activ(conv(h1, 'c2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2))) # h, w = 9, 9
                    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=2, init_scale=np.sqrt(2)))  # h, w = 4, 4
                    h4 = activ(conv(h3, 'c4', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))  # h, w = 2, 2
                    return conv_to_fc(h4)

            wh = encoder(wX_, noise)

            def pv_net(wh, scope = 'worker_module', reuse=reuse):
                with tf.variable_scope(scope, reuse=reuse):
                    with tf.variable_scope('comm'):
                        wh_ = activ(fc(wh, 'wh_', nh=cell, init_scale=np.sqrt(2.)))
                    with tf.variable_scope('w_policy'):
                        wpi = fc(wh_, 'wpi', nact, init_scale=0.01) # shape = [nbatch, M, nact]
                    with tf.variable_scope('w_value'):
                        wvf = fc(wh_, 'wv', 1) #shape = [nbatch, M, 1]
                return wpi, wvf

            wpi, wvf = pv_net(wh)

            if model == 'train_model':
                wX_expand = tf.reshape(tf.tile(tf.expand_dims(wX_, axis=1), [1, M, 1, 1, 1]), [nbatch*M, nh, nw, nc])
                wh_expand = tf.reshape(encoder(wX_expand, tf.reshape(noise_total, [nbatch*M, 20, 20, 1])), [nbatch, M, -1])  # shape=[nbatch, M, cell]
                self.wh_expand = wh_expand#, wh_expand_repr_train
                wpi_expand, wvf_expand = pv_net(tf.reshape(wh_expand, [nbatch*M, -1]))
                wpi_expand, wvf_expand = tf.reshape(wpi_expand, [nbatch, M, -1]), tf.reshape(wvf_expand, [nbatch, M, -1])
                self.wpi_expand, self.wvf_expand = wpi_expand, wvf_expand

                def h_coef(Z_expand, M=M):
                    '''
                    :param Z_expand: shape = [nbatch, M, cell]
                    :param M:
                    :return: h coefficients of RPF kernel
                    '''
                    rpf_hs = tf.constant(0., dtype=tf.float32, shape=[Z_expand.get_shape()[0].value, 0])
                    for i in range(M - 1):
                        for j in range(i + 1, M):
                            distance = tf.reduce_sum(tf.square(Z_expand[:, i, :] - Z_expand[:, j, :]), axis=1, keep_dims=True)  # shape=[nbatch, 1]
                            rpf_hs = tf.concat([rpf_hs, distance], axis=1)
                    values, idx = tf.nn.top_k(rpf_hs, k=rpf_hs.get_shape()[-1].value // 2) # shape=[nbatch, M(M-1)/4]
                    rpf_h = values[:, -1]  # shape=[nbatch]
                    return tf.stop_gradient(0.5*rpf_h/tf.log(float(M)+1.))#prevent denominator to be 0

                rpf_h = h_coef(wh_expand)

                def rpf_kernel(Z_expand, rpf_h, M=M):
                    '''
                    :param Z_expand: shape=[nbatch, M, cell]
                    :param Z_expand_repr_train: shape=[nbatch, M, cell]
                    :param rpf_h:
                    :return: a [nbatch, M, M] kernel matrix and its gradients towards Z_expand_column_dim
                    '''
                    Z_expand_row_dim = tf.expand_dims(Z_expand, axis=1)#shape=[nbatch,1,M,cell]
                    Z_expand_column_dim = tf.expand_dims(Z_expand, axis=2)#shape=[nbatch,M,1,cell]
                    delta = Z_expand_column_dim - Z_expand_row_dim#shape=[nbatch, M, M, cell]
                    delta_square = tf.reduce_sum(tf.square(delta), axis=-1)#shape=[nbatch, M, M]
                    rpf_h_expand = tf.reshape(rpf_h, [-1, 1, 1])
                    rpf_matrix = tf.exp(-delta_square/rpf_h_expand)#shape=[nbatch, M, M]
                    rpf_grads = tf.constant(0., dtype=tf.float32, shape=[Z_expand.get_shape()[0].value, 0, Z_expand.get_shape()[2].value])
                    for i in range(M):
                        rpf_grad = tf.reduce_mean(tf.gradients(rpf_matrix[:, :, i], [Z_expand_column_dim])[0], axis=1)
                        rpf_grads = tf.concat([rpf_grads, rpf_grad], axis=1)
                    return tf.stop_gradient(rpf_matrix), tf.stop_gradient(rpf_grads)#rpf_grads.shape=[nbatch, M, cell]

                rpf_matrix, rpf_grads = rpf_kernel(wh_expand, rpf_h)

                self.rpf_h = rpf_h
                self.rpf_matrix = rpf_matrix
                self.rpf_grads = rpf_grads

                # def T(X, H, scope='T', reuse=reuse):
                #     with tf.variable_scope(scope, reuse=reuse):
                #         with tf.variable_scope('update_params'):
                #             tx1 = tf.nn.elu(conv(X, 'tx1', nf=16, rf=8, stride=4, init_scale=np.sqrt(2)))
                #             tx2 = tf.nn.elu(conv(tx1, 'tx2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2)))
                #             tx3 = tf.nn.elu(conv(tx2, 'tx3', nf=64, rf=3, stride=2, init_scale=np.sqrt(2)))
                #             tx3_flatten = conv_to_fc(tx3)
                #             w_tx = tf.get_variable('w_tx', [1024, cell], initializer=ortho_init(np.sqrt(2)))
                #             w_th = tf.get_variable('w_th', [cell, cell], initializer=ortho_init(np.sqrt(2.)))
                #             w_tb = tf.get_variable('w_tb', [cell], initializer=tf.constant_initializer(0.0))
                #             t1 = tf.nn.elu(tf.matmul(tx3_flatten, w_tx) + tf.matmul(H, w_th) + w_tb)
                #             t2 = tf.nn.elu(
                #                 tf.layers.batch_normalization(fc(t1, 't2', nh=cell), training=istraining, name='tbn2'))
                #             t3 = fc(t2, 't3', nh=1)
                #         with tf.variable_scope('orig_params'):
                #             otx1 = tf.nn.elu(conv(X, 'otx1', nf=16, rf=8, stride=4, init_scale=np.sqrt(2)))
                #             otx2 = tf.nn.elu(conv(otx1, 'otx2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2)))
                #             otx3 = tf.nn.elu(conv(otx2, 'otx3', nf=64, rf=3, stride=2, init_scale=np.sqrt(2)))
                #             otx3_flatten = conv_to_fc(otx3)
                #             ow_tx = tf.get_variable('ow_tx', [1024, cell], initializer=ortho_init(np.sqrt(2)))
                #             ow_th = tf.get_variable('ow_th', [cell, cell], initializer=ortho_init(np.sqrt(2.)))
                #             ow_tb = tf.get_variable('ow_tb', [cell], initializer=tf.constant_initializer(0.0))
                #             ot1 = tf.nn.elu(tf.matmul(otx3_flatten, ow_tx) + tf.matmul(H, ow_th) + ow_tb)
                #             ot2 = tf.nn.elu(tf.layers.batch_normalization(fc(ot1, 'ot2', nh=cell), training=istraining,
                #                                                           name='otbn2'))
                #             ot3 = fc(ot2, 'ot3', nh=1)
                #     return t3

                # idx = tf.reshape(tf.range(start=0, limit=wh.get_shape()[0].value, dtype=tf.int32), [-1, 1])
                # idx_shuffle = tf.random_shuffle(idx)
                # wh_shuffle = tf.gather_nd(wh, indices=idx_shuffle)
                # joint_T, marginal_T = T(wX_, wh), tf.exp(T(wX_, wh_shuffle))
                # self.mi_xh_loss = tf.reduce_mean(joint_T) - tf.log(
                #     tf.reduce_mean(marginal_T))  # mutual information between X and wh
                # self.T_value = tf.reduce_mean(joint_T)


        self.wpdtype = make_pdtype(ac_space)
        self.wpd = self.wpdtype.pdfromflat(wpi)
        wa0 = self.wpd.sample()
        wneglogp0 = self.wpd.neglogp(wa0)
        wv0 = wvf[:, 0]

        self.w_initial_state = None

        def step(obs, *_args, **_kwargs):
            # noise = generate_noise(size=(nbatch, 20, 20, 1))
            n = sess.run(noise)
            wa, wv, wneglogp, whs = sess.run([wa0, wv0, wneglogp0, wh], {wX: obs, istraining: False, noise:n})
            return n, wa, wv, self.w_initial_state, wneglogp, whs
            # wa, wv, wneglogp = sess.run([wa0, wv0, wneglogp0], {wX: obs, istraining: False})
            # return wa, wv, self.w_initial_state, wneglogp

        def wvalue(wobs, *_args, **_kwargs):
            # noise = generate_noise(size=(nbatch, 20, 20, 1))
            return sess.run(wv0, {wX: wobs, istraining: False})

        def get_wh(wobs, noises):
            # noise = generate_noise(size=(nbatch, 20, 20, 1))
            return sess.run(wh, feed_dict = {wX:wobs, istraining:False, noise:noises})

        def get_noise():
            return sess.run(noise)

        self.wX = wX
        self.istraining = istraining
        self.NOISE_KEEP = NOISE_KEEP
        self.noise = noise
        self.noise_expand = noise_expand#, noise_expand_repr_train

        self.wX_ = wX_
        self.wpi = wpi
        self.wvf = wvf
        self.wh = wh

        # self.rpf_kernel = rpf_kernel
        self.step = step
        self.wvalue = wvalue
        self.get_wh = get_wh
        self.get_noise = get_noise