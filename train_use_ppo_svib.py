import os
import time
import datetime
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance

import sys
from baselines.common import set_global_seeds, tf_util
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
# from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
import multiprocessing

import argparse
from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from policy import CnnPolicySVIB
from baselines.a2c.utils import cat_entropy, make_path
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
from utils import grad_clip, tf_l2norm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
env_name = "BreakoutNoFrameskip-v1"

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm,
                 cell=256, sv_M=32, algo='regular', ib_alpha=1e-3):
        sess = tf_util.make_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, 1, cell = cell, M=sv_M, model='step_model', algo=algo)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, 1, nsteps, cell = cell, M=sv_M, model='train_model', algo=algo)

        A = train_model.wpdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC_expand = tf.placeholder(tf.float32, [None, sv_M])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        OLDVPRED_expand = tf.placeholder(tf.float32, [None, sv_M])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        if algo == 'use_svib_uniform' or algo == 'use_svib_gaussian':
            def expand_placeholder(X, M=sv_M):
                return tf.tile(tf.expand_dims(X, axis=-1), [1, M])
            A_expand, R_expand = expand_placeholder(A), expand_placeholder(R)
            neglogpac_expand = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.wpi_expand, labels=A_expand)#shape=[nbatch, sv_M]
            entropy_expand = tf.reduce_mean(cat_entropy(train_model.wpi_expand), axis=-1)#shape=[nbatch]
            vpred_expand = train_model.wvf_expand[:,:,0]
            vpredclipped_expand = OLDVPRED_expand + tf.clip_by_value(train_model.wvf_expand[:,:,0]-OLDVPRED_expand, -CLIPRANGE, CLIPRANGE)
            vf_loss1_expand = tf.square(vpred_expand - R_expand)
            vf_loss2_expand = tf.square(vpredclipped_expand - R_expand)
            vf_loss_expand = .5*tf.reduce_mean(tf.maximum(vf_loss1_expand, vf_loss2_expand) ,axis=-1)#shape = [nbatch]
            ratio_expand = tf.exp(OLDNEGLOGPAC_expand - neglogpac_expand)
            ADV_expand = R_expand - OLDVPRED_expand
            # ADV_expand_mean, ADV_expand_var = tf.nn.moments(ADV_expand, axes=0, keep_dims=True)#shape = [1,sv_M]
            ADV_expand_mean, ADV_expand_var = tf.nn.moments(ADV_expand, axes=[0,1])#shape = [1,sv_M]
            ADV_expand_normal = (ADV_expand-ADV_expand_mean)/(tf.sqrt(ADV_expand_var)+1e-8)
            pg_losses_expand = -ADV_expand_normal * ratio_expand
            pg_losses2_expand = -ADV_expand_normal * tf.clip_by_value(ratio_expand, 1.-CLIPRANGE, 1.+CLIPRANGE)
            pg_loss_expand = tf.reduce_mean(tf.maximum(pg_losses_expand, pg_losses2_expand),axis=-1)
            J_theta = -(pg_loss_expand + vf_coef*vf_loss_expand - ent_coef*entropy_expand)

            loss_expand = -J_theta/float(nbatch_train)
            pg_loss_expand_ = tf.reduce_mean(pg_loss_expand)
            vf_loss_expand_ = tf.reduce_mean(vf_loss_expand)
            entropy_expand_ = tf.reduce_mean(entropy_expand)

            log_p_grads = tf.gradients(J_theta/np.sqrt(ib_alpha), [train_model.wh_expand])[0]#shape=[nbatch, sv_M, cell]
            if algo == 'use_svib_gaussian':
                mean, var = tf.nn.moments(train_model.wh_expand, axes=1, keep_dims=True)#shape=[nbatch, 1,cell]
                gaussian_grad = -(train_model.wh_expand - mean)/(float(sv_M) * (var+1e-3))
                log_p_grads += 5e-3*(tf_l2norm(log_p_grads, axis=-1, keep_dims=True)/tf_l2norm(gaussian_grad, axis=-1, keep_dims=True))*gaussian_grad
            sv_grads = tf.constant(0., tf.float32, shape=[nbatch_train, 0, cell])
            exploit_total_norm_square = 0
            explore_total_norm_square = 0
            explore_coef = 1.
            if env_name == 'SeaquestNoFrameskip-v4':
                explore_coef = 0.01
            elif env_name == 'AirRaidNoFrameskip-v4':
                explore_coef = 0.
            print('env_name:', env_name)
            for i in range(sv_M):
                exploit = tf.reduce_sum(train_model.rpf_matrix[:, :, i:i + 1] * log_p_grads, axis=1)
                explore = np.sqrt(ib_alpha)*explore_coef*train_model.rpf_grads[:, i, :]
                exploit_total_norm_square += tf.square(tf_l2norm(exploit, axis=-1, keep_dims=False))
                explore_total_norm_square += tf.square(tf_l2norm(explore, axis=-1, keep_dims=False))
                sv_grad = exploit + explore#shape=[nbatch, cell]
                sv_grads = tf.concat([sv_grads, tf.expand_dims(sv_grad, axis=1)], axis=1)
            SV_GRADS = tf.placeholder(tf.float32, [nbatch_train, sv_M, cell])
            repr_loss = tf.reduce_mean(SV_GRADS * train_model.wh_expand, axis=1)#shape=[nbatch,cell]
            repr_loss = -tf.reduce_mean(tf.reduce_sum(repr_loss, axis=-1))#max optimization problem to minimization problem

            #op for debugging and visualization
            exploit_explore_ratio = tf.sqrt(exploit_total_norm_square/tf.maximum(explore_total_norm_square, 0.01))[0]
            # rpf_mat = tf.expand_dims(train_model.rpf_matrix, axis=-1)
            # log_p_grads_tile = tf.tile(tf.expand_dims(log_p_grads, axis=2), [1,1,sv_M,1])
            # exploit = tf.reduce_sum(rpf_mat*log_p_grads_tile, axis=1)
            # explore = np.sqrt(ib_alpha) * train_model.rpf_grads
            # sv_grads = exploit + explore
            # ind = 1
            # exploit = tf.reduce_sum(train_model.rpf_matrix[:, :, i:i + 1] * log_p_grads, axis=1)
            # explore = train_model.rpf_grads[:, i, :]
            # clip_coef = tf_l2norm(exploit, axis=-1, keep_dims=True)
            # explore_norm = tf_l2norm(explore, axis=-1, keep_dims=True)
            # explore = explore * 1e-2 * clip_coef / tf.maximum(explore_norm, clip_coef)
            # sv_grad = exploit + np.sqrt(ib_alpha) * explore  # shape=[nbatch, cell]

            grads_expand, global_norm_expand = grad_clip(loss_expand, max_grad_norm, ['model/worker_module'])
            trainer_expand = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            _train_expand = trainer_expand.apply_gradients(grads_expand)
            repr_grads, repr_global_norm = grad_clip(repr_loss, max_grad_norm, ['model/ordinary_encoder'])
            repr_trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            _repr_train = repr_trainer.apply_gradients(repr_grads)
        else:
            neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.wpi, labels=A)
            entropy = tf.reduce_mean(cat_entropy(train_model.wpi))
            vpred = train_model.wvf[:,0]
            vpredclipped = OLDVPRED + tf.clip_by_value(train_model.wvf[:,0] - OLDVPRED, - CLIPRANGE, CLIPRANGE)
            vf_losses1 = tf.square(vpred - R)
            vf_losses2 = tf.square(vpredclipped - R)
            vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
            ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
            pg_losses = -ADV * ratio
            pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
            loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

            grads, _grad_norm = grad_clip(loss, max_grad_norm, ['model/worker_module', 'model/ordinary_encoder'])
            trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            _train = trainer.apply_gradients(grads)

        with tf.variable_scope('model'):
            params = tf.trainable_variables()

        def generate_old_expand_data(obs, noises, masks, actions, states=None):
            noises_expand = sess.run(train_model.noise_expand)
            repr_td_map = {train_model.wX:obs, train_model.istraining:False, A:actions,
                           train_model.noise_expand:noises_expand, train_model.NOISE_KEEP:noises}
            if states is not None:
                repr_td_map[train_model.wS] = states
                repr_td_map[train_model.wM] = masks
            neglogpacs_expand, vpreds_expand = \
                sess.run([neglogpac_expand, vpred_expand], feed_dict=repr_td_map)
            shape = noises_expand.shape
            noises_expand = noises_expand.reshape(nbatch_train, sv_M-1, *shape[1:])
            return [noises_expand, neglogpacs_expand, vpreds_expand]

        def train(lr, cliprange, obs, noises, returns, masks, actions, values, neglogpacs,
                  noises_expand=None, neglogpacs_expand=None, vpreds_expand=None, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            if algo == 'use_svib_uniform' or algo == 'use_svib_gaussian':
                shape = noises_expand.shape
                noises_expand_ = noises_expand.reshape(nbatch_train*(sv_M-1), *shape[2:])
                # print(noises_expand_.shape)
                repr_td_map = {train_model.wX:obs, train_model.istraining:True, A:actions, R:returns, LR:lr, CLIPRANGE:cliprange,
                               train_model.noise_expand:noises_expand_, train_model.NOISE_KEEP:noises,
                               OLDNEGLOGPAC_expand:neglogpacs_expand, OLDVPRED_expand:vpreds_expand}
            rl_td_map = {train_model.istraining: True, A:actions, R:returns, LR:lr, CLIPRANGE:cliprange}
            if states is not None:
                if algo == 'use_svib_uniform' or algo == 'use_svib_gaussian':
                    repr_td_map[train_model.wS] = states
                    repr_td_map[train_model.wM] = masks
                rl_td_map[train_model.wS] = states
                rl_td_map[train_model.wM] = masks

            if algo == 'use_svib_uniform' or algo == 'use_svib_gaussian':
                sv_gradients, whs_expand, ir_ratio = sess.run([sv_grads, train_model.wh_expand, exploit_explore_ratio], feed_dict=repr_td_map)
                rl_td_map[OLDNEGLOGPAC_expand], rl_td_map[OLDVPRED_expand], rl_td_map[train_model.wh_expand] = neglogpacs_expand, vpreds_expand, whs_expand
                value_loss, policy_loss, policy_entropy, _ = sess.run(
                    [vf_loss_expand_, pg_loss_expand_, entropy_expand_, _train_expand],
                    feed_dict=rl_td_map
                )
                repr_td_map[SV_GRADS] = sv_gradients
                repr_grad_norm, represent_loss, __ = sess.run([repr_global_norm, repr_loss, _repr_train], feed_dict=repr_td_map)
            else:
                rl_td_map[train_model.wX], rl_td_map[train_model.noise] = obs, noises#noise won't be used when algo is 'regular'
                rl_td_map[OLDNEGLOGPAC], rl_td_map[OLDVPRED], rl_td_map[ADV] = neglogpacs, values, advs
                value_loss, policy_loss, policy_entropy, _ = sess.run(
                    [ vf_loss, pg_loss, entropy,  _train],
                    feed_dict=rl_td_map
                )
                represent_loss, rpf_norm_, rpf_grad_norm_, sv_gradients, ir_ratio = 0., 0., 0., 0., 0
            return policy_loss, value_loss, policy_entropy, represent_loss, ir_ratio
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'represent_loss', 'exploit_explore_ratio']

        def save(save_path):
            ps = sess.run(params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.generate_old_expand_data = generate_old_expand_data
        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.wvalue
        self.initial_state = act_model.w_initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

class test_Runner(object):
    def __init__(self, env, model, num_episodes=10):
        self.env = env
        self.model = model
        self.num_episodes = num_episodes
        obs = env.reset()
        self.obs = np.copy(obs)
        self.states = model.initial_state
        self.done = False
        self.dones = [False for _ in range(env.num_envs)]
        self.one_episode_r = 0.
        self.total_r = 0.

    def run(self):
        self.one_episode_r = 0.
        self.total_r = 0.
        for i in range(self.num_episodes):
            while not self.done:
                noises, actions, values, self.states, _, h= self.model.step(self.obs, self.states, self.dones)
                # actions, values, self.states, _= self.model.step(self.obs, self.states, self.dones)
                self.obs, rewards, self.dones, _ = self.env.step(actions)
                self.done = self.dones[0]
                self.one_episode_r += rewards
            self.total_r += self.one_episode_r
            self.one_episode_r = 0.
            self.done = False
        return (self.total_r[0] / float(self.num_episodes))

class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        nenv = env.num_envs
        obs = env.reset()
        self.obs = np.copy(obs)
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def run(self):
        mb_obs, mb_whs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_noises = [],[],[],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            noises, actions, values, self.states, neglogpacs, h = self.model.step(self.obs, self.states, self.dones)
            # actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_noises.append(noises)
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_whs.append(h)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_noises = np.asarray(mb_noises, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_whs = np.asarray(mb_whs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_noises, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_whs)),
            mb_states, epinfos)
# obs, noises, returns, masks, actions, values, neglogpacs, states, whs = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

def learn(*, policy, env, seed, test_env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
          cell=256, sv_M=26, algo='regular', ib_alpha=1e-3,
          load_path="saved_nets-data/hrl_a2c/%s/data" % start_time):

    tf.reset_default_graph()
    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    print("algorithm: ",algo)
    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, cell=cell, sv_M=sv_M, algo=algo, ib_alpha=ib_alpha)
    model = make_model()
    try:
        model.load(load_path)
    except Exception as e:
        print("no data to load!!"+str(e))
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    test_runner = test_Runner(env = test_env, model = model)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    prev_r = 0.
    exp_coef = .1
    up_units = 0.
    if env_name == 'PongNoFrameskip-v4':
        up_units = 21.
    reward_list = []
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, noises, returns, masks, actions, values, neglogpacs, whs, states, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)
        mblossvals = []
        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            if algo == 'use_svib_uniform' or algo == 'use_svib_gaussian':
                old_expand_datas = []
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, noises, masks, actions))
                    old_expand_datas.append(model.generate_old_expand_data(*slices))
                len_ = len(old_expand_datas[0])
                noises_expand, neglogpacs_expand, vpreds_expand =\
                    [np.concatenate([old_expand_datas[i][j] for i in range(nminibatches)],axis=0) for j in range(len_)]
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    if algo == 'use_svib_uniform' or algo == 'use_svib_gaussian':
                        slices = (arr[mbinds] for arr in (obs, noises, returns, masks, actions, values, neglogpacs,
                                                          noises_expand, neglogpacs_expand, vpreds_expand))
                    else:
                        slices = (arr[mbinds] for arr in (obs, noises, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version, notice that currently our implementation does not support recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, noises, returns, masks, actions, values, neglogpacs, whs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv("repr_mean", np.mean(whs[0]))
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
            if update % (40*log_interval) == 0 or update == 1:
                save_th = update//(40*log_interval)
                model.save("saved_nets-data/%s/ppo_svib/%s/%s/data" % (env_name, start_time, save_th))
                episode_r = exp_coef*(test_runner.run()+up_units) + (1-exp_coef)*prev_r
                prev_r = np.copy(episode_r)
                reward_list.append(episode_r)
                logger.logkv('episode_r', float(episode_r))
                logger.dumpkvs()
    env.close()
    return np.asarray(reward_list)

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def train(env_id, num_env, num_timesteps, seed, policy, algo='regular', ib_alpha=1e-3):
    # ncpu = multiprocessing.cpu_count()
    # if sys.platform == 'darwin': ncpu //= 2
    # config = tf.ConfigProto(allow_soft_placement=True,
    #                         intra_op_parallelism_threads=ncpu,
    #                         inter_op_parallelism_threads=ncpu)
    # config.gpu_options.allow_growth = True #pylint: disable=E1101
    # tf.Session(config=config).__enter__()

    env = VecFrameStack(make_atari_env(env_id, num_env, seed), 4)
    test_env = VecFrameStack(make_atari_env(env_id, num_env, seed+1), 4)
    policy = {'cnn_svib' : CnnPolicySVIB}[policy]
    reward_list = learn(policy=policy, env=env, seed=seed, test_env=test_env, nsteps=32, nminibatches=4,
                         lam=0.95, gamma=0.99, noptepochs=4, log_interval=10,
                         ent_coef=.01,
                         lr=lambda f : f * 2.5e-4,
                         cliprange=lambda f : f * 0.1,
                         total_timesteps=int(num_timesteps * 1.),
                        algo=algo, ib_alpha=ib_alpha)
    return reward_list

def config_log(FLAGS):
    logdir = "tensorboard/%s/ppo_svib/%s_lr%s_%s/%s_%s_%s_seed_%s" % (
        FLAGS.env,FLAGS.num_timesteps, '0.00025',FLAGS.policy, start_time, FLAGS.train_option, str(FLAGS.ib_alpha), str(FLAGS.seed))
    if FLAGS.log == "tensorboard":
        Logger.DEFAULT = Logger.CURRENT = Logger(dir=logdir, output_formats=[TensorBoardOutputFormat(logdir)])
    elif FLAGS.log == "stdout":
        Logger.DEFAULT = Logger.CURRENT = Logger(dir=logdir, output_formats=[HumanOutputFormat(sys.stdout)])

def train_all(algos, env_id, num_timesteps, seed, policy, num_env):
    all = pd.DataFrame([])
    for i in range(3):
        for algo in algos.keys():
            ib_alpha = algos[algo]
            reward_list = np.reshape(train(env_id, num_env=num_env, num_timesteps=num_timesteps,
                                       seed=seed+i, policy=policy, algo=algo, ib_alpha=ib_alpha), (-1, 1))
            len = reward_list.shape[0]
            data = pd.DataFrame(np.ones((len, 4)))
            data.columns = ['step', 'avg_reward', 'algorithm', 'seed']
            data['step'] = np.linspace(0, len-1, len).reshape(-1, 1)
            data['avg_reward'] = reward_list
            data['algorithm'] = algo
            data['seed'] = seed+i
            all = pd.concat([all, data], 0)
    sns.lineplot(x='step', y='avg_reward', data=all, hue='algorithm')
    plt.show()
    return all

def main():
    global env_name
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='AirRaidNoFrameskip-v4')
    parser.add_argument('--num_env', help='number of environments', type=int, default=5)
    parser.add_argument('--seed', help='RNG seed', type=int, default=42)
    parser.add_argument('--single_seed', help='single seed or multi-seeds?', type=bool, default=False)
    parser.add_argument('--num_timesteps', type=int, default=int(10e6))
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn_svib'], default='cnn_svib')
    parser.add_argument('--train_option', help='which algorithm do we train',
                        choices=['compare_with_regular', 'compare_with_none', 'uniform', 'gaussian', 'regular_noise_uniform', 'regular_gaussian', 'regular'],
                        default='regular')
    # parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear', 'double_linear_con'], default='double_linear_con')
    parser.add_argument('--log', help='logging type', choices=['tensorboard', 'stdout'], default='tensorboard')
    parser.add_argument('--great_time', help='the time gets great result:%Y%m%d%H%M', default=202008181728)#default='201904181134')
    parser.add_argument('--great_th', help='the timeth gets great result', default=73)
    parser.add_argument('--ib_alpha', type=float, default=0.8)
    args = parser.parse_args()
    config_log(args)
    env_name = args.env
    if args.train_option == 'compare_with_regular':
        algos = {'regular': args.ib_alpha, 'use_svib_uniform': args.ib_alpha, 'regular_with_noise':args.ib_alpha, 'use_svib_gaussian': args.ib_alpha, }
        all = train_all(algos=algos, env_id=args.env, num_timesteps=args.num_timesteps, seed=args.seed,
                        policy=args.policy, num_env=args.num_env)
    elif args.train_option == 'compare_with_none':
        algos = {'use_svib_gaussian': args.ib_alpha, 'use_svib_uniform': args.ib_alpha}
        all = train_all(algos=algos, env_id=args.env, num_timesteps=args.num_timesteps, seed=args.seed,
                        policy=args.policy, num_env=args.num_env)
    elif args.train_option == 'uniform':
        if not args.single_seed:
            algos = {'use_svib_uniform': args.ib_alpha}
            all = train_all(algos=algos, env_id=args.env, num_timesteps=args.num_timesteps, seed=args.seed,
                            policy=args.policy, num_env=args.num_env)
        else:
            all = train(env_id=args.env, num_env=args.num_env, num_timesteps=args.num_timesteps, seed=args.seed,
                        policy=args.policy, algo='use_svib_uniform', ib_alpha=args.ib_alpha)
    elif args.train_option == 'gaussian':
        if not args.single_seed:
            algos = {'use_svib_gaussian': args.ib_alpha}
            all = train_all(algos=algos, env_id=args.env, num_timesteps=args.num_timesteps, seed=args.seed,
                            policy=args.policy, num_env=args.num_env)
        else:
            all = train(env_id=args.env, num_env=args.num_env, num_timesteps=args.num_timesteps, seed=args.seed,
                        policy=args.policy, algo='use_svib_gaussian', ib_alpha=args.ib_alpha)
    elif args.train_option == 'regular_noise_uniform':
        algos = {'regular_with_noise': args.ib_alpha, 'use_svib_uniform': args.ib_alpha}
        all = train_all(algos=algos, env_id=args.env, num_timesteps=args.num_timesteps, seed=args.seed,
                        policy=args.policy, num_env=args.num_env)
    elif args.train_option == 'regular':
        algos = {'regular': args.ib_alpha}
        all = train_all(algos=algos, env_id=args.env, num_timesteps=args.num_timesteps, seed=args.seed,
                        policy=args.policy, num_env=args.num_env)
    else:
        algos = {'regular': args.ib_alpha, 'use_svib_gaussian': args.ib_alpha}
        all = train_all(algos=algos, env_id=args.env, num_timesteps=args.num_timesteps, seed=args.seed,
                        policy=args.policy, num_env=args.num_env)
    return all

if __name__ == '__main__':
    all = main()
