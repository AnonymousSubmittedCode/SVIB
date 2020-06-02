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
from runner import Runner
from baselines.a2c.utils import mse, cat_entropy, find_trainable_variables, Scheduler, make_path
from baselines.common import set_global_seeds, tf_util
from baselines.common.cmd_util import make_atari_env
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import grad_clip, explained_variance
from policy import CnnPolicy
ds = tf.contrib.distributions
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
env_name = "MoveToBeaconNoFrameskip-v1"
# gpu_options = tf.GPUOptions(allow_growth=True)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, master_ts = 1, worker_ts = 30,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4, cell = 256,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear',
            algo='regular', beta=1e-3):

        print('Create Session')
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        nact = ac_space.n
        nbatch = nenvs*master_ts*worker_ts

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, 1, cell = cell, model='step_model', algo=algo)
        train_model = policy(sess, ob_space, ac_space, nbatch, master_ts, worker_ts, model='train_model', algo=algo)
        print('model_setting_done')

        #loss construction
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.wpi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.wvf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.wpi))
        pg_loss = pg_loss - entropy * ent_coef
        print('algo: ', algo, 'max_grad_norm: ', str(max_grad_norm))
        try:
            if algo == 'regular':
                loss = pg_loss + vf_coef * vf_loss
            elif algo == 'VIB':
                '''
                implement VIB here, apart from the vf_loss and pg_loss, there should be a third loss,
                the kl_loss = ds.kl_divergence(model.encoding, prior), where prior is a Gaussian distribution with mu=0, std=1
                the final loss should be pg_loss + vf_coef * vf_loss + beta*kl_loss
                '''
                prior = ds.Normal(0.0, 1.0)
                kl_loss = tf.reduce_mean(ds.kl_divergence(train_model.encoding, prior))
                loss = pg_loss + vf_coef * vf_loss + beta*kl_loss
                # pass
            else:
                raise Exception('Algorithm not exists')
        except Exception as e:
            print(e)

        grads, global_norm = grad_clip(loss, max_grad_norm, ['model'])
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(wobs, whs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(whs)):
                cur_lr = lr.value()

            td_map = {train_model.wX:wobs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.wS] = states
                td_map[train_model.wM] = masks

            '''
            you can add and run additional loss for VIB here for debugging, such as kl_loss
            '''
            tloss, value_loss, policy_loss, policy_entropy, _ = sess.run(
                [loss, vf_loss, pg_loss, entropy, _train],
                feed_dict=td_map
            )
            return tloss, value_loss, policy_loss, policy_entropy

        params = find_trainable_variables("model")
        def save(save_path):
            ps = sess.run(params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.wvalue
        self.get_wh = step_model.get_wh
        self.initial_state = step_model.w_initial_state
        self.train = train
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

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
                actions, values, self.states, _= self.model.step(self.obs, self.states, self.dones)
                self.obs, rewards, self.dones, _ = self.env.step(actions)
                self.done = self.dones[0]
                self.one_episode_r += rewards
            self.total_r += self.one_episode_r
            self.one_episode_r = 0.
            self.done = False
        return (self.total_r[0] / float(self.num_episodes))

def learn(policy, env, test_env, seed, master_ts = 1, worker_ts = 8, cell = 256,
          ent_coef = 0.01, vf_coef = 0.5, max_grad_norm = 0.5, lr = 7e-4,
          alpha = 0.99, epsilon = 1e-5, total_timesteps = int(80e6), lrschedule = 'linear',
          log_interval = 10, gamma = 0.99, load_path="saved_nets-data/hrl_a2c/%s/data"%start_time,
          algo='regular', beta=1e-3):

    tf.reset_default_graph()
    set_global_seeds(seed)
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    print(str(nenvs)+"------------"+str(ob_space)+"-----------"+str(ac_space))
    max_grad_norm_tune = max_grad_norm
    max_grad_norm_tune_env_list = ['BreakoutNoFrameskip-v4', 'MsPacmanNoFrameskip-v4']
    global  env_name
    if env_name in max_grad_norm_tune_env_list:
        print('tune max grad norm')
        max_grad_norm_tune = 1.0
    model = Model(policy = policy, ob_space = ob_space, ac_space = ac_space, nenvs = nenvs, master_ts=master_ts, worker_ts=worker_ts,
                  ent_coef = ent_coef, vf_coef = vf_coef, max_grad_norm = max_grad_norm_tune, lr = lr, cell = cell,
                  alpha = alpha, epsilon = epsilon, total_timesteps = total_timesteps, lrschedule = lrschedule,
                  algo=algo, beta=beta)
    try:
        model.load(load_path)
    except Exception as e:
        print("no data to load!!"+str(e))
    runner = Runner(env = env, model = model, nsteps=master_ts*worker_ts, gamma=gamma)
    test_runner = test_Runner(env = test_env, model = model)

    tf.get_default_graph().finalize()
    nbatch = nenvs * master_ts * worker_ts
    tstart = time.time()
    reward_list = []
    prev_r = 0.
    exp_coef = .1
    for update in range(1, total_timesteps//nbatch+1):
        b_obs, b_whs, states, b_rewards, b_wmasks, b_actions, b_values = runner.run()
        tloss, value_loss, policy_loss, policy_entropy = model.train(b_obs, b_whs, states, b_rewards, b_wmasks, b_actions, b_values)
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(b_values, b_rewards)
            logger.record_tabular("fps", fps)
            logger.record_tabular("tloss", float(tloss))
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("policy_loss", float(policy_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()
            if update % (200*log_interval) == 0:
                save_th = update//(200*log_interval)
                model.save("saved_nets-data/%s/hrl_a2c/%s/%s/data" % (env_name, start_time, save_th))
                episode_r = exp_coef*(test_runner.run()) + (1-exp_coef)*prev_r
                prev_r = np.copy(episode_r)
                reward_list.append(episode_r)
                logger.record_tabular('episode_r', float(episode_r))
                logger.dump_tabular()
    env.close()
    return reward_list

def train(env_id, num_timesteps, seed, policy, lrschedule, num_env,load_path,
          algo='regular', beta=1e-3):
    if policy == 'cnn_svib':
        policy_fn = CnnPolicy
    else:
        policy_fn = CnnPolicy
    env = VecFrameStack(make_atari_env(env_id, num_env, seed), 4)
    test_env = VecFrameStack(make_atari_env(env_id, num_env, seed+1), 4)
    reward_list = learn(policy_fn, env, test_env, seed, total_timesteps=int(num_timesteps),
                        lrschedule=lrschedule, load_path=load_path, algo=algo, beta=beta)
    # env.close()
    return reward_list

def config_log(FLAGS):
    logdir = "tensorboard/%s/hrl_a2c_svib/%s_lr%s_%s/%s_%s_%s" % (
        FLAGS.env,FLAGS.num_timesteps, '0.0007',FLAGS.policy, start_time, FLAGS.train_option, str(FLAGS.beta))
    if FLAGS.log == "tensorboard":
        Logger.DEFAULT = Logger.CURRENT = Logger(dir=logdir, output_formats=[TensorBoardOutputFormat(logdir)])
    elif FLAGS.log == "stdout":
        Logger.DEFAULT = Logger.CURRENT = Logger(dir=logdir, output_formats=[HumanOutputFormat(sys.stdout)])

def train_all(algos, env_id, num_timesteps, seed, policy, lrschedule, num_env, load_path):
    all = pd.DataFrame([])
    for i in range(3):
        for algo in algos.keys():
            beta = algos[algo]
            reward_list = np.reshape(np.array(train(
                env_id, num_timesteps=num_timesteps, seed=seed+i, policy=policy, lrschedule=lrschedule,
                num_env=num_env, load_path=load_path, algo=algo, beta=beta)), (-1, 1))
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
    parser.add_argument('--env', help='environment ID', default='AsteroidsNoFrameskip-v4')
    parser.add_argument('--num_env', help='number of environments', type=int, default=5)
    parser.add_argument('--seed', help='RNG seed', type=int, default=42)
    parser.add_argument('--num_timesteps', type=int, default=int(14e6))
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn_svib', 'lstm'], default='cnn_svib')
    parser.add_argument('--train_option', help='which algorithm do we train',
                        choices=['regular', 'compare_with_regular', 'VIB'],
                        default='VIB')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear', 'double_linear_con'], default='double_linear_con')
    parser.add_argument('--log', help='logging type', choices=['tensorboard', 'st dout'], default='tensorboard')
    parser.add_argument('--great_time', help='the time gets great result:%Y%m%d%H%M', default=202008291101)#default='201904181134')
    parser.add_argument('--great_th', help='the timeth gets great result', default=45)
    parser.add_argument('--beta', type=float, default=1e-3)
    args = parser.parse_args()
    config_log(args)
    env_name = args.env
    load_path = "saved_nets-data/%s/hrl_a2c_svib/%s/%s/data" % (args.env, args.great_time,args.great_th)
    all = 0
    try:
        if args.train_option == 'regular':
            algos = {'regular': args.beta}
            all = train_all(algos=algos, env_id=args.env, num_timesteps=args.num_timesteps, seed=args.seed,
                            policy=args.policy, lrschedule=args.lrschedule, num_env=args.num_env, load_path=load_path)
        elif args.train_option == 'compare_with_regular':
            algos = {'regular': args.beta, 'VIB': args.beta}
            all = train_all(algos=algos, env_id=args.env, num_timesteps=args.num_timesteps, seed=args.seed,
                            policy=args.policy, lrschedule=args.lrschedule, num_env=args.num_env, load_path=load_path,)
        elif args.train_option == 'VIB':
            algos = {'VIB': args.beta}
            all = train_all(algos=algos, env_id=args.env, num_timesteps=args.num_timesteps, seed=args.seed,
                            policy=args.policy, lrschedule=args.lrschedule, num_env=args.num_env, load_path=load_path,)
        else:
            raise Exception('No such option')
    except Exception as e:
        print(e)
    return all

if __name__ == "__main__":
    print('start running...')
    all=main()