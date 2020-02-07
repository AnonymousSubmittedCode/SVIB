import os
import os.path as osp
import gym
import time
import joblib
import logging
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common import tf_util

from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.utils import cat_entropy

# from obs_buffer import Memory

from utils import np_compute_cosine, np_l2norm1, compute_ir

class Runner_hrl(object):
    def __init__(self, env, model, master_ts = 5, worker_ts = 10, gamma=0.99, er_coef = .4, use_vae = False):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        self.nenv = env.num_envs
        self.master_ts = master_ts
        self.worker_ts = worker_ts
        self.nsteps = master_ts * worker_ts
        self.m_batch_ob_shape = (self.nenv * self.master_ts, nh, nw, nc)
        self.w_batch_ob_shape = (self.nenv * self.nsteps, nh, nw, nc)
        self.w_batch_goal_shape = (self.nenv * self.nsteps, self.model.cell)
        self.nc = nc
        obs = env.reset()
        h = self.model.get_wh(obs)
        self.mobs = np.copy(obs)
        self.mobs_ = np.copy(obs)
        self.wobs = np.copy(obs)
        self.mh = np.copy(h)
        self.mh_ = np.copy(h)
        self.wh = np.copy(h)
        self.gamma = gamma
        self.er_coef = er_coef
        self.use_vae = use_vae
        self.mstates = model.m_initial_state
        self.wstates = model.w_initial_state
        '''
        whether the previous step is done
        '''
        self.dones = [True for _ in range(self.nenv)]
        '''
        after a goal is setting, whether the goal is done
        '''
        self.mdones = [True for _ in range(self.nenv)]#if an env is firstly done during a worker-ts rollout, then mdones of that env is True
        '''
        whether the worker's job is done
        '''
        self.wdones = [True for _ in range(self.nenv)]

    def run(self):
        m_mb_obs, m_mb_obs_ ,w_mb_obs = [], [], []
        m_mb_rewards, m_mb_actions, m_mb_values, m_mb_dones, m_mb_mhs, m_mb_mh_s = [], [], [], [], [], []
        w_mb_rewards, w_mb_actions, w_mb_values, w_mb_dones, w_mb_goals, w_mb_whs = [], [], [], [], [], []
        w_mb_last_values = []
        b_dones = []
        m_mb_states = self.mstates
        w_mb_states = self.wstates#actually this ought to be zeros tensor, thanks to signal wdones, we don't have to specify it
        goals = np.zeros(shape=(self.nenv, self.model.cell), dtype=np.float32)
        m_rewards = np.zeros(shape=(self.nenv), dtype=np.float32)
        for n in range(self.nsteps):
            if n % self.worker_ts == 0:
                '''
                once last step is done, a new goal should be set
                '''
                if self.use_vae:
                    goals, mvalues, mstates, actions, wvalues, wstates, _ =\
                        self.model.step(mhs = self.mh, mstate = self.mstates, mmask = self.mdones,
                                        whs = self.wh, wstate = self.wstates, wmask = self.wdones,
                                        origin_goal = goals, goal_mask = [True for _ in range(self.nenv)])
                else:
                    goals, mvalues, mstates, actions, wvalues, wstates, _ =\
                        self.model.step(mobs = self.mobs, mstate = self.mstates, mmask = self.mdones,
                                        wobs = self.wobs, wstate = self.wstates, wmask = self.wdones,
                                        origin_goal = goals, goal_mask = [True for _ in range(self.nenv)])
                m_mb_obs.append(np.copy(self.mobs))
                w_mb_obs.append(np.copy(self.wobs))
                m_mb_actions.append(goals)
                w_mb_actions.append(actions)
                m_mb_values.append(mvalues)
                w_mb_values.append(wvalues)
                m_mb_dones.append(self.mdones)
                w_mb_dones.append(self.wdones)
                m_mb_mhs.append(self.mh)
                w_mb_whs.append(self.wh)
                b_dones.append(self.dones)
                w_mb_goals.append(goals)

                obs, ers, dones, _ = self.env.step(actions)
                h = self.model.get_wh(obs)
                self.mdones = [False for _ in range(self.nenv)]#reset mdones
                m_rewards += (1. - np.asarray(self.mdones, dtype=np.bool)) * ers
                irs = np.mean((np.square(h - goals)) / 2, axis=1) + self.er_coef*ers
                w_mb_rewards.append(irs)

                show_img(obs[0], scope='obs')

                self.mstates = mstates
                self.wstates = wstates
                for i, done in enumerate(dones):
                    if done:
                        if self.mdones[i] is False:
                            self.mdones[i] = True
                            self.mobs_[i] = obs[i]
                            self.mh_[i] = h[i]
                self.wdones = dones
                self.dones = dones
                self.wobs = obs
                self.wh = h
            else:
                if self.use_vae:
                    goals, mvalues, mstates, actions, wvalues, wstates, __ =\
                        self.model.step(mhs=self.wh, mstate=self.mstates, mmask=self.mdones,
                                        whs=self.wh, wstate=self.wstates, wmask=self.wdones,
                                        origin_goal=goals, goal_mask=self.dones)
                else :
                    goals, mvalues, mstates, actions, wvalues, wstates, __ =\
                        self.model.step(mobs = self.wobs, mstate =self.mstates, mmask = self.mdones,
                                        wobs = self.wobs, wstate = self.wstates, wmask = self.wdones,
                                        origin_goal = goals, goal_mask = self.dones)
                w_mb_obs.append(np.copy(self.wobs))
                w_mb_actions.append(actions)
                w_mb_values.append(wvalues)
                w_mb_dones.append(self.wdones)
                b_dones.append(self.dones)
                w_mb_goals.append(goals)
                w_mb_whs.append(self.wh)

                obs, ers, dones, _ = self.env.step(actions)
                h = self.model.get_wh(obs)
                m_rewards += (1. - np.asarray(self.mdones, dtype=np.bool)) * ers
                irs = np.mean((np.square(h - goals)) / 2, axis=1) + self.er_coef*ers
                w_mb_rewards.append(irs)

                show_img(obs[0], scope='obs')

                self.wstates = wstates
                for i, done in enumerate(dones):
                    '''
                    if there exists one done that is True during the whole worker_ts-step process, then master's job is done
                    '''
                    if done:
                        if self.mdones[i] is False:
                            self.mdones[i] = True
                            self.mh_[i] = h[i]
                            self.mobs_[i] = obs[i]
                self.wdones = dones
                self.dones = dones
                self.wobs = obs
                self.wh = h
                if n % self.worker_ts == self.worker_ts - 1:
                    '''
                    if so, then all workers' work are done
                    '''
                    m_mb_rewards.append(m_rewards)
                    m_rewards = np.zeros((self.nenv), dtype = np.float32)
                    self.wdones = [True for _ in range(self.nenv)]
                    self.mobs = np.copy(self.wobs)
                    self.mh = np.copy(self.wh)
                    self.mh_ = np.expand_dims(np.asarray(self.mdones, dtype=np.float32), axis=1)*self.mh_ +\
                               np.expand_dims(1.-np.asarray(self.mdones, dtype = np.float32), axis=1)*self.wh
                    m_mb_mh_s.append(self.mh_)
                    self.mobs_ = np.asarray(self.mdones, dtype = np.float32).reshape(self.nenv, 1,1,1) * self.mobs_ + \
                                 (1. - np.asarray(self.mdones, dtype=np.float32).reshape(self.nenv, 1,1,1)) * self.wobs
                    m_mb_obs_.append(self.mobs_)
                    if self.use_vae:
                        w_last_values = self.model.wvalue(self.wh, goals, self.wstates, self.dones).tolist()
                    else:
                        w_last_values = self.model.wvalue(self.wobs, goals, self.wstates, self.dones).tolist()
                    w_mb_last_values.append(w_last_values)

        m_mb_dones.append(self.wdones)
        w_mb_dones.append(self.mdones)
        b_dones.append(self.dones)
        #batch of steps to batch of rollouts
        m_mb_obs = np.asarray(m_mb_obs, dtype = np.float32).swapaxes(1, 0).reshape(self.m_batch_ob_shape)
        w_mb_obs = np.asarray(w_mb_obs, dtype = np.float32).swapaxes(1, 0).reshape(self.w_batch_ob_shape)
        m_mb_rewards = np.asarray(m_mb_rewards, dtype = np.float32).swapaxes(1, 0)
        w_mb_rewards = np.asarray(w_mb_rewards, dtype = np.float32).swapaxes(1, 0).reshape(self.nenv * self.master_ts, self.worker_ts)
        m_mb_actions = np.asarray(m_mb_actions, dtype = np.float32).swapaxes(1, 0)
        w_mb_actions = np.asarray(w_mb_actions, dtype = np.int32).swapaxes(1, 0).reshape(self.nenv * self.master_ts, self.worker_ts)
        m_mb_values = np.asarray(m_mb_values, dtype = np.float32).swapaxes(1, 0)
        w_mb_values = np.asarray(w_mb_values, dtype = np.float32).swapaxes(1, 0).reshape(self.nenv * self.master_ts, self.worker_ts)
        m_mb_dones = np.asarray(m_mb_dones, dtype = np.bool).swapaxes(1, 0)#shape = [nenv, master_ts + 1]
        m_mb_masks = m_mb_dones[:, :-1]
        m_mb_dones = m_mb_dones[:, 1:]
        w_mb_dones = np.asarray(w_mb_dones, dtype = np.bool).swapaxes(1, 0)
        w_mb_masks = w_mb_dones[:, :-1]
        w_mb_dones = w_mb_dones[:, 1:]
        b_dones = np.asarray(b_dones, dtype = np.bool).swapaxes(1, 0)#shape = [nenv, master_ts + 1]
        b_masks = b_dones[:, :-1].reshape(self.nenv * self.master_ts, self.worker_ts)
        b_dones = b_dones[:, 1:].reshape(self.nenv * self.master_ts, self.worker_ts)
        w_mb_goals = np.asarray(w_mb_goals, dtype=np.float32).swapaxes(1, 0).reshape((self.w_batch_goal_shape))
        m_mb_mhs = np.asarray(m_mb_mhs, dtype=np.float32).swapaxes(1, 0)
        m_mb_mh_s = np.asarray(m_mb_mh_s, dtype=np.float32).swapaxes(1, 0)
        w_mb_whs = np.asarray(w_mb_whs, dtype=np.float32).swapaxes(1, 0)
        if self.use_vae:
            m_last_values = self.model.mvalue(self.mh_, self.mstates, self.mdones).tolist()
        else:
            m_last_values = self.model.mvalue(self.mobs_, self.mstates, self.mdones).tolist()
        w_mb_last_values = np.asarray(w_mb_last_values, dtype=np.float32).swapaxes(1, 0)#shape = [nenv, master_ts]
        #discount/bootstrap off value fn
        #mrewards
        for n, (rewards, dones,  value) in enumerate(zip(m_mb_rewards, m_mb_dones, m_last_values)):
            rewards = rewards.tolist()#shape = [master_ts]
            dones = dones.tolist()#shape = [master_ts]
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            m_mb_rewards[n] = rewards
        m_mb_rewards = m_mb_rewards.flatten()
        for n, (rewards, dones, value) in enumerate(zip(w_mb_rewards, b_dones, w_mb_last_values)):
            rewards_list = []
            for i in range(self.master_ts):
                rewards_slice = np.copy(rewards[i*self.worker_ts:(i+1)*self.worker_ts]).tolist()
                dones_slices = np.copy(dones[i*self.worker_ts:(i+1)*self.worker_ts]).tolist()
                if dones_slices[-1] == 0:
                    rewards_slice = discount_with_dones(rewards_slice+[value[i]], dones_slices+[0], self.gamma)[:-1]
                else:
                    rewards_slice = discount_with_dones(rewards_slice, dones_slices, self.gamma)[:-1]
                rewards_list += rewards_slice
            w_mb_rewards[n] = rewards_list
            # rewards = rewards.tolist()
            # dones = dones.tolist()
            # if dones[-1] == 0:
            #     rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            # else:
            #     rewards = discount_with_dones(rewards, dones, self.gamma)
        w_mb_rewards = w_mb_rewards.flatten()
        m_mb_actions = m_mb_actions.reshape(m_mb_actions.shape[0] * m_mb_actions.shape[1], -1)
        w_mb_actions = w_mb_actions.flatten()
        m_mb_values = m_mb_values.flatten()
        w_mb_values = w_mb_values.flatten()
        m_mb_masks = m_mb_masks.flatten()
        w_mb_masks = w_mb_masks.flatten()
        m_mb_mhs = m_mb_mhs.reshape(m_mb_mhs.shape[0]*m_mb_mhs.shape[1], -1)
        m_mb_mh_s = m_mb_mh_s.reshape(m_mb_mh_s.shape[0]*m_mb_mh_s.shape[1], -1)
        w_mb_whs = w_mb_whs.reshape(w_mb_whs.shape[0]*w_mb_whs.shape[1], -1)
        return m_mb_obs, m_mb_states, m_mb_rewards, m_mb_masks, m_mb_actions, m_mb_values, m_mb_mhs, m_mb_mh_s,\
               w_mb_obs, w_mb_states, w_mb_rewards, w_mb_masks, w_mb_actions, w_mb_values, w_mb_goals, w_mb_whs

class Runner_prac(object):

    def __init__(self, env, model, nsteps=8, gamma=0.99, subgoal_rollout = 1000):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.nenv = nenv
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc)
        self.nc = nc
        obs = env.reset()
        h = model.get_wh(obs)
        self.obs = np.copy(obs)
        self.whs = np.copy(h)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.wdones = [True for _ in range(nenv)]
        self.dones = [True for _ in range(nenv)]
        self.subgoal_rollout = subgoal_rollout
        self.subgoal_current_rollout = 0
        self.goals = None
        self.decoder = None
        # self.defualt_dones = [True for _ in range(nenv)]

    def run(self):
        b_obs, b_whs, b_rewards, b_actions, b_values, b_dones, b_wdones, b_goals = [],[],[],[],[],[],[],[]
        states = self.states
        if self.subgoal_current_rollout % self.subgoal_rollout == 0:
            self.goals = self.model.goal_generator()
            self.decoder = self.model.goal_decoder(self.goals)
        for n in range(self.nsteps):
            actions, values, states, _ = self.model.practice(self.whs, self.states, self.dones, self.goals)
            b_obs.append(np.copy(self.obs))
            b_whs.append(np.copy(self.whs))
            b_actions.append(actions)
            b_values.append(values)
            b_dones.append(self.dones)
            b_wdones.append(self.wdones)
            b_goals.append(self.goals)
            obs, rewards, dones, _ = self.env.step(actions)
            h = self.model.get_wh(obs)
            show_img(self.obs[0], scope='current_obs')
            show_img(self.decoder[0]/200., scope='current_goals')
            irs = compute_ir(h, self.goals)
            b_rewards.append(irs)
            self.states = states
            self.dones = dones
            self.wdones = dones
            self.obs = obs
            self.whs = h
        if self.subgoal_current_rollout%self.subgoal_rollout == self.subgoal_rollout - 1:
            self.wdones = [True for _ in range(self.nenv)]
        else:
            self.wdones = self.dones
        self.subgoal_current_rollout = (self.subgoal_current_rollout + 1) % self.subgoal_rollout
        b_dones.append(self.dones)
        b_wdones.append(self.wdones)
        #batch of steps to batch of rollouts
        b_obs = np.asarray(b_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        b_whs = np.asarray(b_whs, dtype=np.float32).swapaxes(1, 0)
        b_whs = b_whs.reshape(b_whs.shape[0]*b_whs.shape[1], -1)
        b_rewards = np.asarray(b_rewards, dtype=np.float32).swapaxes(1, 0)
        b_actions = np.asarray(b_actions, dtype=np.int32).swapaxes(1, 0)
        b_values = np.asarray(b_values, dtype=np.float32).swapaxes(1, 0)
        b_dones = np.asarray(b_dones, dtype=np.bool).swapaxes(1, 0)
        b_wdones = np.asarray(b_wdones, dtype=np.bool).swapaxes(1, 0)
        b_masks = b_dones[:, :-1]
        b_dones = b_dones[:, 1:]
        b_wmasks = b_wdones[:, :-1]
        b_wdones = b_wdones[:, 1:]
        b_goals = np.asarray(b_goals, dtype=np.float32).swapaxes(1, 0)
        b_goals = b_goals.reshape(b_goals.shape[0]*b_goals.shape[1], -1)
        last_values = self.model.value(self.whs, self.states, self.dones, self.goals).tolist()
        raw_ir = np.sum(b_rewards[0])
        for n, (rewards, dones, value) in enumerate(zip(b_rewards, b_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            # raw_ir += discount_with_dones(rewards, dones, self.gamma)[0]
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            b_rewards[n] = rewards
        # raw_ir = raw_ir / float(self.nenv)
        b_rewards = b_rewards.flatten()
        b_actions = b_actions.flatten()
        b_values = b_values.flatten()
        b_wmasks = b_wmasks.flatten()
        return b_whs, states, b_rewards, b_wmasks, b_actions, b_values, b_goals, raw_ir

class Runner(object):
    def __init__(self, env, model, nsteps=8, gamma=0.99):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.nenv = nenv
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc)
        self.nc = nc
        obs = env.reset()
        h = model.get_wh(obs)
        self.obs = np.copy(obs)
        self.whs = np.copy(h)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [True for _ in range(nenv)]
        self.episode_r = 0.

    def run(self):
        b_obs, b_whs, b_rewards, b_actions, b_values, b_dones = [],[],[],[],[],[]
        b_states = self.states
        for n in range(self.nsteps):
            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)
            b_obs.append(np.copy(self.obs))
            b_whs.append(np.copy(self.whs))
            b_actions.append(actions)
            b_values.append(values)
            b_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            h = self.model.get_wh(obs)
            b_rewards.append(rewards)
            self.states = states
            self.dones = dones
            self.obs = obs
            self.whs = h
        b_dones.append(self.dones)
        #batch of steps to batch of rollouts
        b_obs = np.asarray(b_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        b_whs = np.asarray(b_whs, dtype=np.float32).swapaxes(1, 0)
        b_whs = b_whs.reshape(b_whs.shape[0]*b_whs.shape[1], -1)
        b_rewards = np.asarray(b_rewards, dtype=np.float32).swapaxes(1, 0)
        b_actions = np.asarray(b_actions, dtype=np.int32).swapaxes(1, 0)
        b_values = np.asarray(b_values, dtype=np.float32).swapaxes(1, 0)
        b_dones = np.asarray(b_dones, dtype=np.bool).swapaxes(1, 0)
        b_masks = b_dones[:, :-1]
        b_dones = b_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(b_rewards, b_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            b_rewards[n] = rewards
        b_rewards = b_rewards.flatten()
        b_actions = b_actions.flatten()
        b_values = b_values.flatten()
        b_masks = b_masks.flatten()
        return b_obs, b_whs, b_states, b_rewards, b_masks, b_actions, b_values

class Runner_baseline(object):

    def __init__(self, env, model, nsteps=5, gamma=0.99):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        # nh, nw, nc = 84, 84, nc // 3
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc)
        self.obs = np.zeros((nenv, nh, nw, nc), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            # print((obs[0, :, :, 0] == obs[0, :, :, 1]).all())
            # obs = obs_preprocess(obs)
            show_img(obs[0], 'obs')
            self.states = states
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        raw_rewards = np.sum(mb_rewards[0])
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, raw_rewards

class Runner_explore(object):

    def __init__(self, env, model, nsteps, phase = 'train_vae'):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc)
        # _, eh, ew, ec = model.encoder.get_shape()
        # self.mean_shape = (eh.value, ew.value, ec.value)
        # self.var_dataset_init_shape = (0, eh.value, ew.value, ec.value)
        self.nc = nc
        obs = env.reset()
        self.obs = np.copy(obs)
        self.nsteps = nsteps
        self.phase = phase
        # self.mean = np.zeros(self.mean_shape, dtype=np.float32)
        # self.var = np.zeros(self.mean_shape, dtype=np.float32)
        # self.var_dataset = np.zeros(self.var_dataset_init_shape, dtype=np.float32)

    def change_phase(self, phase):
        self.phase = phase

    def run(self):
        # if rollnum == 1:
        #     self.mean = np.zeros(self.mean_shape, dtype=np.float32)
        #     self.var = np.zeros(self.mean_shape, dtype=np.float32)
        #     self.var_dataset = np.zeros(self.var_dataset_init_shape, dtype=np.float32)
        wb_obs = []
        for n in range(self.nsteps):
            actions, decoder = self.model.exploration(self.obs, phase=self.phase)
            # print(np.max(self.obs[0]), np.min(self.obs[0]))
            # print(np.max(wh_decoder[0]), np.min(wh_decoder[0]))
            show_img(self.obs[0], scope='obs')
            show_img(decoder[0] / 200., scope='decoder')

            # k = (rollnum - 1) * self.nsteps + n + 1
            # self.mean = (k-1.)/k * self.mean + 1./k * np.mean(wh_encoder, axis=0)
            # #print(self.mean.shape)
            # if self.var_dataset.shape[0] <= self.data_size:
            #     if np.random.rand() <= 0.3:
            #         self.var_dataset= np.concatenate((self.var_dataset, wh_encoder), axis=0)

            wb_obs.append(np.copy(self.obs))
            obs, rewards, dones, _ = self.env.step(actions)
            self.obs = obs

        # if self.var_dataset.shape[0] > 0:
        #     self.var = np.mean((self.var_dataset - self.mean) ** 2, axis=0)

        #batch of steps to batch of rollouts
        wb_obs = np.asarray(wb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        return wb_obs

class Runner_hrl_tpg(object):

    def __init__(self, env, model, master_ts = 3, worker_ts = 20, gamma=0.99, ir_coef = 5e-2, er_coef = 20e-2, use_vae = False):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        self.nenv = env.num_envs
        self.master_ts = master_ts
        self.worker_ts = worker_ts
        self.nsteps = master_ts * worker_ts
        self.m_batch_ob_shape = (self.nenv * self.master_ts, nh, nw, nc)
        self.w_batch_ob_shape = (self.nenv * self.nsteps, nh, nw, nc)
        self.w_batch_goal_shape = (self.nenv * self.nsteps, self.model.cell)
        # self.obs = np.zeros((nenv, nh, nw, nc), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        h = self.model.get_wh(obs)
        self.mobs = np.copy(obs)
        self.mobs_ = np.copy(obs)
        self.wobs = np.copy(obs)
        self.mh = np.copy(h)
        self.mh_ = np.copy(h)
        self.wh = np.copy(h)
        self.gamma = gamma
        self.ir_coef = ir_coef
        self.er_coef = er_coef
        self.use_vae = use_vae
        self.mstates = model.m_initial_state
        self.wstates = model.w_initial_state
        '''
        whether last step is done
        '''
        self.dones = [True for _ in range(self.nenv)]
        '''
        after a goal is setting, whether the goal is done
        '''
        self.mdones = [False for _ in range(self.nenv)]#if an env is firstly done during a worker-ts rollout, then mdones of that env is True
        '''
        whether the worker's job is done
        '''
        self.wdones = [True for _ in range(self.nenv)]

    def run(self):
        m_mb_obs, m_mb_obs_, w_mb_obs = [], [], []
        m_mb_rewards, m_mb_actions, m_mb_values, m_mb_dones, m_mb_mhs, m_mb_mh_s = [], [], [], [], [], []
        w_mb_rewards, w_mb_actions, w_mb_values, w_mb_dones, w_mb_goals, w_mb_whs = [], [], [], [], [], []
        b_dones = []
        goals = np.zeros(shape=(self.nenv, self.model.cell), dtype=np.float32)
        m_rewards = np.zeros(shape=(self.nenv), dtype=np.float32)
        for n in range(self.nsteps):
            if n % self.worker_ts == 0:
                # self.mh, self.wh = self.model.get_mh(self.mobs), self.model.get_wh(self.wobs)#embedding first
                '''
                once last step is done, a new goal should be set
                '''
                if self.use_vae:
                    goals, mvalues, mstates, actions, wvalues, wstates, __ = self.model.step(mhs = self.wh, mstate =self.mstates, mmask = self.mdones,
                                                                                            whs = self.wh, wstate = self.wstates, wmask = self.wdones,
                                                                                            origin_goal = goals, goal_mask = [True for _ in range(self.nenv)])
                else:
                    goals, mvalues, mstates, actions, wvalues, wstates, __ = self.model.step(mobs = self.wobs, mstate =self.mstates, mmask = self.mdones,
                                                                                            wobs = self.wobs, wstate = self.wstates, wmask = self.wdones,
                                                                                            origin_goal = goals, goal_mask = [True for _ in range(self.nenv)])
                m_mb_obs.append(np.copy(self.mobs))
                w_mb_obs.append(np.copy(self.wobs))
                m_mb_actions.append(goals)
                w_mb_actions.append(actions)
                m_mb_values.append(mvalues)
                w_mb_values.append(wvalues)
                m_mb_dones.append(self.mdones)
                w_mb_dones.append(self.wdones)
                m_mb_mhs.append(self.mh)
                w_mb_whs.append(self.wh)
                b_dones.append(self.dones)
                w_mb_goals.append(goals)

                obs, ers, dones, _ = self.env.step(actions)
                h = self.model.get_wh(obs)
                irs = 2. * np_compute_cosine(h, goals) + self.er_coef*ers
                w_mb_rewards.append(irs)
                ers += self.ir_coef * dones * (2. - 2. * np_compute_cosine(h, goals))#if done is true, next step will be a new goal
                self.mdones = [False for _ in range(self.nenv)]
                m_rewards += 2*(1.-np.asarray(self.mdones, dtype=np.bool)) * ers

                show_img(obs[0], scope='obs')

                self.mstates = mstates
                self.wstates = wstates
                # self.one_shot = [False for _ in range(self.nenv)]
                for i, done in enumerate(dones):
                    if done:
                        if self.mdones[i] is False:
                            self.mdones[i] = True
                            self.mobs_ = obs[i]
                            self.mh_[i] = h[i]

                self.wdones = dones
                self.dones = dones
                self.wobs = obs
                self.wh = h
            else:
                if self.use_vae:
                    goals, mvalues, mstates, actions, wvalues, wstates, __ = self.model.step(mhs=self.wh, mstate=self.mstates, mmask=self.mdones,
                                                                                            whs=self.wh, wstate=self.wstates, wmask=self.wdones,
                                                                                            origin_goal=goals, goal_mask=self.dones)
                else :
                    goals, mvalues, mstates, actions, wvalues, wstates, __ = self.model.step(mobs = self.wobs, mstate =self.mstates, mmask = self.mdones,
                                                                                            wobs = self.wobs, wstate = self.wstates, wmask = self.wdones,
                                                                                            origin_goal = goals, goal_mask = [True for _ in range(self.nenv)])
                w_mb_obs.append(np.copy(self.wobs))
                w_mb_actions.append(actions)
                w_mb_values.append(wvalues)
                w_mb_dones.append(self.wdones)
                b_dones.append(self.dones)
                w_mb_goals.append(goals)
                w_mb_whs.append(self.wh)

                obs, ers, dones, _ = self.env.step(actions)
                h = self.model.get_wh(obs)
                irs = 2 * np_compute_cosine(h, goals) + self.er_coef*ers
                # irs = self.model.get_ir(h, goals) - self.model.get_ir(self.wh, goals) + self.er_coef*ers
                w_mb_rewards.append(irs)
                if n % self.worker_ts == self.worker_ts - 1:
                    ers += self.ir_coef * (2. - 2. * np_compute_cosine(h, goals))
                else:
                    '''
                    if done is True, next step will be a new goal.
                    '''
                    ers += self.ir_coef * dones * (2. - 2. * np_compute_cosine(h, goals))
                m_rewards += 2.*(1.-np.asarray(self.mdones, dtype=np.bool)) * ers
                show_img(obs[0], scope='obs')

                self.wstates = wstates
                self.wdones = dones
                for i, done in enumerate(dones):
                    '''
                    if there exists one done that is True during the whole worker_ts-step process, then master's job is done
                    '''
                    if done:
                        if self.mdones[i] is False:
                            self.mdones[i] = True
                            self.mh_[i] = h[i]
                            print(self.mobs_.shape)
                            self.mobs_[i] = obs[i]


                self.dones = dones
                self.wobs = obs
                self.wh = h
                if n % self.worker_ts == self.worker_ts - 1:
                    '''
                    if so, then all workers' work are done
                    '''
                    m_mb_rewards.append(m_rewards)
                    m_rewards = np.zeros((self.nenv), dtype = np.float32)
                    self.wdones = [True for _ in range(self.nenv)]
                    self.mobs = np.copy(self.wobs)
                    self.mh_ = np.expand_dims(np.asarray(self.mdones, dtype=np.float32), axis=1)*self.mh_ +\
                              np.expand_dims(1.-np.asarray(self.mdones, dtype = np.float32), axis=1)*self.wh
                    m_mb_mh_s.append(self.mh_)
                    self.mobs_ = np.asarray(self.mdones, dtype = np.float32).reshape(self.nenv, 1,1,1) * self.mobs_ + \
                                 (1. - np.asarray(self.mdones, dtype=np.float32).reshape(self.nenv, 1,1,1) * self.mobs_) * self.wobs
                    m_mb_obs_.append(self.mobs_)

        m_mb_dones.append(self.wdones)
        w_mb_dones.append(self.mdones)
        b_dones.append(self.dones)
        #batch of steps to batch of rollouts
        m_mb_obs = np.asarray(m_mb_obs, dtype = np.float32).swapaxes(1, 0).reshape(self.m_batch_ob_shape)
        w_mb_obs = np.asarray(w_mb_obs, dtype = np.float32).swapaxes(1, 0).reshape(self.w_batch_ob_shape)
        m_mb_rewards = np.asarray(m_mb_rewards, dtype = np.float32).swapaxes(1, 0)
        w_mb_rewards = np.asarray(w_mb_rewards, dtype = np.float32).swapaxes(1, 0).reshape(self.nenv * self.master_ts, self.worker_ts)
        m_mb_actions = np.asarray(m_mb_actions, dtype = np.float32).swapaxes(1, 0)
        w_mb_actions = np.asarray(w_mb_actions, dtype = np.int32).swapaxes(1, 0).reshape(self.nenv * self.master_ts, self.worker_ts)
        m_mb_values = np.asarray(m_mb_values, dtype = np.float32).swapaxes(1, 0)
        w_mb_values = np.asarray(w_mb_values, dtype = np.float32).swapaxes(1, 0).reshape(self.nenv * self.master_ts, self.worker_ts)
        m_mb_dones = np.asarray(m_mb_dones, dtype = np.bool).swapaxes(1, 0)#shape = [nenv, master_ts + 1]
        m_mb_masks = m_mb_dones[:, :-1]
        m_mb_dones = m_mb_dones[:, 1:]
        w_mb_dones = np.asarray(w_mb_dones, dtype = np.bool).swapaxes(1, 0)
        w_mb_masks = w_mb_dones[:, :-1]
        w_mb_dones = w_mb_dones[:, 1:]
        b_dones = np.asarray(b_dones, dtype = np.bool).swapaxes(1, 0)#shape = [nenv, master_ts + 1]
        b_masks = b_dones[:, :-1].reshape(self.nenv * self.master_ts, self.worker_ts)
        b_dones = b_dones[:, 1:].reshape(self.nenv * self.master_ts, self.worker_ts)
        w_mb_goals = np.asarray(w_mb_goals, dtype=np.float32).swapaxes(1, 0).reshape((self.w_batch_goal_shape))
        m_mb_mhs = np.asarray(m_mb_mhs, dtype=np.float32).swapaxes(1, 0)
        m_mb_mh_s = np.asarray(m_mb_mh_s, dtype=np.float32).swapaxes(1, 0)
        w_mb_whs = np.asarray(w_mb_whs, dtype=np.float32).swapaxes(1, 0)
        if self.use_vae:
            m_last_values = self.model.mvalue(self.mh_, self.mstates, self.mdones).tolist()
            w_last_values = self.model.wvalue(self.wh, self.wstates, self.dones, goals).tolist()
        else:
            m_last_values = self.model.mvalue(self.mobs_, self.mstates, self.mdones).tolist()
            w_last_values = self.model.wvalue(self.wobs, self.wstates, self.dones, goals).tolist()
        #discount/bootstrap off value fn
        #mrewards
        for n, (rewards, dones,  value) in enumerate(zip(m_mb_rewards, m_mb_dones, m_last_values)):
            rewards = rewards.tolist()#shape = [master_ts]
            dones = dones.tolist()#shape = [master_ts]
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            m_mb_rewards[n] = rewards
        m_mb_rewards = m_mb_rewards.flatten()
        for n, (rewards, dones, value) in enumerate(zip(w_mb_rewards, b_dones, w_last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            w_mb_rewards[n] = rewards
        w_mb_rewards = w_mb_rewards.flatten()
        m_mb_actions = m_mb_actions.reshape(m_mb_actions.shape[0] * m_mb_actions.shape[1], -1)
        w_mb_actions = w_mb_actions.flatten()
        m_mb_values = m_mb_values.flatten()
        w_mb_values = w_mb_values.flatten()
        m_mb_masks = m_mb_masks.flatten()
        w_mb_masks = w_mb_masks.flatten()
        m_mb_mhs = m_mb_mhs.reshape(m_mb_mhs.shape[0]*m_mb_mhs.shape[1], -1)
        m_mb_mh_s = m_mb_mh_s.reshape(m_mb_mh_s.shape[0]*m_mb_mh_s.shape[1], -1)
        w_mb_whs = w_mb_whs.reshape(w_mb_whs.shape[0]*w_mb_whs.shape[1], -1)
        wstates_ = np.zeros(shape=(self.wstates.shape[0] * self.master_ts, self.wstates.shape[1]), dtype=np.float32)
        return m_mb_obs, self.mstates, m_mb_rewards, m_mb_masks, m_mb_actions, m_mb_values, m_mb_mhs, m_mb_mh_s,\
               w_mb_obs, wstates_, w_mb_rewards, w_mb_masks, w_mb_actions, w_mb_values, w_mb_goals, w_mb_whs

class Runner_hrl_em(object):

    def __init__(self, env, model, master_ts = 5, worker_ts = 10, gamma=0.99, er_coef = .4, use_vae = False):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        self.nenv = env.num_envs
        self.master_ts = master_ts
        self.worker_ts = worker_ts
        self.nsteps = master_ts * worker_ts
        self.m_batch_ob_shape = (self.nenv * self.master_ts, nh, nw, nc)
        self.w_batch_ob_shape = (self.nenv * self.nsteps, nh, nw, nc)
        self.w_batch_goal_shape = (self.nenv * self.nsteps, self.model.cell)
        self.nc = nc
        obs = env.reset()
        h = self.model.get_wh(obs)
        self.mobs = np.copy(obs)
        self.mobs_ = np.copy(obs)
        self.wobs = np.copy(obs)
        self.mh = np.copy(h)
        self.mh_ = np.copy(h)
        self.wh = np.copy(h)
        self.gamma = gamma
        self.er_coef = er_coef
        self.use_vae = use_vae
        self.mstates = model.m_initial_state
        self.wstates = model.w_initial_state
        self.log_wpi =  model.init_log_wpi#shape = [nenv]
        '''
        whether the previous step is done
        '''
        self.dones = [True for _ in range(self.nenv)]
        '''
        after a goal is setting, whether the goal is done
        '''
        self.mdones = [False for _ in range(self.nenv)]#if an env is firstly done during a worker-ts rollout, then mdones of that env is True
        '''
        whether the worker's job is done
        '''
        self.wdones = [True for _ in range(self.nenv)]

    def run(self):
        m_mb_obs, m_mb_obs_ ,w_mb_obs = [], [], []
        m_mb_rewards, m_mb_actions, m_mb_values, m_mb_dones, m_mb_mhs, m_mb_mh_s = [], [], [], [], [], []
        w_mb_rewards, w_mb_actions, w_mb_values, w_mb_dones, w_mb_goals, w_mb_whs = [], [], [], [], [], []
        b_dones = []
        m_mb_states = self.mstates
        w_mb_states = np.zeros(shape=(self.wstates.shape[0] * self.master_ts, self.wstates.shape[1]), dtype=np.float32)
        goals = np.zeros(shape=(self.nenv, self.model.cell), dtype=np.float32)
        m_rewards = np.zeros(shape=(self.nenv), dtype=np.float32)
        for n in range(self.nsteps):
            if n % self.worker_ts == 0:
                '''
                once last step is done, a new goal should be set
                '''
                if self.use_vae:
                    goals, mvalues, mstates, actions, wvalues, wstates, log_wpi =\
                        self.model.step(mhs = self.wh, mstate = self.mstates, mmask = self.mdones,
                                        whs = self.wh, wstate = self.wstates, wmask = self.wdones,
                                        origin_goal = goals, goal_mask = [True for _ in range(self.nenv)])
                else:
                    goals, mvalues, mstates, actions, wvalues, wstates, log_wpi =\
                        self.model.step(mobs = self.wobs, mstate = self.mstates, mmask = self.mdones,
                                        wobs = self.wobs, wstate = self.wstates, wmask = self.wdones,
                                        origin_goal = goals, goal_mask = [True for _ in range(self.nenv)])
                m_mb_obs.append(np.copy(self.mobs))
                w_mb_obs.append(np.copy(self.wobs))
                m_mb_actions.append(goals)
                w_mb_actions.append(actions)
                m_mb_values.append(mvalues)
                w_mb_values.append(wvalues)
                m_mb_dones.append(self.mdones)
                w_mb_dones.append(self.wdones)
                m_mb_mhs.append(self.mh)
                w_mb_whs.append(self.wh)
                b_dones.append(self.dones)
                w_mb_goals.append(goals)

                obs, ers, dones, _ = self.env.step(actions)
                h = self.model.get_wh(obs)
                self.mdones = [False for _ in range(self.nenv)]#reset mdones
                m_rewards += (1. - np.asarray(self.mdones, dtype=np.bool)) * ers
                irs = ers + self.ir_coef * (-log_wpi+self.log_wpi)
                w_mb_rewards.append(irs)
                self.log_wpi = np.asarray(dones, dtype=np.bool)*self.model.init_log_wpi +\
                               (1. - np.asarray(dones, dtype=np.bool))*log_wpi

                show_img(obs[0], scope='obs')

                self.mstates = mstates
                self.wstates = wstates
                # self.one_shot = [False for _ in range(self.nenv)]
                for i, done in enumerate(dones):
                    if done:
                        if self.mdones[i] is False:
                            self.mdones[i] = True
                            self.mobs_[i] = obs[i]
                            self.mh_[i] = h[i]

                self.wdones = dones
                self.dones = dones
                self.wobs = obs
                self.wh = h
            else:
                if self.use_vae:
                    goals, mvalues, mstates, actions, wvalues, wstates, __ =\
                        self.model.step(mhs=self.wh, mstate=self.mstates, mmask=self.mdones,
                                        whs=self.wh, wstate=self.wstates, wmask=self.wdones,
                                        origin_goal=goals, goal_mask=self.dones)
                else :
                    goals, mvalues, mstates, actions, wvalues, wstates, __ =\
                        self.model.step(mobs = self.wobs, mstate =self.mstates, mmask = self.mdones,
                                        wobs = self.wobs, wstate = self.wstates, wmask = self.wdones,
                                        origin_goal = goals, goal_mask = self.dones)
                w_mb_obs.append(np.copy(self.wobs))
                w_mb_actions.append(actions)
                w_mb_values.append(wvalues)
                w_mb_dones.append(self.wdones)
                b_dones.append(self.dones)
                w_mb_goals.append(goals)
                w_mb_whs.append(self.wh)

                obs, ers, dones, _ = self.env.step(actions)
                h = self.model.get_wh(obs)
                m_rewards += (1. - np.asarray(self.mdones, dtype=np.bool)) * ers
                irs = ers + self.ir_coef * (log_wpi - self.log_wpi)
                w_mb_rewards.append(irs)
                self.log_wpi = np.asarray(dones, dtype=np.bool)*self.model.init_log_wpi +\
                               (1. - np.asarray(dones, dtype=np.bool))*log_wpi

                show_img(obs[0], scope='obs')

                self.wstates = wstates
                self.wdones = dones
                for i, done in enumerate(dones):
                    '''
                    if there exists one done that is True during the whole worker_ts-step process, then master's job is done
                    '''
                    if done:
                        if self.mdones[i] is False:
                            self.mdones[i] = True
                            self.mh_[i] = h[i]
                            if len(list(self.mobs_.shape)) < 4:
                                print('oh my god')
                            self.mobs_[i] = obs[i]

                self.dones = dones
                self.wobs = obs
                self.wh = h
                if n % self.worker_ts == self.worker_ts - 1:
                    '''
                    if so, then all workers' work are done
                    '''
                    m_mb_rewards.append(m_rewards)
                    m_rewards = np.zeros((self.nenv), dtype = np.float32)
                    self.wdones = [True for _ in range(self.nenv)]
                    self.mobs = np.copy(self.wobs)
                    self.mh_ = np.expand_dims(np.asarray(self.mdones, dtype=np.float32), axis=1)*self.mh_ +\
                               np.expand_dims(1.-np.asarray(self.mdones, dtype = np.float32), axis=1)*self.wh
                    m_mb_mh_s.append(self.mh_)
                    self.mobs_ = np.asarray(self.mdones, dtype = np.float32).reshape(self.nenv, 1,1,1) * self.mobs_ + \
                                 (1. - np.asarray(self.mdones, dtype=np.float32).reshape(self.nenv, 1,1,1)) * self.wobs
                    m_mb_obs_.append(self.mobs_)

        m_mb_dones.append(self.wdones)
        w_mb_dones.append(self.mdones)
        b_dones.append(self.dones)
        #batch of steps to batch of rollouts
        m_mb_obs = np.asarray(m_mb_obs, dtype = np.float32).swapaxes(1, 0).reshape(self.m_batch_ob_shape)
        w_mb_obs = np.asarray(w_mb_obs, dtype = np.float32).swapaxes(1, 0).reshape(self.w_batch_ob_shape)
        m_mb_rewards = np.asarray(m_mb_rewards, dtype = np.float32).swapaxes(1, 0)
        w_mb_rewards = np.asarray(w_mb_rewards, dtype = np.float32).swapaxes(1, 0).reshape(self.nenv * self.master_ts, self.worker_ts)
        m_mb_actions = np.asarray(m_mb_actions, dtype = np.float32).swapaxes(1, 0)
        w_mb_actions = np.asarray(w_mb_actions, dtype = np.int32).swapaxes(1, 0).reshape(self.nenv * self.master_ts, self.worker_ts)
        m_mb_values = np.asarray(m_mb_values, dtype = np.float32).swapaxes(1, 0)
        w_mb_values = np.asarray(w_mb_values, dtype = np.float32).swapaxes(1, 0).reshape(self.nenv * self.master_ts, self.worker_ts)
        m_mb_dones = np.asarray(m_mb_dones, dtype = np.bool).swapaxes(1, 0)#shape = [nenv, master_ts + 1]
        m_mb_masks = m_mb_dones[:, :-1]
        m_mb_dones = m_mb_dones[:, 1:]
        w_mb_dones = np.asarray(w_mb_dones, dtype = np.bool).swapaxes(1, 0)
        w_mb_masks = w_mb_dones[:, :-1]
        w_mb_dones = w_mb_dones[:, 1:]
        b_dones = np.asarray(b_dones, dtype = np.bool).swapaxes(1, 0)#shape = [nenv, master_ts + 1]
        b_masks = b_dones[:, :-1].reshape(self.nenv * self.master_ts, self.worker_ts)
        b_dones = b_dones[:, 1:].reshape(self.nenv * self.master_ts, self.worker_ts)
        w_mb_goals = np.asarray(w_mb_goals, dtype=np.float32).swapaxes(1, 0).reshape((self.w_batch_goal_shape))
        m_mb_mhs = np.asarray(m_mb_mhs, dtype=np.float32).swapaxes(1, 0)
        m_mb_mh_s = np.asarray(m_mb_mh_s, dtype=np.float32).swapaxes(1, 0)
        w_mb_whs = np.asarray(w_mb_whs, dtype=np.float32).swapaxes(1, 0)
        if self.use_vae:
            m_last_values = self.model.mvalue(self.mh_, self.mstates, self.mdones).tolist()
            w_last_values = self.model.wvalue(self.wh, self.wstates, self.dones, goals).tolist()
        else:
            m_last_values = self.model.mvalue(self.mobs_, self.mstates, self.mdones).tolist()
            w_last_values = self.model.wvalue(self.wobs, self.wstates, self.dones, goals).tolist()
        #discount/bootstrap off value fn
        #mrewards
        for n, (rewards, dones,  value) in enumerate(zip(m_mb_rewards, m_mb_dones, m_last_values)):
            rewards = rewards.tolist()#shape = [master_ts]
            dones = dones.tolist()#shape = [master_ts]
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            m_mb_rewards[n] = rewards
        m_mb_rewards = m_mb_rewards.flatten()
        for n, (rewards, dones, value) in enumerate(zip(w_mb_rewards, b_dones, w_last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            w_mb_rewards[n] = rewards
        w_mb_rewards = w_mb_rewards.flatten()
        m_mb_actions = m_mb_actions.reshape(m_mb_actions.shape[0] * m_mb_actions.shape[1], -1)
        w_mb_actions = w_mb_actions.flatten()
        m_mb_values = m_mb_values.flatten()
        w_mb_values = w_mb_values.flatten()
        m_mb_masks = m_mb_masks.flatten()
        w_mb_masks = w_mb_masks.flatten()
        m_mb_mhs = m_mb_mhs.reshape(m_mb_mhs.shape[0]*m_mb_mhs.shape[1], -1)
        m_mb_mh_s = m_mb_mh_s.reshape(m_mb_mh_s.shape[0]*m_mb_mh_s.shape[1], -1)
        w_mb_whs = w_mb_whs.reshape(w_mb_whs.shape[0]*w_mb_whs.shape[1], -1)
        return m_mb_obs, m_mb_states, m_mb_rewards, m_mb_masks, m_mb_actions, m_mb_values, m_mb_mhs, m_mb_mh_s,\
               w_mb_obs, w_mb_states, w_mb_rewards, w_mb_masks, w_mb_actions, w_mb_values, w_mb_goals, w_mb_whs

class Runner_ib(object):
    def __init__(self, env, model, nsteps=8, gamma=0.99):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.nenv = nenv
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc)
        self.nc = nc
        obs = env.reset()
        # h = model.get_wh(obs)
        self.obs = np.copy(obs)
        # self.whs = np.copy(h)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [True for _ in range(nenv)]

    def run(self):
        b_obs, b_whs, b_rewards, b_actions, b_values, b_dones, b_noises = [],[],[],[],[],[],[]
        b_states = self.states
        for n in range(self.nsteps):
            noises, actions, values, states, _, whs= self.model.step(self.obs, self.states, self.dones)
            b_obs.append(np.copy(self.obs))
            b_noises.append(noises)
            b_whs.append(np.copy(whs))
            b_actions.append(actions)
            b_values.append(values)
            b_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            show_img(obs[0], scope='current_obs')
            b_rewards.append(rewards)
            self.states = states
            self.dones = dones
            self.obs = obs
        b_dones.append(self.dones)
        #batch of steps to batch of rollouts
        b_obs = np.asarray(b_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        b_whs = np.asarray(b_whs, dtype=np.float32).swapaxes(1, 0)
        b_whs = b_whs.reshape(b_whs.shape[0]*b_whs.shape[1], -1)
        b_rewards = np.asarray(b_rewards, dtype=np.float32).swapaxes(1, 0)
        b_actions = np.asarray(b_actions, dtype=np.int32).swapaxes(1, 0)
        b_values = np.asarray(b_values, dtype=np.float32).swapaxes(1, 0)
        b_noises = np.asarray(b_noises, dtype=np.float32).swapaxes(1, 0)
        b_noises = b_noises.reshape(self.nenv*self.nsteps, b_noises.shape[2], b_noises.shape[3], b_noises.shape[4])
        b_dones = np.asarray(b_dones, dtype=np.bool).swapaxes(1, 0)
        b_masks = b_dones[:, :-1]
        b_dones = b_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(b_rewards, b_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            b_rewards[n] = rewards
        b_rewards = b_rewards.flatten()
        b_actions = b_actions.flatten()
        b_values = b_values.flatten()
        b_masks = b_masks.flatten()
        return b_obs, b_whs, b_states, b_rewards, b_masks, b_actions, b_values, b_noises

class Runner_svib(object):
    def __init__(self, env, model, nsteps=8, gamma=0.99):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.nenv = nenv
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc)
        self.nc = nc
        obs = env.reset()
        # noises = model.get_noise()
        # h = model.get_wh(obs, noises)
        self.obs = np.copy(obs)
        # self.whs = np.copy(h)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [True for _ in range(nenv)]
        self.episode_r = 0.

    def run(self):
        b_obs, b_whs, b_rewards, b_actions, b_values, b_dones, b_noises = [],[],[],[],[],[],[]
        b_states = self.states
        for n in range(self.nsteps):
            noises, actions, values, states, _, h = self.model.step(self.obs, self.states, self.dones)
            # actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)
            b_obs.append(np.copy(self.obs))
            # h = self.model.get_wh(self.obs, noises)
            b_whs.append(np.copy(h))
            b_actions.append(actions)
            b_values.append(values)
            b_dones.append(self.dones)
            b_noises.append(noises)
            obs, rewards, dones, _ = self.env.step(actions)
            # show_img(obs[0], scope='current_obs')
            b_rewards.append(rewards)
            self.states = states
            self.dones = dones
            self.obs = obs
            # self.whs = h
        b_dones.append(self.dones)
        #batch of steps to batch of rollouts
        b_obs = np.asarray(b_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        b_whs = np.asarray(b_whs, dtype=np.float32).swapaxes(1, 0)
        b_whs = b_whs.reshape(self.nenv*self.nsteps, b_whs.shape[-1])
        b_rewards = np.asarray(b_rewards, dtype=np.float32).swapaxes(1, 0)
        b_actions = np.asarray(b_actions, dtype=np.int32).swapaxes(1, 0)
        b_values = np.asarray(b_values, dtype=np.float32).swapaxes(1, 0)
        b_noises = np.asarray(b_noises, dtype=np.float32).swapaxes(1, 0)
        b_noises = b_noises.reshape(self.nenv*self.nsteps, b_noises.shape[2], b_noises.shape[3], b_noises.shape[4])
        b_dones = np.asarray(b_dones, dtype=np.bool).swapaxes(1, 0)
        b_masks = b_dones[:, :-1]
        b_dones = b_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(b_rewards, b_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            b_rewards[n] = rewards
        b_rewards = b_rewards.flatten()
        b_actions = b_actions.flatten()
        b_values = b_values.flatten()
        b_masks = b_masks.flatten()
        # return b_obs, b_whs, b_states, b_rewards, b_masks, b_actions, b_values

        return b_obs, b_whs, b_states, b_rewards, b_masks, b_actions, b_values, b_noises

class Runner_vae_a2c(object):
    def __init__(self, env, model, nsteps=8, gamma=0.99, algo='vae_a2c', phase='explore'):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.nenv = nenv
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc)
        self.nc = nc
        obs = env.reset()
        h = model.embedding(obs)
        self.obs = np.copy(obs)
        self.whs = np.copy(h)
        self.gamma = gamma
        self.algo = algo
        self.phase = phase
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [True for _ in range(nenv)]

    def change_pahse(self, phase):
        self.phase = phase

    def run(self):
        if self.algo == 'vae_a2c':
            if self.phase == 'explore':
                b_obs = []
                for n in range(self.nsteps):
                    actions, decoder = self.model.exploration(self.obs)
                    show_img(self.obs[0], scope='obs')
                    show_img(decoder[0], scope='decoder')
                    b_obs.append(np.copy(self.obs))
                    obs, rewards, dones, _ = self.env.step(actions)
                    self.obs = obs
                b_obs = np.asarray(b_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
                return b_obs

        b_obs, b_whs, b_rewards, b_actions, b_values, b_dones = [],[],[],[],[],[]
        b_states = self.states
        for n in range(self.nsteps):
            actions, values, states, _ = self.model.step(algo=self.algo, wobs=self.obs, whs=self.whs, states=self.states, dones=self.dones)
            b_obs.append(np.copy(self.obs))
            b_whs.append(np.copy(self.whs))
            b_actions.append(actions)
            b_values.append(values)
            b_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            h = self.model.embedding(obs)
            show_img(obs[0], scope='obs')
            if self.algo == 'vae_a2c':
                show_img(self.model.embedding_decoder(h)[0], scope='decoder')
            b_rewards.append(rewards)
            self.states = states
            self.dones = dones
            self.obs = obs
            self.whs = h
        b_dones.append(self.dones)
        #batch of steps to batch of rollouts
        b_obs = np.asarray(b_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        b_whs = np.asarray(b_whs, dtype=np.float32).swapaxes(1, 0)
        b_whs = b_whs.reshape(b_whs.shape[0]*b_whs.shape[1], -1)
        b_rewards = np.asarray(b_rewards, dtype=np.float32).swapaxes(1, 0)
        b_actions = np.asarray(b_actions, dtype=np.int32).swapaxes(1, 0)
        b_values = np.asarray(b_values, dtype=np.float32).swapaxes(1, 0)
        b_dones = np.asarray(b_dones, dtype=np.bool).swapaxes(1, 0)
        b_masks = b_dones[:, :-1]
        b_dones = b_dones[:, 1:]
        last_values = self.model.value(algo=self.algo, whs=self.whs, wobs=self.obs,
                                       states=self.states, dones=self.dones).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(b_rewards, b_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            # raw_ir += discount_with_dones(rewards, dones, self.gamma)[0]
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            b_rewards[n] = rewards
        b_rewards = b_rewards.flatten()
        b_actions = b_actions.flatten()
        b_values = b_values.flatten()
        b_masks = b_masks.flatten()
        return b_obs, b_whs, b_states, b_rewards, b_masks, b_actions, b_values