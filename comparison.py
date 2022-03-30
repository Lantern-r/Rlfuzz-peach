#!/usr/bin/env python
# coding: utf-8
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd
import gym
import sys
import time
import datetime
import rlfuzz
import argparse

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Lambda
from keras.callbacks import TensorBoard, Callback
from keras.optimizers import Adam
from keras import backend as K

from rl.agents import DDPGAgent
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import Callback

from rlfuzz.envs.restart_remote_monitor import restart_ssh

np.random.seed(5)

os.chdir('/home/real/rlfuzz-socket/')

# 每个环境的初始文件路径
INITIAL_SEED_PATH = {
    'FuzzBase64-v0': r'/home/real/rlfuzz-master/rlfuzz/mods/lava-m-mod/lava_corpus/LAVA-M/base64/inputs/utmp.b64',
    'FuzzMd5sum-v0': r'/home/real/rlfuzz-master/rlfuzz/mods/lava-m-mod/lava_corpus/LAVA-M/md5sum/inputs/bin-ls-md5s',
    'FuzzUniq-v0': r'/home/real/rlfuzz-master/rlfuzz/mods/lava-m-mod/lava_corpus/LAVA-M/uniq/inputs/man-clang3-sorted',
    'FuzzWho-v0': r'/home/real/rlfuzz-master/rlfuzz/mods/lava-m-mod/lava_corpus/LAVA-M/who/inputs/utmp',
    'FuzzAC68U-v0': r'/home/real/rlfuzz-socket/rlfuzz/mods/router-mod/AC68U/4.txt',
    'FuzzAC9-v0': r'/home/real/AIfuzz/multimutatefuzz/rlfuzz/gym_fuzzing/gym_fuzz1ng/mods/router-mod/AC68U/host10.txt',
    'Fuzzgzip-v0': r'/home/real/rlfuzz-socket/rlfuzz/mods/gzip-mod/seed',#/1.ppt.gz',
    'Fuzzlibpng-v0': r'/home/real/rlfuzz-socket/rlfuzz/mods/fuzzer-test-suite-mod/libpng-1.2.56/seeds/pngtest.png',
    'FuzzPngquant-v0':r'/home/real/rlfuzz-socket/rlfuzz/mods/pngquant-mod/pngquant-master/test/img/metadata.png'
}


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.start_time = time.time()

    def on_train_end(self, logs={}):
        self.end_time = time.time()
        self.training_time = self.end_time - self.start_time


def show_graghs(env, history, training_time, json_name):
    data_json = {}

    if args.peach:
        data = env.seed_block
        data_json['seed_block'] = data

    data = env.input_len_history
    data_json['input_len_history'] = data

    data = env.transition_count
    data_json['transition_count'] = data

    data = env.reward_history
    data_json['reward_history'] = data

    # from collections import Counter
    data = env.mutate_history
    data_json['mutate_history'] = data

    if args.peach:
        data = env.mutate_num_history
        data_json['mutate_num_history'] = data

    data = env.unique_path_history
    data_json['unique_path_history'] = data

    data = env.virgin_count
    data_json['virgin_count'] = data

    data_json['training_time'] = training_time

    if not os.path.exists('./data/{}'.format(DIR_NAME)):
        os.makedirs('./data/{}'.format(DIR_NAME))
    np.save('./data/{}/{}.npy'.format(DIR_NAME, json_name), data_json)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Comparison script for different env and methods.')
    parser.add_argument('--env', '-e', help='gym env.')
    parser.add_argument('--method', '-m', help='model method.')
    parser.add_argument('--start_time', '-st', help='start time.')
    parser.add_argument('--use_seed', '-us', action='store_true', help='if use initial seed.')
    parser.add_argument('--activation', '-a', default='relu', help='activation function.')
    parser.add_argument('--steps', default=20000, help='all steps number.', type=int)
    parser.add_argument('--radio', default=0.1, help='warmup radio.', type=float)
    parser.add_argument('--peach', action='store_true', help='Use Peach')
    parser.add_argument('--pit', help='Pit File Path')
    args = parser.parse_args()

    ALL_STEPS = args.steps
    WARMUP_STEPS = int(ALL_STEPS * args.radio)
    ACTIVATION = args.activation
    ENV_NAME = args.env
    METHOD = args.method.lower()
    START_TIME = args.start_time
    print('[+] {} {}'.format(ENV_NAME, METHOD))

    if args.peach and not args.use_seed:
        print('please use seed in peach mode')
        exit(0)
    if args.use_seed:
        DIR_NAME = 'COMPARISON-{}-{}-{}-{}-with_seed'.format(ACTIVATION, ALL_STEPS, WARMUP_STEPS, START_TIME)
    else:
        DIR_NAME = 'COMPARISON-{}-{}-{}-{}'.format(ACTIVATION, ALL_STEPS, WARMUP_STEPS, START_TIME)
    print('[+] {}'.format(DIR_NAME))

    # [e.id for e in gym.envs.registry.all()]
    if ENV_NAME in ['FuzzBase64-v0', 'FuzzMd5sum-v0', 'FuzzUniq-v0', 'FuzzWho-v0', 'FuzzPngquant-v0',
                    'FuzzAC68U-v0', 'FuzzAC9-v0', 'Fuzzgzip-v0', 'Fuzzlibpng-v0','FuzzPngquant-v0'] and METHOD in ["random", "ddpg",
                                                                                                 "dqn", "double-dqn",
                                                                                                 "duel-dqn"]:
        env = gym.make(ENV_NAME)
        env.seed(5)  # 起点相同

        if args.use_seed:  # 输入初始数据
            SEED_PATH = INITIAL_SEED_PATH[ENV_NAME]
            if os.path.exists(SEED_PATH):
                env.set_seed(SEED_PATH)
                if args.peach:
                    env.set_peach_seed(SEED_PATH)
                    env.set_peach()
            else:
                print('[!] {} not exist !'.format(SEED_PATH))

        # nb_actions = env.action_space.shape[0]
        # env.setDiscreteEnv()
        if args.peach:
            nb_actions = env.action_space['mutate'].n + len(env.action_space['loc']) * env.action_space['loc'][
                0].n + len(
                env.action_space['density']) * env.action_space['density'][0].n + env.action_space['block_num'].n
            nb_observation = env.observation_space.shape[0]
        else:
            nb_actions = env.action_space['mutate'].n + len(env.action_space['loc']) * env.action_space['loc'][
                0].n + len(
                env.action_space['density']) * env.action_space['density'][0].n
            nb_observation = env.observation_space.shape[0]



        if METHOD == "random":
            nb_steps = []

            start = time.time()
            for s in range(ALL_STEPS):
                if s % 10000 == 0:
                    print('[+] {} steps...'.format(s))
                state, reward, done, _ = env.step(env.action_space.sample())
                if done:
                    nb_steps.append(s)
            end = time.time()

            print('[+] {}s'.format(end - start))
            history = {}
            history['nb_steps'] = nb_steps
            show_graghs(env, history, end - start, '{}-random-{}-{}'.format(ENV_NAME, ACTIVATION, WARMUP_STEPS))

        elif METHOD == "ddpg":
            actor_input = Input(shape=(1,) + env.observation_space.shape, name='actor_observation_input')
            f_actor_input = Flatten()(actor_input)
            x = Dense(1024)(f_actor_input)
            x = Activation(ACTIVATION)(x)
            x = Dense(128)(x)
            x = Activation(ACTIVATION)(x)
            x = Dense(nb_actions)(x)
            x = Activation('tanh')(x)  # include nav
            actor = Model(inputs=actor_input, outputs=x)

            critic_action_input = Input(shape=(nb_actions,), name='critic_action_input')
            critic_observation_input = Input(shape=(1,) + env.observation_space.shape, name='critic_observation_input')
            f_critic_observation_input = Flatten()(critic_observation_input)
            x = Concatenate()([critic_action_input, f_critic_observation_input])
            x = Dense(1024)(x)
            x = Activation(ACTIVATION)(x)
            x = Dense(128)(x)
            x = Activation(ACTIVATION)(x)
            x = Dense(1)(x)
            x = Activation('sigmoid')(x)
            critic = Model(inputs=[critic_action_input, critic_observation_input], outputs=x)

            memory = SequentialMemory(limit=100000, window_length=1)
            random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
            agent = DDPGAgent(
                nb_actions=nb_actions,
                actor=actor,
                critic=critic,
                critic_action_input=critic_action_input,
                memory=memory,
                nb_steps_warmup_critic=WARMUP_STEPS,
                nb_steps_warmup_actor=WARMUP_STEPS,
                random_process=random_process,
                gamma=.99,
                target_model_update=1e-3
            )
            agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

            timeCb = TimeHistory()
            history = agent.fit(env, nb_steps=ALL_STEPS, visualize=False, verbose=1, callbacks=[timeCb])

            show_graghs(env, history.history, timeCb.training_time,
                        '{}-ddpg-{}-{}'.format(ENV_NAME, ACTIVATION, WARMUP_STEPS))

        elif METHOD == "dqn":  # DQN
            model = Sequential()
            model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
            model.add(Dense(1024))
            model.add(Activation(ACTIVATION))
            model.add(Dense(128))
            model.add(Activation(ACTIVATION))
            model.add(Dense(nb_actions))
            model.add(Activation('softmax'))

            dqn = DQNAgent(model=model, nb_actions=nb_actions,
                           memory=SequentialMemory(limit=100000, window_length=1),
                           nb_steps_warmup=WARMUP_STEPS,  # default 1000
                           enable_double_dqn=False,  # default True
                           enable_dueling_network=False,  # default False
                           #                dueling_type='avg', # avg max naive, defalut avg
                           target_model_update=1e-2,  # soft (hard >= 1)
                           policy=EpsGreedyQPolicy()  # default
                           )
            dqn.compile(Adam(lr=1e-3, clipnorm=1.), metrics=['mae'])

            timeCb = TimeHistory()
            history = dqn.fit(env, nb_steps=ALL_STEPS, visualize=False, verbose=1, callbacks=[timeCb])

            show_graghs(env, history.history, timeCb.training_time,
                        '{}-dqn-{}-{}'.format(ENV_NAME, ACTIVATION, WARMUP_STEPS))
        elif METHOD == "double-dqn":
            model = Sequential()
            model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
            model.add(Dense(1024))
            model.add(Activation(ACTIVATION))
            model.add(Dense(128))
            model.add(Activation(ACTIVATION))
            model.add(Dense(nb_actions))
            model.add(Activation('softmax'))

            dqn = DQNAgent(model=model, nb_actions=nb_actions,
                           memory=SequentialMemory(limit=100000, window_length=1),
                           nb_steps_warmup=WARMUP_STEPS,  # default 1000
                           enable_double_dqn=True,  # default True
                           enable_dueling_network=False,  # default False
                           #                dueling_type='avg', # avg max naive, defalut avg
                           target_model_update=1e-2,  # soft (hard >= 1)
                           policy=EpsGreedyQPolicy()  # default
                           )
            dqn.compile(Adam(lr=1e-3, clipnorm=1.), metrics=['mae'])

            timeCb = TimeHistory()
            history = dqn.fit(env, nb_steps=ALL_STEPS, visualize=False, verbose=1, callbacks=[timeCb])

            show_graghs(env, history.history, timeCb.training_time,
                        '{}-double-dqn-{}-{}'.format(ENV_NAME, ACTIVATION, WARMUP_STEPS))
        else:
            model = Sequential()
            model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
            model.add(Dense(1024))
            model.add(Activation(ACTIVATION))
            model.add(Dense(128))
            model.add(Activation(ACTIVATION))
            model.add(Dense(nb_actions))
            model.add(Activation('softmax'))

            dqn = DQNAgent(model=model, nb_actions=nb_actions,
                           memory=SequentialMemory(limit=100000, window_length=1),
                           nb_steps_warmup=WARMUP_STEPS,  # default 1000
                           enable_double_dqn=False,  # default True
                           enable_dueling_network=True,  # default False
                           #                dueling_type='avg', # avg max naive, defalut avg
                           target_model_update=1e-2,  # soft (hard >= 1) # 10000
                           policy=EpsGreedyQPolicy()  # default
                           )
            dqn.compile(Adam(lr=1e-3, clipnorm=1.), metrics=['mae'])

            timeCb = TimeHistory()
            history = dqn.fit(env, nb_steps=ALL_STEPS, visualize=False, verbose=1, callbacks=[timeCb])

            show_graghs(env, history.history, timeCb.training_time,
                        '{}-duel-dqn-{}-{}'.format(ENV_NAME, ACTIVATION, WARMUP_STEPS))
