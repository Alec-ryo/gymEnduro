import argparse
import gym
import time
import keyboard
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import random
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from skimage.color import rgb2gray

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

def controller():

    command = 0
    # Se o gatilho estiver apertado
    if keyboard.is_pressed('space'):
        if keyboard.is_pressed('a') and not(keyboard.is_pressed('d')):
            command = 8
        elif keyboard.is_pressed('d') and not(keyboard.is_pressed('a')):
            command = 7
        else:
            command = 1
    else:
        if keyboard.is_pressed('a') and not(keyboard.is_pressed('d')):
            command = 2
        elif keyboard.is_pressed('d') and not(keyboard.is_pressed('a')):
            command = 3
        else:
            command = 0

    return command

def linear_estimator(reward, previous_reward, previous_action):
    if previous_action == 7:
        antiaction = 8
    else:
        antiaction = 7

    if reward < previous_reward:
        return antiaction
    else:
        return previous_action

def print_image(observation):              
    w, h = len(observation), len(observation[0])
    data = observation
    img = Image.fromarray(data, 'RGB')
    img.save('created_screen.png')
    img.show()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def def_agent(observation):
    h, w = observation[0], observation[1]
    n_input_layer = h*w*3
    n_output_layer = 3
    n_hidden_layer = round(sqrt(n_input_layer*n_output_layer))
    model = Sequential()
    # model.add(Flatten(input_shape = observation))
    model.add(n_input_layer)
    model.add(Dense(n_hidden_layer, activation='tanh'))
    model.add(Dense(n_output_layers, activation='softmax'))
    return model

def playing_game(env):
    for i_episode in range(20):
        observation = env.reset()
        action = 8
        previous_reward = 0.0

        for t in range(1000):
            env.render()
            observation, reward, done, info = env.step(action)
            # print(env.action_space)
            # print(reward)
            previous_action = action
            action = controller()
            # action = linear_estimator(reward, previous_reward, previous_action)
            previous_reward = reward
            if t%20 == 0:
                img = rgb2gray(observation)
                plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
                plt.show()
            # time.sleep(0.2)

class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

# comandos de linha de comando
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='Enduro-v0')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

playing_game(env)

env.close()
