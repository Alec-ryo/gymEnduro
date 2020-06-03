import argparse
import gym
import time
import keyboard
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import random
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import Adam

# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras

from skimage.color import rgb2gray

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

import math 

from numpy import asarray, load
from numpy import savez_compressed

INPUT_SHAPE = (84,84)
WINDOW_LENGTH = 1
nb_actions = 3
# set True to store observation and action on RAM memory while playing Enduro
save_information = True

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
    elif(not(keyboard.is_pressed('space'))):
        if keyboard.is_pressed('a') and not(keyboard.is_pressed('d')):
            command = 3
        elif keyboard.is_pressed('d') and not(keyboard.is_pressed('a')):
            command = 2
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
    img = Image.fromarray(data, 'L')
    img.save('created_screen.png')
    img.show()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def create_model(observation):
    # observation = rgb2gray(observation)
    h, w = len(observation), len(observation[0])
    print("h:",h)
    print("w:",w)
    n_input_layer = h*w
    n_output_layer = nb_actions
    n_hidden_layer = round(math.sqrt((n_input_layer*n_output_layer)))
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=INPUT_SHAPE),
        keras.layers.Dense(n_hidden_layer, activation='tanh'),
        keras.layers.Dense(3, activation='softmax')
    ])
    print(model.summary())

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def playing_game(env):
    observation_list = []
    action_list = []
    sleep_time = 0.1
    # 10 iterations to 1 second
    # 60 seconds to 1 minute
    # 5 minutes gaming
    time_game = int(1/sleep_time)*60*5
    for i_episode in range(1):
        observation = env.reset()
        action = 8
        previous_reward = 0.0

        for t in range(time_game):
            env.render()
            observation, reward, done, info = env.step(action)
            # print(env.action_space)
            # print(reward)
            previous_action = action
            action = controller()
            # action = linear_estimator(reward, previous_reward, previous_action)
            previous_reward = reward
            # if t%20 == 0:
            #     img = rgb2gray(observation)
            #     plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
            #     plt.show()
            
            observation = rgb2gray(observation)
            img = Image.fromarray(observation)
            img = img.resize(INPUT_SHAPE)
            processed_observation = np.array(img)

            print(processed_observation)

            # print(observation)
            if save_information:
                observation_list.append(processed_observation)
                action_list.append(action)
            time.sleep(sleep_time)
    
    return observation_list, action_list

def arguments():
    # comandos de linha de comando
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--env-name', type=str, default='Enduro-v0')
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()
    return args

def save_on_npy(observation_list, action_list):
    savez_compressed('observation_list.npz', observation_list)
    savez_compressed('action_list.npz', action_list)

def game_to_generate_images(env):
    observation_list, action_list = playing_game(env)
    save_on_npy(observation_list, action_list)
    return observation_list, action_list

# onehot is array representing number in binary representation
def onehot_to_int(onehot):
    return int("".join(str(x) for x in onehot), 2) 

args = arguments()
env = gym.make(args.env_name)
observation = env.reset()
l_training = True

if l_training:
    observation_list, action_list = game_to_generate_images(env)
else:
    load_actions = np.load('banco/action_list.npz')
    print(load_actions.keys())
    action_list = load_actions.f.arr_0
    load_observations = np.load('banco/observation_list.npz')
    observation_list = load_observations.f.arr_0

#type
observation = rgb2gray(observation)
img = Image.fromarray(observation)
img = img.resize(INPUT_SHAPE)
processed_observation = np.array(img)

x_train = np.random.random((1000, 20))
print(type(x_train))

# model = create_model(processed_observation)

observation_arr = np.array(observation_list)
action_arr = np.array(action_list)

print(len(action_list))
print("type action_list:", type(action_list))
# np.set_printoptions(threshold=np.inf)

print("type action_list:", type(action_list))
print("observation_list:", observation_list)
# model.fit(observation_arr, action_arr, epochs=150, batch_size=10)

observation_list, action_list = playing_game(env)

save_on_npy(observation_list, action_list)

env.close()