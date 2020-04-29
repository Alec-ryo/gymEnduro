import argparse
import gym
import time
import keyboard
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import random
from keras.layers import Dense, Activation, Flatten, Permute, Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam

from skimage.color import rgb2gray

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

import math 

INPUT_SHAPE = (84,84)
WINDOW_LENGTH = 4
nb_actions = 3

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

def create_model(observation):
    h, w = len(observation[0]), len(observation[1])
    n_input_layer = h*w*3
    n_output_layer = nb_actions
    n_hidden_layer = round(math.sqrt((n_input_layer*n_output_layer)))
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

    model = Sequential()

    # (width, height, channels)
    model.add(Permute((2, 3, 1), input_shape=input_shape))

    # model.add(n_input_layer)
    # model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
    # model.add(Activation('tanh'))
    model.add(Flatten())
    model.add(Dense(n_hidden_layer))
    model.add(Activation('tanh'))
    model.add(Dense(n_output_layer))
    model.add(Activation('softmax'))
    print(model.summary())
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

def arguments():
    # comandos de linha de comando
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--env-name', type=str, default='Enduro-v0')
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()
    return args

class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        # assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

args = arguments()
env = gym.make(args.env_name)
observation = env.reset()
np.random.seed(123)
env.seed(123)

model = create_model(observation)

# Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that now you can use the built-in tensorflow.keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=50000, log_interval=10000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)


# playing_game(env)

env.close()
