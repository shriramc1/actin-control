
import random
import numpy as np
from collections import namedtuple


class ReplayBuffer:
    def __init__(self, capacity):
        """Replay buffer to store samples of (state, action, reward, next_state, done)
        Arguments:
            capacity {int} -- capacity of replay buffer
        """
        self.capacity = capacity
        self.buffer = []
        self.position = int(0)
        self.transition = namedtuple("Transition", 
                                     ("state", "action", "reward", "next_state", "done"))

    def push(self, state, action, reward, next_state, done):
        """Adds new tuple of (state, action, reward, next_state, done) sample to replay buffer
        If buffer is full (i.e. more than self. capacity) add new samples to beginning of
        replay buffer
        Arguments:
            state {np.array} -- state of environment
            action {int} -- action taken in environment
            reward {float} -- reward received from environment
            next_state {np.array} -- next state of environment
            done {bool} -- whether episode is done
        """

        to_add = [state, action, reward, next_state, done]
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = self.transition(*to_add)
        self.position = int((self.position + 1) % self.capacity)

    def get_element_from_buffer(self, element_num):
        """Returns element_num element from replay buffer
        Arguments:
            element_num {int} -- element number to return from replay buffer
        Returns:
            tuple -- tuple of (state, action, reward, next_state, done) from replay buffer
        """
        return self.buffer[element_num]    

    def sample(self, batch_size):
        """Samples batch_size samples from replay buffer
        Arguments:
            batch_size {int} -- number of samples to sample from replay buffer
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Returns length of replay buffer
        Returns:
            int -- length of replay buffer
        """
        return (len(self.buffer))

    def load_buffer(self, filename):
        """Loads replay buffer from file
        Arguments:
            filename {str} -- name of file to load replay buffer from
        """
        self.buffer = np.load(filename, allow_pickle=True).tolist()
        self.position = len(self.buffer)
