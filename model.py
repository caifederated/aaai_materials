"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import numpy as np
import os
import sys
import tensorflow as tf

class Model(ABC):

    def __init__(self, seed, lr, train_bs=32, test_bs=32, input_shape=None, optimizer=None, sem=False):
        self.lr = lr
        self.seed = seed
        self.train_bs = train_bs
        self.test_bs = test_bs
        self._optimizer = optimizer
        self._input_shape = input_shape
        self._sem = sem
        self._run_sgd = False
        
    def create_dataset(self, data, set_to_use='train'):
        features = self.process_x(data['x'])
        labels = self.process_y(data['y'])
        data = tf.data.Dataset.from_tensor_slices((features, labels))
        if set_to_use == 'train':
            if self._run_sgd:
                size = len(labels)
            else:
                size = self.train_bs
            buffer = size + 1000
            data = data.shuffle(buffer_size=buffer).batch(size)
        elif set_to_use == 'pred':
            data = data.batch(self.test_bs)
        else:
            data = data.batch(self.test_bs)
        return data
    
    @property
    def SGD(self):
        return self._run_sgd
    
    @SGD.setter
    def SGD(self, val):
        self._run_sgd = val
    
    @property
    def get_SEM(self):
        return self._sem
    
    @property
    def get_input_shape(self):
        return self._input_shape

    @property
    def optimizer(self):
        """Optimizer to be used by the model."""
        if self._optimizer is None:
            self._optimizer = tf.keras.optimizers.SGD(self.lr, 0.95)
#             self._optimizer = tf.keras.optimizers.Adam(self.lr)

        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
    
    @abstractmethod
    def create_model(self):
        return None

    @abstractmethod
    def process_x(raw_x):
        return raw_x
    
    @abstractmethod
    def process_y(raw_y):
        return raw_y
    
    def train(self, data, num_epochs=1, batch_size=10):
        return
            
    def test(self, data):
        return

