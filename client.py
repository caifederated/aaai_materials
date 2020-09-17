import random
import warnings
import numpy as np

from utils_misc import get_flatten_vec, compute_euclidean_distance, has_vec_not_init


class Client:
    
    def __init__(self, client_id, group=None, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}, model=None):
        self._model = model
        self.id = client_id # integer
        self.group = group
        self.num_train = len(train_data['y'])
        self.num_test = len(eval_data['y'])
        self.train_data = train_data
        self.eval_data = eval_data
        self._sem_vector = None     
        
    def find_cluster_identity(self, c_model, nets, firstRound):
        netidxs = np.array([0.0] * len(nets))
        if firstRound:
            netidxs = np.random.uniform(0.0, 1.0, (len(nets)))
        elif (self._model.get_SEM):
            if has_vec_not_init(self.sem_vec):
                netidxs = np.random.uniform(0.0, 1.0, (len(nets)))
            else:
                layer = len(nets[0]) - 4
                for i in range(len(nets)):
                    centroid = get_flatten_vec(nets[i])
                    netidxs[i] = compute_euclidean_distance(self.sem_vec, centroid)
        else:
            for i in range(len(nets)):
                c_model.set_weights(nets[i])
                ds = self._model.create_dataset(self.eval_data, set_to_use='test')
                test_loss, _ = c_model.evaluate(ds, verbose=0)
                netidxs[i] = test_loss
            
        min_net = np.argmin(netidxs)
        return min_net

    def train(self, num_epochs=1, batch_size=10, minibatch=None):
        return None, None, None

    def test(self, c_model, set_to_use='test'):        
        assert set_to_use in ['train', 'test']
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test':
            data = self.eval_data          
        return c_model.evaluate(data, verbose=0)
    
    @property
    def sem_vec(self):
        return self._sem_vector
    
    @sem_vec.setter
    def sem_vec(self, vec):
        self._sem_vector = vec

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.eval_data is None:
            return 0
        return self.num_test

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return self.num_train

    @property
    def num_samples(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data is not None:
            train_size = self.num_train

        test_size = 0 
        if self.eval_data is not None:
            test_size = self.num_test
        return train_size + test_size

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model
