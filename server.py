import numpy as np
import copy
import tensorflow as tf

from sklearn.cluster import MiniBatchKMeans
from server_constants import *

## neuron match
from pfnm_communication import layer_group_descent, gaus_init
from utils_misc import agg_layers, reverse_layers, reverse_matched_2_original_weights
from utils_misc import get_flatten_vec, has_vec_not_init, get_all_L_next, get_first_den_indx

import logging

logger = logging.getLogger("server")

class Server:
    
    def __init__(self, dataclass, num_clusters):
        self.dataclass = dataclass
        self.selected_clients = []
        self.batch_weights = []
        self.num_clusters = num_clusters
        self.firstCommunicationRound = True
#         self._global_models = []
        self._clients_metrics = {METRIC_ACCURACY_KEY:list(), METRIC_LOSS_KEY: list()}
        self._sem_dataset = []
        
        model = self.dataclass.create_model() 
        model.build(self.dataclass.get_input_shape)
        weights = model.get_weights()
        self.init_fcidx(weights)        
        self.batch_weights = [copy.copy(weights) for _ in range(self.num_clusters)]            
        
    @property
    def get_models(self):
        return self._global_models
    
    @property
    def shared_nn(self):
        if self._sharednn == None:
            m = self.dataclass.create_model()
            self._sharednn = m
        return self._sharednn
        
    
    @property
    def history(self):
        return self._clients_metrics
    
    def init_fcidx(self, weights):
        self._units = get_all_L_next(weights) 
        if len(get_all_L_next(weights)) == 4 :
            self.fc_idx = 6
        else:
            self.fc_idx = len(weights) - 2        
    
    def init_sem_data(self, D, clients=None):
        def rnd_vec():
            units = self._units      
            num_classes = units[-1] 
            ret = gaus_init(units[0:-1], D, num_classes) 
            # logger.info("units is {}, and num_classes is {}".format(units, num_classes))
            return  get_flatten_vec(ret, self.fc_idx)
        
        if clients == None:
            clients = self.selected_clients

        theta_batch = list()
        for c in clients:
            if has_vec_not_init(c.sem_vec):
                vec = rnd_vec()
                logger.info("using gaus init to set client sem vectors and dimensions {}, vec shape {}".format(D, vec.shape))
                theta_batch.append(vec)
            else:
                vec = c.sem_vec
                logger.info("using client own sem vector, vec shape {}".format(vec.shape))
                theta_batch.append(c.sem_vec)
        
        self._sem_dataset = theta_batch
        return
    
    def assignments(self, g_w, clients):
        id2clients = {}
        clients2id = {}
        assign_using_sem = len(self._sem_dataset) > 0
        if not assign_using_sem:
            for c in clients:
                cl_id = c.find_cluster_identity(self.shared_nn, g_w,  self.firstCommunicationRound)
                if cl_id not in id2clients:
                    id2clients[cl_id] = []
                id2clients[cl_id].append(c)
                clients2id[c.id] = cl_id
        else:
            cluster_labels = self.update_kmeans_estimate(g_w, clients)
            for k in range(self.num_clusters):
                for index, cl_id in enumerate(cluster_labels):
                    if cl_id == k:
                        if k not in id2clients:
                            id2clients[k] = list()
                        id2clients[k].append(clients[index])
                        c = clients[index]
                        clients2id[c.id] = k
                        
        return id2clients, clients2id
        
    def init_shared_models(self, bbp_map=False):
        self._sharednn = None
        
    def update_weights(self, weights, key2clients, clients2key, mode=None):
        def simple_update(weights, num):
            w_agg = []
            for w in weights:
                w_agg.append(w / float(num))
            return w_agg
        
        tot_train_samples = {i: 0 for i in range(self.num_clusters)}
        for (key,val) in key2clients.items():
            tot_train_samples[key] = sum([c.num_train_samples for c in val])
        
        for key in key2clients:
            arr = [weights[c] for c in clients2key if clients2key[c] == key]            
            self.batch_weights[key] = np.sum(arr, axis=0) / tot_train_samples[key]
            
            
    def evaluate_global_models(self, eval_data):
        K_clus = range(self.num_clusters)
        L_clus = [0.0] * self.num_clusters
        A_clus = [0.0] * self.num_clusters
        model = self.shared_nn
        for k, weight in zip(K_clus, self.batch_weights):
            model.set_weights(weight)
            loss, acc = model.evaluate(eval_data, verbose = 1)
            A_clus[k] = acc
            L_clus[k] = loss
            print("cluster %2d test accuracy %.2f" % (k, acc))
        return A_clus, L_clus
            
    def evaluate_global_models_with_ma(self, eval_data):
        L_clus = [0.0] * self.num_clusters
        A_clus = [0.0] * self.num_clusters        
        rever_global_weights = reverse_matched_2_original_weights(
            self.batch_weights, self.num_clusters, self.shared_nn)
        K_clus = range(self.num_clusters)
        model = self.shared_nn
        for k, weight in zip(K_clus, rever_global_weights):
            model.set_weights(weight)
            loss, acc = model.evaluate(eval_data, verbose = 1)
            A_clus[k] = acc
            L_clus[k] = loss            
            print("cluster %2d test accuracy %.2f" % (k, acc)) 
        return A_clus, L_clus

    def select_clients(self, possible_clients, num_clients=2):
        # put init model here
        self.init_shared_models()
        num_clients = min(num_clients, len(possible_clients))
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]
    
    def train_model(self, num_epochs=1, clients=None, agg_mode=None):
        if clients == None:
            clients = self.selected_clients
            
        
        # i) first find which cluster fits client most
        id2clients, clients2id = self.assignments(self.batch_weights, clients)       
        
        # ii then we use dict to train them all
        avg_w = {}
        acc_list = self._clients_metrics[METRIC_ACCURACY_KEY]
        lo_list = self._clients_metrics[METRIC_LOSS_KEY]
        met_acc_history = np.zeros((len(clients)))
        met_lo_history = np.zeros((len(clients)))
        for seq, c in enumerate(clients):
            cl_id = clients2id[c.id]
            modl = self.shared_nn
            modl.set_weights(self.batch_weights[cl_id])
            ds = self.dataclass.create_dataset(c.train_data)
            history =  modl.fit(ds,
                        epochs= num_epochs,
                        verbose = 0)
                        

            met_acc_history[seq] = history.history["accuracy"][-1]
            met_lo_history[seq] = history.history["loss"][-1]
            
            client_net = modl.get_weights()
            self.init_client_vector(c, client_net)
#             params = [x * c.num_train_samples for x in client_net]
            avg_w[c.id] = client_net
        
        # iii end training, we set all historys
        acc_list.append(met_acc_history)
        self._clients_metrics[METRIC_ACCURACY_KEY] = acc_list
        lo_list.append(met_lo_history)
        self._clients_metrics[METRIC_LOSS_KEY] = lo_list
        
        return clients2id, id2clients, avg_w

    def train_model_with_ma(self, num_epochs=1, clients=None, agg_mode=None):
        def simple_update(weights, num, n_layer):
            w_agg = []
            num_params = n_layer
            num_worker = len(weights)
            for i in range(num_params):
                arr = [weights[j][i] for j in range(num_worker) ]
                m = np.mean(arr, axis=0)
                w_agg.append(m)
            return w_agg
        
        if clients == None:
            clients = self.selected_clients
                    
        # i) first find which cluster fits client most
        if not self.firstCommunicationRound:
            reverse_gw = reverse_matched_2_original_weights(
                self.batch_weights, self.num_clusters, self.shared_nn)
        else:
            reverse_gw = self.batch_weights
            
        id2clients, clients2id = self.assignments(reverse_gw, clients)
        
        # ii then we use dict to train them with only
        # inner_model trainable = true
        avg_cnn_w = {cl_idx: list() for cl_idx in id2clients}
        inner_modl, den_modl, out_modl = self.dataclass.create_CNN_ma_model()
        inner_modl.trainable = True
        den_modl.trainable = False
#         self.dataclass.set_model_trainable_compile(out_modl)
        for seq, c in enumerate(clients):
            cl_idx = clients2id[c.id]
            out_modl.set_weights(reverse_gw[cl_idx])
            ds = self.dataclass.create_dataset(c.train_data)
            out_modl.fit(ds, 
                         epochs = num_epochs,
                         verbose = 0)
            
            cnn_net = inner_modl.get_weights()
            avg_cnn_w[cl_idx].append(cnn_net)
        
        # we update the cnn with the aggreagte w
        # means cnn_w contains only one modl weight 
        # each cluster
        cnn_w = {cl_idx: simple_update(
            avg_cnn_w[cl_idx], 
            len(avg_cnn_w[cl_idx]), len(inner_modl.get_weights())) for cl_idx in avg_cnn_w}
        
        inner_modl.trainable = False
        den_modl.trainable = True
#         self.dataclass.set_model_trainable_compile(out_modl)
        avg_den_w = {cl_idx: list() for cl_idx in id2clients}
        for seq, c in enumerate(clients):
            cl_idx = clients2id[c.id]
            out_modl.set_weights(reverse_gw[cl_idx])
            #one more set cnn with weights calculated above
            inner_modl.set_weights(cnn_w[cl_idx])
            ds = self.dataclass.create_dataset(c.train_data)
            history = out_modl.fit(ds,
                         epochs = num_epochs,
                         verbose = 0)
            
            avg_den_w[cl_idx].append(den_modl.get_weights())
            client_net = out_modl.get_weights()
            logger.info("shape of client net {}".format(get_all_L_next(client_net)))
            self.init_client_vector(c, client_net)
        
        # update dense model weights
        den_w = {cl_idx: self.get_matched_avg(avg_den_w[cl_idx]) for cl_idx in avg_den_w}
        
        # append dense model weights to cnn weights
        # use list obj's extend method
        for cl_idx in cnn_w:
#             print("cnn_w len: ", len(cnn_w[cl_idx]))
#             print("den_w len: ", len(den_w[cl_idx]))
            cnn_w[cl_idx].extend(den_w[cl_idx])
#             print("after concat: ", len(cnn_w[cl_idx]))
            
        # finally update our server storage batch_weights
        for cl_idx in cnn_w:
            self.batch_weights[cl_idx] = cnn_w[cl_idx]
        
        # compared to normal train, we don't return weights
        return clients2id, id2clients       

    def get_matched_avg(self, clus_batch_weights):
        gamma = 6.0
        sigma = 1.0
        sigma0 = 1.0
        n_classes = 10
        it=4   

        glob_out, assignment = layer_group_descent(
            clus_batch_weights,
            None,
            sigma_layers = sigma,
            sigma0_layers = sigma0,
            gamma_layers = gamma,
            it = it,
            assignments_old = None
        )
        
        glob_weight = agg_layers(clus_batch_weights, glob_out, assignment)
        
        return glob_weight
    
    def init_client_vector(self, c, client_net):
        """this sets client's sem vector which be used by
        EM algorithm later stage.
        use last 4 layer (dense_weight1, bias1, dense_weight2, bias2)
        as the sem_vector
        """
        c.sem_vec = get_flatten_vec(client_net, self.fc_idx)
        
    def update_kmeans_estimate(self, g_w, clients):
        batch_szie = 20
        n_init = 20
        if self.firstCommunicationRound:
            init = 'k-means++'
        else:
            d = get_flatten_vec(self.shared_nn.get_weights(), self.fc_idx)
            init = np.array( [np.zeros(d.shape)] * self.num_clusters)
            for j in range(self.num_clusters):
                init[j] = get_flatten_vec(g_w[j], self.fc_idx)
            
        mbk = MiniBatchKMeans(init=init, n_clusters=self.num_clusters, 
                              batch_size=batch_szie,
                              n_init=n_init, max_no_improvement=10, 
                              random_state=0)
        mbk.partial_fit(self._sem_dataset)
        self._sem_dataset = []

        return mbk.labels_
                            
    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.
        Returns info about self.selected_clients if clients=None;
        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples
    
    def save_model(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        self.client_model.set_params(self.model)
        return self.client_model.save_file(path)

    def close_model(self):
        self.client_model.close()