DATASETS = ['mnist', 'femnist', 'celeba', 'cifar10']
exp_dataset = DATASETS[2]

mod = importlib.import_module(exp_dataset)
ClientModel = getattr(mod, "ClientModel")

attributes = DATASET_ATTRIBUTES[exp_dataset]
SEED = 4151971
input_shape = attributes['input_shape']
dimension = attributes['dimension']
'''
move this out if you are using tf 2
 
'''
tf.random.set_seed(SEED) 
np.random.seed(SEED)

eval_counter = 0
def get_global_data(set_to_use='test'):
    global eval_counter
    num_stacking = round(len(clients) /4)  
    id_start =  0 if eval_counter == 0 else num_stacking * eval_counter
    id_end = num_stacking * (eval_counter + 1)
    eval_counter += 1
    if eval_counter >= 4:
        eval_counter = 0 
    stack_list = [c.id for c in clients[id_start: id_end]] 
         
    for i in range(num_stacking):
        if i == 0:
            datax = test_data[stack_list[0]]['x']
            datay = test_data[stack_list[0]]['y']
        else:
            x = test_data[stack_list[i]]['x']
            datax = np.concatenate((datax, x), axis=0)
            datay = np.concatenate((datay, test_data[stack_list[i]]['y']), axis=0)
    
    return {'x': datax, 'y': datay}

def restore_set():
    with open("./glob_testset_femnist", "rb") as f:
        datax, datay = pickle.load(f)
        
    dataset = tf.data.Dataset.from_tensor_slices((datax, datay))
    global_data = dataset.batch(32)
    
    return global_data

def restore_mnist_test():
    dataset = tf.data.Dataset.from_tensor_slices(test_data)
    dataset = dataset.batch(32)
    return dataset

def get_history(d):
    get_mean = lambda x: np.mean(x)
    acc = [get_mean(i) for i in d['accuracy']]
    return np.array(acc)
    
def get_loss(d):
    get_mean = lambda x: np.mean(x)
    lo = [get_mean(i) for i in d['loss']] 
    return np.array(lo)

def cust_evaluate(batch_w):
    first_key = [k for k in batch_w][0]
    L_n = get_all_L_next(batch_w[first_key])
    weight = server.batch_weights[0]
    nn = current_model.create_CNNmodel(L_n)
    nn.build(ins_c_model.get_input_shape)
    nn.set_weights(weight)
    history = nn.evaluate(test_set, verbose=1)
    
def IMCK(server, local_epochs=10, sem=False, bbp_map=False):
    tf.keras.backend.clear_session()
    server.select_clients(clients, num_worker_per_round)
    if sem:
        server.init_sem_data(dimension)
        print("before sem running, data x looks like {}".format(np.array(server._sem_dataset).shape ))
    clients2key, key2clients, avg_w = server.train_model(local_epochs)
    server.update_weights(avg_w, key2clients, clients2key)
    server.evaluate_global_models(test_set)    
    for key in key2clients:
        logger.info("cluster_{} assigned {} clients".format(key, len(key2clients[key])))    
    return clients2key, key2clients, avg_w    


def IMCK_MA(server, local_epochs=10, sem=False):
    tf.keras.backend.clear_session()
    server.select_clients(clients, num_worker_per_round)
    if sem:
        server.init_sem_data(dimension)    
    clients2key, key2clients = server.train_model_with_ma(local_epochs)
    for key in key2clients:
        logger.info("cluster_{} assigned {} clients".format(key, len(key2clients[key])))       



# Prepare data for training

mod = setup_clients
_setup_func = getattr(mod, 'setup_clients_{}'.format(exp_dataset))

# lr = attributes['lr']
lr = 0.01
avg_batch_size = 10

op = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, decay=1e-6)
current_model = ClientModel(SEED, lr, train_bs=avg_batch_size, optimizer = op, input_shape=input_shape)
if exp_dataset in ['femnist', 'celeba']:
    clients, train_data, test_data = _setup_func(current_model)
    test_set = current_model.create_dataset(get_global_data(), 'test') 
else:
    clients, train_data, test_data = _setup_func(100, current_model)
    test_set = restore_mnist_test()
    
iterations = 10
# num_workers = len(clients)
num_worker_per_round = 15
# num_clusters = 1
# local_epochs = 5

num_clusters = 6
local_epochs = 4
current_model.SGD = False
logging.info("=="*15)
logger.info("Start training on <{}>".format(exp_dataset))
server = Server(current_model, num_clusters = num_clusters)

for t in trange(iterations):
    clients2key, key2clients, avg_w = IMCK(server, local_epochs, sem=True)
    server.firstCommunicationRound = False
