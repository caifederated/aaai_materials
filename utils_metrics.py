import numpy as np
import random
from sklearn.metrics import accuracy_score

def get_random_test_data(clients, num_worker):
    selected = random.choices(clients, k=num_worker)
    datax = [x_i for c in selected for x_i in c.eval_data['x']]
    datay = [y_i for c in selected for y_i in c.eval_data['y']]
    sample_weight = get_per_sample_weight(selected)
    return {"x": datax, "y": datay}, sample_weight

def get_per_sample_weight(selected_clients):
    num_samples = [c.num_test_samples for c in selected_clients]
    total_samples = np.sum(num_samples)
    per_client_w = np.array(num_samples) / total_samples
    per_sample = np.repeat(per_client_w, num_samples)
    return per_sample 

def eval_svr_accuracy_(svr, data, sample_weight, cent_idx=1):
    labels = [y for y in data['y'] ]
    data_test = svr.dataclass.create_dataset(data, 'pred')
    svr.c_model.set_weights(svr.batch_weights[cent_idx])
    loss, acc = svr.c_model.evaluate(data_test, verbose=0)
    return acc
#     y_pred = svr.c_model.predict_classes(data_test)
#     return (accuracy_score(labels, y_pred, sample_weight=sample_weight))