
import numpy as np

def build_network(FC_parameters):
    weight_list = []
    bias_list = []
    for i in range(len(FC_parameters)-1):
        temp_weight= np.random.rand(FC_parameters[i],FC_parameters[i+1])
        temp_bias = np.random.rand(1,FC_parameters[i+1])
        weight_list.append(temp_weight)
        bias_list.append(temp_bias)
    
    return  (weight_list,bias_list)  

def get_result(input_data,FC_list):
    weight_list = FC_list[0]
    bias_list = FC_list[1]
    temp = input_data
    for i in range(len(weight_list)):
        temp = temp.dot(weight_list[i])+bias_list[i]
    return temp

def random_change_sequence(FC_list):
    weight_list = FC_list[0]
    bias_list = FC_list[1]
    
    weight_list_new = weight_list
    for i in range(len(weight_list)-1):
        # get two layers
        temp_weight = weight_list_new[i]
        temp_weight_next_layer = weight_list_new[i+1]
        
        #get random orders for shuffle two layers
        out_put_dimension = temp_weight.shape[-1]
        new_orders = np.random.choice(out_put_dimension, out_put_dimension,replace=False)
        
        #shuffle two layers with order
        temp_weight_new = temp_weight[:,new_orders]
        temp_weight_next_layer_new = temp_weight_next_layer[new_orders,:]
        
        #shuffle two layers with order
        weight_list_new[i] = temp_weight_new
        weight_list_new[i+1] = temp_weight_next_layer_new
        
    return (weight_list_new,bias_list)  


    

input_data = np.random.rand(1,3)
FC_parameters = [3,4,2]

FC_list = build_network(FC_parameters)


print('input_data = ',input_data)
print('layers of FC:')
for i,content in enumerate(FC_list[0]):
    print(i,'th layer of FC: ')
    print('\t weight = \n', content,'\n\t bias = ',FC_list[1][i])
print('FC_old(input) = ', get_result(input_data,FC_list))
print('\n\n\n')


print('input_data = ',input_data)
print('layers of FC:')
FC_list_new = random_change_sequence(FC_list)
for i,content in enumerate(FC_list_new[0]):
    print(i,'th layer of FC: ')
    print('weight = \n', content,'\n bias = ',FC_list_new[1][i])
print('FC_old(input) = ', get_result(input_data,FC_list_new))
print('\n\n\n')