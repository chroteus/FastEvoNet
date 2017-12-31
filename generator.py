import random, math
import numpy as np
from . import evo_globals

"""
    Generate weights and connections data for Network.
"""
def new_weight():
    weight = random.random()
    weight *= random.choice((-1,1))
    return weight

def new_weights(rows,cols, mode=None):
    if not mode:
        mode = evo_globals.activator_id

    if mode == "tanh":
        return 2 * np.random.random_sample((rows,cols)) - 1 # return [-1,1]
    elif mode == "sigmoid":
        return np.random.random((rows,cols)) # return [0,1]

# small reminder:
# number of rows in weight == number of neurons in next layer
# number of cols in weight == number of neurons in previous layer
def network(input_no,output_no,min_n=10,max_n=500):
    neuron_number = min_n + random.randint(0, max_n-min_n) # total intermediate neurons
    # note that this is not the exact value, but more like an approximation

    weights = []
    first_layer_num = random.randint(0, round(neuron_number/5))
    first_layer_num = max(first_layer_num,input_no)

    weights.append(new_weights(first_layer_num,input_no))
    neuron_number -= first_layer_num

    layer_number = max(2, random.randint(0,round(neuron_number/50)))
    neurons_per_layer = round(neuron_number/layer_number)

    ci = 1
    while neuron_number > 0:
        new_layer_num = random.randint(round(neurons_per_layer/(ci+random.random())), \
                                       neurons_per_layer)
        new_layer_num = max(2,new_layer_num)
        new_weight = new_weights(new_layer_num, weights[-1].shape[0])
        weights.append(new_weight)
        neuron_number -= new_layer_num
        ci += 1

    weights.append(new_weights(output_no, weights[-1].shape[0]))

    return weights
