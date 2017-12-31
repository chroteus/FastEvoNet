net_dir = "nets"
import os
from . import helpers

if not os.path.exists(helpers.rel_path(net_dir)):
    try:
        os.mkdir(helpers.rel_path(net_dir))
    except OSError:
        pass

##################
### Activators ###
import numpy as np
import math
activator_id = "tanh" # tanh is preferred as we can use numpy's own efficent function

def activator(val):
    return activators[activator_id](val)

activators = {}
activators["tanh"] = np.tanh

def sigmoid(val): return 1/(1+math.exp(-val))
activators["sigmoid"] = np.frompyfunc(sigmoid, 1,1)

def relu(val): return max(0,val)
activators["relu"] = np.frompyfunc(relu, 1,1)


##########
import random
def new_weight():
    if activator_id == "tanh":
        return random.uniform(-1,1)
    elif activator_id == "sigmoid":
        return random.random()
