"""
    Network class.
"""
import os, random, itertools, json
from . import helpers, evo_globals, generator
import numpy as np
import random

import math

class Network:
    def __init__(self,weights, input_no=1,output_no=1):
        self.input_no = input_no
        self.input_layer = np.ndarray(input_no)
        self.output_no = output_no
        self.output_layer = np.ndarray(self.output_no)
        self.weights = weights
        self.hidden_layers = []

        self.fitness = -999999


    def set_input(self, input_arr):
        self.input_layer = np.array(input_arr)

    def get_output(self, weights=None):
        weights = weights if weights else self.weights
        # v this v is output layer if no hidden layers present

        self.hidden_layers = []
        output = weights[0].dot(self.input_layer)
        output = evo_globals.activator(output)

        for i in range(1,len(self.weights)):
            output = weights[i].dot(output)
            output = evo_globals.activator(output)

        self.output_layer = output
        return self.output_layer


    def reproduce_with(self, mate, class_to_use=None):
        if not class_to_use: class_to_use = Network
        # nets to mate are named tiny and giant here to denote largest and smallest network respectively
        # this is done to prevent confusion with weights' individiual size
        tiny   = self if len(self.weights) <= len(mate.weights) else mate
        giant  = self if len(self.weights)  > len(mate.weights) else mate

        new_weights = np.copy(tiny.weights).tolist()
        for i,tinys_weight in enumerate(tiny.weights):
            giants_weight = giant.weights[i] # tiny always has less layers, so this won't fail

            # NOTE: tiny's weights at i are not necessarily smaller than giant's weights!
            smaller_weight = min(giants_weight,tinys_weight, key=lambda x:x.size)
            bigger_weight = giants_weight if smaller_weight is tinys_weight else tinys_weight

            rows_to_remove = bigger_weight.shape[0] - smaller_weight.shape[0]
            cols_to_remove = bigger_weight.shape[1] - smaller_weight.shape[1]
            # NOTE: Splicing is done separately to prevent indexing related bugs
            # np.resize is specifically not used as it's not the same as splicing
            if rows_to_remove > 0:
                bigger_weight = bigger_weight[:-rows_to_remove, :]
            if cols_to_remove > 0:
                bigger_weight = bigger_weight[:, :-cols_to_remove]

            # in rare cases, smaller size doesn't mean that bigger weight fully covers it
            # like in cases of 7x1 weights
            if rows_to_remove < 0:
                smaller_weight = smaller_weight[:rows_to_remove, :]
            if cols_to_remove < 0:
                smaller_weight = smaller_weight[:, :cols_to_remove]

            new_weights[i] = np.mean((bigger_weight,smaller_weight), axis=0)

        return class_to_use(new_weights, input_no=self.input_no,output_no=self.output_no)

    def mutate(self, mutation_rate=0.01):
        for li,layer in enumerate(self.weights):
            mutations = generator.new_weights(layer.shape[0],layer.shape[1], mode="tanh")
            mutations *= mutation_rate

            self.weights[li] += mutations

        return self

    def size_info(self):
        s = "Number of layers: " + str(len(self.hidden_layers)+2)
        s += "\n"
        min_len = str(len(min(self.hidden_layers, key=len)))
        max_len = str(len(max(self.hidden_layers, key=len)))
        avg_len = 0
        for x in self.hidden_layers: avg_len += len(x)
        avg_len /= len(self.hidden_layers)
        avg_len = str(round(avg_len,5))
        s += "Layer size (min/avg/max):" + min_len +"/"+ avg_len +"/"+ max_len

        return s
    """
        Condenses all weights into one string.
        __str__ of Neuron returns a 12*n digit number, every 12 digits representing
        a weight. Weights for each neuron are divided by newline.
    """
    def get_weight_str(self):
        weights = [x.tolist() for x in self.weights]
        return json.dumps(weights)


    """
        Non-random name based on weights data. Unique for each net.
    """
    def generate_name(self):
        max_char = 122
        min_char = 64
        name_len = 10
        diff = max_char - min_char
        n = name_len if len(self.weights) >= name_len else len(self.weights)
        name = ""
        for i in range(n):
            w = self.weights[i][0][0]
            w = evo_globals.activators["sigmoid"](w) #squish w
            dec = round(min_char + (diff*float(w)))
            name += chr(dec)
        while len(name) < name_len: name += name[-1]

        return name

    def save(self, path=None):
        path = path if path else helpers.rel_path(evo_globals.net_dir, self.generate_name())
        if os.path.exists(path): path += " - 1" # first copy
        while os.path.exists(path): # if path still exists somehow, go to next index
            split_path = path.split("-")
            index = int(split_path[1])
            path = split_path[0] + " - " + str(index+1)

        #now that we are sure our path is unique
        with open(path, "w+") as f:
            f.write(str(self))

    def load(self, path_or_net):
        try:
            with open(path_or_net) as f:
                return helpers.decode_net_str(f.read().rstrip(os.linesep))

        except FileNotFoundError:
            # no file, try decoding directly
            return helpers.decode_net_str(path_or_net)

    def __str__(self):
        return self.get_weight_str().rstrip(os.linesep)

    def reset_values(self):
        pass # dummy function for compatibility
