import warnings
import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import os
import math
import bisect
import tensorflow as tf
import random

from tensorflow.python.ops.check_ops import is_strictly_increasing


from cde.density_estimator import GPDExtremeValueMixtureDensityNetwork
from cde.density_estimator import MixtureDensityNetwork
from cde.density_estimator import ExtremeValueMixtureDensityNetwork

from cde.data_collector import MatlabDataset, MatlabDatasetH5, get_most_common_unique_states
from cde.density_estimator import plot_conditional_hist, measure_percentile, measure_percentile_allsame, measure_tail, measure_tail_allsame, init_tail_index_hill, estimate_tail_index_hill
from cde.evaluation.empirical_eval import evaluate_models_singlestate, empirical_measurer, evaluate_model_allstates, evaluate_models_allstates_plot, obtain_exp_value, evaluate_models_allstates_agg



class MatlabInterface():
    def __init__(self,n_epoch, n_centers, hidden_layer_n, ndim_x, address_name, is_emm, learning_rate_p1, learning_rate_p2):
        # hidden_layer_n = 16, n_centers = 3, ndim_x = 3, n_epoch = 7000, learning_rate_p1=1, learning_rate_p2=-4 (1e-4 big) or -5 (1e-5 small) 
        self.is_emm = is_emm
        n = random.randint(0,1000000)
        if(self.is_emm):
            self.model = GPDExtremeValueMixtureDensityNetwork("EMM-"+str(n), ndim_x=ndim_x, n_centers=n_centers, ndim_y=1, n_training_epochs=n_epoch, hidden_sizes=(hidden_layer_n, hidden_layer_n),verbose_step=math.floor(n_epoch/10), weight_decay=1e-4, learning_rate=float(learning_rate_p1*(10**learning_rate_p2)),epsilon=1e-6)
        else:
            self.model = MixtureDensityNetwork("GMM-"+str(n), ndim_x=ndim_x, n_centers=n_centers, ndim_y=1,n_training_epochs=n_epoch,hidden_sizes=(hidden_layer_n, hidden_layer_n))

        self.model._setup_inference_and_initialize()
        try:
            with open(address_name, 'rb') as input:
                self.model = pickle.load(input)
        except Exception:
            pass
            #raise RuntimeError(address_name)

    def fit_model(self, train_data):
        Y = train_data[:,0]
        X = train_data[:,1:]
        self.model.fit(X, Y)

    def save_model(self,address_name):
        with open(address_name, 'wb') as output:
            pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)

    def load_model(self,address_name):
        self.model._setup_inference_and_initialize()
        with open(address_name, 'rb') as input:
            self.model = pickle.load(input)
            self.model._setup_inference_and_initialize()

    def tail(self,mx,my):
        mx2 = np.array([mx])
        my2 = np.array([my])
        if self.is_emm:
            res = self.model.tail(mx2,my2)
            return res.item()
        else:
            res = self.model.tail(mx2,my2)
            return res.item()


def myinit(n_epoch, n_centers, hidden_layer_n, ndim_x, address_name, is_emm, learning_rate_p1, learning_rate_p2):
    #raise RuntimeError(address_name)
    global predictor
    global is_set
    try:
        is_set
    except Exception:
        predictor = MatlabInterface(n_epoch, n_centers, hidden_layer_n, ndim_x, address_name, is_emm, learning_rate_p1, learning_rate_p2)
        is_set = True

def mytail(mx,my):
    global predictor
    return predictor.tail(mx,my)
