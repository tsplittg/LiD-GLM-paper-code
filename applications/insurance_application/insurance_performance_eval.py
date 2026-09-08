import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import date
import itertools
import os
import ray
from ray import train, tune
from ray.tune import Tuner
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.neighbors import KNeighborsClassifier

from lidglm import ExtendedGLM, Transformer, ExtendedGLM_Utils, TransformerUtils
from lidglm.application_utilities.metric_functions import LidglmMetrics
from lidglm.application_utilities.training_functions import train_models, train_lidglm, train_nn, train_custom_sddr
from lidglm.application_utilities.initialization_functions import trad_glm_create_params, basic_nn_create_params, lidglm_create_params, elastic_net_create_params, sddr_create_params, custom_sddr_create_params
from lidglm.application_utilities.performance_analysis import ray_hyperparam_opt, k_fold_performance_analysis_lidglm, k_fold_performance_analysis_smglm, k_fold_performance_analysis_custom_sddr
#from lidglm.application_utilities.sddr_performance_analysis import k_fold_performance_analysis_sddr
from pathlib import Path
ray.shutdown()

test_name = "k_fold_performance_analysis_insurance_normalized"
temp_dir = os.path.join(Path.home(), "Ray_res", date.today().strftime("%m-%d"))
ray.init(_temp_dir = temp_dir)




##################import and preprocess dataset:#####################

torch.set_num_threads(1)
#Dataset loading and preprocessing:

# fetch dataset 
insurance_data = pd.read_csv("../Datasets/insurance_data/insurance.csv")

# the sex and smoker variables are strings but should be binary
# sex is coded as 'female', 'male' and smoker as 'yes', 'no'
# We'll recode female=1, male=0 and yes=1, no=0

insurance_data["sex"] = insurance_data["sex"].replace({"female":1, "male":0}).astype(int)
insurance_data["smoker"] = insurance_data["smoker"].replace({"yes":1, "no":0}).astype(int)

# region is a categorical variable with categories 'southwest', 'southeast', 'northwest', 'northeast'
# we'll do a dummy encoding and delete the reference
insurance_data = pd.get_dummies(insurance_data, columns=["region"], drop_first=True, dtype=int)

#finally, we pop the target column
target = insurance_data.pop("charges")
target = np.array(target)
covariates = np.array(insurance_data)

################Prepare the plotting functions used later:################################
"""
def plot_compare_to_classic_glm(extended_glm, exog, endog, save_path, **kwargs):
    traditional_glm = sm.GLM(endog=endog.detach().numpy(), exog=exog.detach().numpy(), family=sm.families.Binomial(sm.families.links.Logit()) )
    traditional_results=traditional_glm.fit(maxiter=10**5)
    util = ExtendedGLM_Utils()
    util.ALE_matrix(extended_glm=extended_glm, exog=exog, save_path=save_path)
"""


metrics = {"mse": LidglmMetrics.mse, 
                                    "negative log_like":LidglmMetrics.negative_loglike,
                                    "dist_trad_beta": LidglmMetrics.mean_difference_to_base_glm}

ldglm_parameter_space = lidglm_create_params(families = ["gaussian"], links = ["identity"],
                                                                       #lipschitz_consts = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.99, 2.1, 3, 5, 7, 9],
                                                                       lipschitz_consts = [1.01] +list(np.linspace(1.1, 2, 19)),
                                                                       norms = [1, 2],
                                                                       lrs= [1e-3, 1e-5], epochs = [3000], loss_crits =["log_like"],
                                                                       depths=[1,2,3], num_blocks=[1,2],
                                                                       widths=[9, 18, 36],
                                                                       parts_to_train=["network", "full"], 
                                                                       activations =["relu", "group_sort"], 
                                                                       nums_batches = [1])
ldglm_with_nu_d_parameter_space = lidglm_create_params(families = ["gaussian"], links = ["identity"],
                                                                       #lipschitz_consts = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.99, 2.1, 3, 5, 7, 9],
                                                                       lipschitz_consts = [1.01] +list(np.linspace(1.5, 2, 5)),
                                                                       norms = [2],
                                                                       lrs= [1e-3], epochs = [3000], loss_crits =["log_like"],
                                                                       depths=[2,3], num_blocks=[1],
                                                                       widths=[9, 18],
                                                                       parts_to_train=["full", "nu_p_then_nu_d"], 
                                                                       activations =["relu"], 
                                                                       nums_batches = [1],
                                                                       use_nu_d=[True, False],
                                                                       lipschitz_consts_2 = [1.4, 1.99, 3.99],
                                                                       widths_2 = [6, 16],
                                                                       depths_2 = [2,3], 
                                                                       num_blocks_2 = [1, 2],
                                                                       activations_2 = ["relu", "swish"])

sm_glm_param_space = trad_glm_create_params(families = ["gaussian"], links = ["identity"])

elastic_net_param_space = elastic_net_create_params(families = ["gaussian"], links = ["identity"], alphas = list(np.linspace(0, 10, 40)))


custom_sddr_param_space = custom_sddr_create_params(epochs = [3000], depths =[2,3,4],
                                                                       widths =[9, 18, 36],
                                                                          lrs= [1e-1, 1e-2, 1e-3, 1e-4], widths_2 =[9, 18, 36],  depths_2 =[2, 3, 4],
                                                                          dropout_rates=[0., 0.1, 0.2], num_batches=[1, 10])




traditional_metrics_dataframe = k_fold_performance_analysis_smglm(covariates, target, sm_glm_param_space, metrics, temp_dir,  k=5, test_name =test_name + "_glm", number_of_trials_per_split=128, mode = "continuous")
elastic_net_metrics_dataframe = k_fold_performance_analysis_smglm(covariates, target, elastic_net_param_space, metrics, temp_dir,  k=5, test_name=test_name + "_elastic", number_of_trials_per_split=128, mode = "continuous", regularization="elastic_net")
torch.save(traditional_metrics_dataframe, os.path.join(temp_dir, test_name+date.today().strftime("%m-%d")+"_traditional_metrics_dataframe.pt"))
torch.save(elastic_net_metrics_dataframe, os.path.join(temp_dir, test_name+date.today().strftime("%m-%d")+"_elastic_net_metrics_dataframe.pt"))


lidglm_with_nu_d_metrics_dataframe = k_fold_performance_analysis_lidglm(covariates, target, ldglm_with_nu_d_parameter_space, metrics, temp_dir,  k=5, test_name=test_name+"_ldglm_with_nu_d", number_of_trials_per_split=256, mode = "continuous")
torch.save(lidglm_with_nu_d_metrics_dataframe, os.path.join(temp_dir, test_name+date.today().strftime("%m-%d")+"_lidglm_with_nu_d_metrics_dataframe.pt"))
custom_sddr_metrics_dataframe = k_fold_performance_analysis_custom_sddr(covariates, target, custom_sddr_param_space, metrics, temp_dir,  k=5, test_name=test_name + "_custom_sddr", number_of_trials_per_split=256, mode = "continuous", normalize = True)
torch.save(custom_sddr_metrics_dataframe, os.path.join(temp_dir, test_name+date.today().strftime("%m-%d")+"_custom_sddr_metrics_dataframe.pt"))



lidglm_metrics_dataframe = k_fold_performance_analysis_lidglm(covariates, target, ldglm_parameter_space, metrics, temp_dir,  k=5, test_name=test_name+"_ldglm", number_of_trials_per_split=256, mode = "continuous")
torch.save(lidglm_metrics_dataframe, os.path.join(temp_dir, test_name+date.today().strftime("%m-%d")+"_lidglm_metrics_dataframe.pt"))




print("temp dir is: ", temp_dir)