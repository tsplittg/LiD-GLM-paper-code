import torch
import numpy as np
import pandas as pd
from datetime import date
import os
import ray


from lidglm import ExtendedGLM, Transformer, ExtendedGLM_Utils, TransformerUtils
from lidglm.application_utilities.metric_functions import LidglmMetrics
from lidglm.application_utilities.training_functions import train_models, train_lidglm, train_nn
from lidglm.application_utilities.initialization_functions import trad_glm_create_params, basic_nn_create_params, lidglm_create_params, elastic_net_create_params, sddr_create_params
from lidglm.application_utilities.performance_analysis import ray_hyperparam_opt, k_fold_performance_analysis_lidglm, k_fold_performance_analysis_smglm, k_fold_performance_analysis_nn
from lidglm.application_utilities.sddr_performance_analysis import k_fold_performance_analysis_sddr
from pathlib import Path
ray.shutdown()


test_name = "k_fold_performance_analysis_stroke"
temp_dir = os.path.join(Path.home(), "Ray_res", date.today().strftime("%m-%d")+ "_stroke")
ray.init(_temp_dir = temp_dir)




##################import and preprocess dataset:#####################
stroke = pd.read_csv("../Datasets/Diabetes, Hypertension and Stroke Prediction (open)/stroke_data.csv", sep=",")

#3 entries contain NA values for "sex", these are a negligible fraction of the dataset, so we will drop them
stroke = stroke.dropna()

#The columns "age", "bmi", "avg_glucose_level" are non binary

#Does the table contain negative values for age?
#Yes. We assume these values code missingness and will drop the corresponding rows:
stroke = stroke[stroke["age"]>=0]

# The work_type covariate seems dubious. E.g., there exist subjects coded as children with an age far beyond 21. We will therefore dropo this column:
#stroke = pd.get_dummies(stroke, columns=["work_type"], dtype= float, drop_first=True)
stroke = stroke.drop(["work_type"], axis = 1)


target = np.array(stroke["stroke"])
covariates = stroke.drop(["stroke"], axis=1)


metrics = {"accuracy": LidglmMetrics.accuracy, "recall": LidglmMetrics.recall, "precision": LidglmMetrics.precision,
                                    "F1 Score": LidglmMetrics.f1_score, "negative log_like": LidglmMetrics.negative_loglike,
                                    "Roc_Auc": LidglmMetrics.roc_auc}


ldglm_param_space = lidglm_create_params(families = ["binomial"], links = ["logit"],
                                                                       lipschitz_consts = list(np.linspace(1.1, 16, 15)),
                                                                       norms = [1, 2],
                                                                       lrs= [1e-3, 1e-4, 1e-5], epochs = [3000], loss_crits =["log_like"],
                                                                       depths=[3, 4, 5, 6], num_blocks=[3, 4, 5],
                                                                       widths=[18, 36, 48],
                                                                       parts_to_train=["full", "network"], 
                                                                       activations =["group_sort", "relu"], 
                                                                       nums_batches = [10])

sm_glm_param_space = trad_glm_create_params(families = ["binomial"], links = ["logit"])

elastic_net_param_space = elastic_net_create_params(families = ["binomial"], links = ["logit"], alphas = list(np.linspace(0, 10, 40)))
nn_param_space = basic_nn_create_params(lrs= [1e-3, 1e-4, 1e-5], epochs = [5000], widths=[8,16, 32, 64, 100], depths=[2,3, 4, 5, 6], num_batches=[1, 10])


sddr_param_space = sddr_create_params(families = ["Bernoulli"],epochs = [3000], 
                                                                       depths=[3, 4, 5, 6],
                                                                       widths=[36, 64,100], dfs=[0], output_shape=[1],
                                                                       batch_size= [3000], optimizers = ["adam"], lrs= [1e-2, 1e-3, 1e-4], weight_decays=[0., 0.01],
                                                                       dropout_rates=[0., 0.1], early_stop_epochs=[200])
##################Run the performance analysis:#####################

traditional_metrics_dataframe = k_fold_performance_analysis_smglm(covariates, target, sm_glm_param_space, metrics, temp_dir,
                                                                  k=5, test_name =test_name + "_glm", number_of_trials_per_split=128, mode = "binary")
elastic_net_metrics_dataframe = k_fold_performance_analysis_smglm(covariates, target, elastic_net_param_space, metrics, temp_dir,
                                                                  k=5, test_name=test_name + "_elastic_aki", number_of_trials_per_split=128, mode = "binary", regularization="elastic_net")
torch.save(traditional_metrics_dataframe, os.path.join(temp_dir, test_name+date.today().strftime("%m-%d")+"_traditional_metrics_dataframe.pt"))
torch.save(elastic_net_metrics_dataframe, os.path.join(temp_dir, test_name+date.today().strftime("%m-%d")+"_elastic_net_metrics_dataframe.pt"))




#lidglm_metrics_dataframe = k_fold_performance_analysis_lidglm(covariates, target, ldglm_param_space, metrics, temp_dir,  k=5,
#                                                              test_name= test_name +"_ldglm", number_of_trials_per_split=260, mode = "binary")
#torch.save(lidglm_metrics_dataframe, os.path.join(temp_dir, test_name+date.today().strftime("%m-%d")+"_lidglm_metrics_dataframe.pt"))

sddr_metrics_dataframe = k_fold_performance_analysis_sddr(covariates, target, sddr_param_space, metrics, temp_dir,  k=5, test_name=test_name + "_sddr",
                                                          number_of_trials_per_split=260, mode = "binary")
torch.save(sddr_metrics_dataframe, os.path.join(temp_dir, test_name+date.today().strftime("%m-%d")+"_sddr_metrics_dataframe.pt"))

#nn_metrics_dataframe = k_fold_performance_analysis_nn(covariates, target, nn_param_space, metrics, temp_dir,  k=5, test_name=test_name + "_nn", number_of_trials_per_split=260, mode = "binary")
#torch.save(nn_metrics_dataframe, os.path.join(temp_dir, test_name+date.today().strftime("%m-%d")+"_nn_metrics_dataframe.pt"))


print("temp dir is: ", temp_dir)