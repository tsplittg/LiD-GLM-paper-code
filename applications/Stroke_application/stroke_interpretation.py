#import standard packages:
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tqdm import tqdm
import matplotlib.pyplot as plt

#import modules from our method:
from lidglm import ExtendedGLM, ExtendedGLM_Utils
from lidglm.extended_glm.extended_glm_eval_utls import Extended_glm_eval_utils
from lidglm import TransformerUtils
torch.set_num_threads(1)

output_path = "./outputs/"

#%% Load the dataset, and do first preprocessing:
stroke = pd.read_csv("../Datasets/Diabetes, Hypertension and Stroke Prediction (open)/stroke_data.csv", sep=",")
seed = 42

#3 entries contain NA values for "sex", these are a negligible fraction of the dataset, so we will drop them
stroke = stroke.dropna()

#The columns "age", "bmi", "avg_glucose_level" are non binary

#Does the table contain negative values for age?
#Yes. We assume these values code missingness and will drop the corresponding rows:
stroke = stroke[stroke["age"]>=0]

# The work_type covariate seems dubious. E.g., there exist subjects coded as children with an age far beyond 21. We will therefore dropo this column:
#stroke = pd.get_dummies(stroke, columns=["work_type"], dtype= float, drop_first=True)
stroke = stroke.drop(["work_type"], axis = 1)


target =stroke["stroke"]
covariates = stroke.drop(["stroke"], axis=1)

# Train/Validation split:
X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(covariates, target, test_size=0.2, random_state=seed)
# normalize the data:
non_binary_indices =  np.array([i for i in range(covariates.shape[1]) if len(np.unique(covariates.iloc[:, i])) > 2])
original_means = np.zeros(X_train_df.shape[1])
original_stds = np.ones(X_train_df.shape[1])

original_means[non_binary_indices] = X_train_df.iloc[:, non_binary_indices].mean(axis=0)
original_stds[non_binary_indices] = X_train_df.iloc[:, non_binary_indices].std(axis=0)    

X_train_df = (X_train_df - original_means)/original_stds
X_val_df = (X_val_df - original_means)/original_stds
X = (covariates- original_means)/original_stds
original_means_contin = pd.Series(original_means, index = X.columns)
original_stds_contin = pd.Series(original_stds, index = X.columns)
# convert data to numpy arrays, which are needed for some models
X_train = X_train_df.to_numpy()
X_val = X_val_df.to_numpy()
y_train = y_train_df.to_numpy().reshape(-1)
y_val = y_val_df.to_numpy().reshape(-1)


#%% First Model: traditional (Generalized) Linear Model:

sm_glm = sm.GLM(endog=y_train, exog=sm.add_constant(X_train, prepend = True, has_constant="add"), family=sm.families.Binomial(sm.families.links.Logit()))
sm_results = sm_glm.fit(maxiter=10**4, disp = True)

print(sm_results.summary())

#%% Second model: LiD-GLM:

#our method:
#first, compute number of covariates (important for shape of i-ResNet)
num_params = X_train.shape[1]

# Lipschitz constant of the \emph{entire} residual construct, evently distributed among the blocks. Each block has a Lipschitz constant in (1,2)
# it holds that (block_Lip_const)^{num_blocks} = Lip_const, meaning num_blocks = log(Lip_const)/log(block_Lip_const)
total_lipschitz_const = 14.99
norm_used = 2

#initialize the transformer (i-ResNet) for the covariates:
transformer = TransformerUtils().initialize_transformer(num_params= num_params, lipschitz_const=total_lipschitz_const, width=48, depth=6, num_blocks=5,
                                                        domain_codomain=norm_used, activation_string="group_sort", group_size=2)

#convert the traditional GLM into a deep GLM by converting it to TorchGLM and prepending the transformer:
util = ExtendedGLM_Utils()
extended_glm, target_tensor, X_tensor = util.convert_sm_glm(glm=sm_results, net_transform=transformer, bias = "first")
extended_glm.read_linear_weights()
extended_glm = extended_glm.to("cpu")
target_tensor = target_tensor.to("cpu")
X_tensor = X_tensor.to("cpu")

epochs = 10000
train_model=False
path = f"{output_path}stroke_extended_glm_trained" #paper version

if train_model:
    #train the model:
    loss_hist, test_error_hist = util.train_model(extended_glm = extended_glm, exog=X_tensor, endog=target_tensor,epochs= epochs, part_to_train="network", lr=1e-3, loss_crit="log_like", num_batches=10, 
                                                X_test=torch.tensor(X_val, dtype=torch.float32, device="cpu"), y_test= torch.tensor(y_val, dtype=torch.float32, device="cpu"), 
                                                early_stopping=True, patience = 10)
    extended_glm.save_model(path)
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    ax[0].plot(loss_hist)
    ax[0].set_title("Training Loss History")
    ax[0].set_xlabel("Iteration")
    ax[1].plot(test_error_hist)
    ax[1].set_title("Validation Loss History")
    ax[1].set_xlabel("Iteration")
else:
    extended_glm = util.load_from_param_dict_and_state(torch.load(f"{path}_params.pth"), torch.load(f"{path}_state_dict.pth"))

extended_glm.eval()


#%% LiD-GLM evaluation:

lidglm_val_pred = extended_glm.predict(torch.tensor(X_val, dtype=torch.float32, device=extended_glm.used_device)).detach().cpu().numpy().reshape(-1)
lidglm_val_loglike = extended_glm.log_like_obs(endog=torch.tensor(y_val, dtype=torch.float32, device=extended_glm.used_device), exog=torch.tensor(X_val, dtype=torch.float32, device=extended_glm.used_device)).detach().cpu().numpy().reshape(-1)
lidglm_val_acc = np.sum(np.round(lidglm_val_pred) == y_val)/len(y_val)
lidglm_val_binary_ce = metrics.log_loss(y_val, lidglm_val_pred)
lidglm_brier_score = metrics.brier_score_loss(y_val, lidglm_val_pred)
print("Validation Accuracy LDGLM: ", lidglm_val_acc)
print("Validation Binary Cross Entropy LDGLM: ", lidglm_val_binary_ce)
print("Validation Brier Score LDGLM: ", lidglm_brier_score)
print("validation loglike of the LDGLM: ", lidglm_val_loglike.mean())

#%% compute coeff table:
eval_utils = Extended_glm_eval_utils()
coeff_table = eval_utils.coefficient_table(extended_glm=extended_glm, exog=X_tensor, coefficient_names=["intercept"] + X_train_df.columns.tolist(), original_glm = sm_results)
coeff_table.to_csv(f"{output_path}stroke_extended_glm_coefficient_table.csv")

#%% define some helper functions for PDP plots over logit values:

class sklearn_logits_pdp_adapter(BaseEstimator):

    def __init__(self, extended_glm, pdp_type="prediction", **kwargs):
        """
        Parameters
        ----------
        extended_glm : ExtendedGLM
            The extended GLM to use for predictions.
        pdp_type : str, optional
            The type of PDP to compute. The default is "prediction".
        """
        self.extended_glm = extended_glm
        self.pdp_type = pdp_type
        self.linear_weights_ = extended_glm.read_linear_weights()
        self._estimator_type = "regressor"
        self.classes_ = np.array([0,1])
        self.fit()
        self.target_variable = kwargs.get("target_variable", None)
    
    def predict(self, X):
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32, device=self.extended_glm.used_device)
        if self.pdp_type == "prediction":
            return self.extended_glm.linear_predictor(X_tensor).reshape(-1).detach().cpu().numpy()
        elif self.pdp_type == "T_p":
            if self.target_variable is None:
                raise ValueError("target_variable must be specified for T_p PDP")
            return self.extended_glm.NN_transform(X_tensor)[:, self.target_variable].reshape(-1).detach().cpu().numpy()
            
    
    def fit(self, X=None, y=None):
        self.is_fitted_ = True
        return self
    

class sklearn_logits_statsmodels_pdp_adapter(BaseEstimator):
    """Adapter class to use the traditional GLM with sklearn's PDP functions.
    This is needed because sklearn's PDP functions expect a sklearn-like estimator.
    """
    def __init__(self, sm_glm :sm.GLM):
        """
        Parameters
        ----------
        sm_glm : sm.GLM
            The statsmodels GLM to use for predictions.
        pdp_type : str, optional
            The type of PDP to compute. The default is "prediction".
        """
        self.sm_glm = sm_glm
        self._estimator_type = "regressor"
        self.fit()

    def predict(self, X):
        X = np.array(X)
        # we return the logits (linear predictor)
        return self.sm_glm.predict(X, which="linear").reshape(-1)
    
    def fit(self, X=None, y=None):
        self.is_fitted_ = True
        return self
    
class linear_function_pdp_adapter(BaseEstimator):
    """Adapter class to compute PDPs of the linear function of the extended GLM, using sklearn's PDP functions.
    This is needed because sklearn's PDP functions expect a sklearn-like estimator.
    """
    def __init__(self, linear_weights):
        self.linear_weights = linear_weights
        self._estimator_type = "regressor"
        self.fit()

    def predict(self, X):
        X = np.array(X)
        return X @ self.linear_weights.reshape(-1)
    
    def fit(self, X=None, y=None):
        self.is_fitted_ = True
        return self
    




#%% compute PDPs:

# we do 3 figures in one, showing the PDPs for age, avg_glucose_level and bmi
fig, axs = plt.subplots(1,3, figsize=(18, 6), sharey=True)


# to be ablt to compare to the PHO linear weights, we first compute those:
pho_lin_weights = eval_utils.custom_PHO(extended_glm=extended_glm, exog=X_tensor)[0]


#First, we do a PDP for "age"
age_mean = original_means[stroke.columns.get_loc("age")]
age_std = original_stds[stroke.columns.get_loc("age")]
partial_dependence_age = partial_dependence(sklearn_logits_pdp_adapter(extended_glm=extended_glm, pdp_type="prediction"), X, features=["age"], grid_resolution=100, method="brute", kind ="both", percentiles = (0,1))
axs[0].plot(partial_dependence_age["grid_values"][0]*age_std+age_mean, 
        partial_dependence_age["average"][0], color="blue", linewidth=3, label="LiD-GLM PDP")
axs[0].set_xlabel("Age", fontsize = 16)
axs[0].set_ylabel("Partial dependence", fontsize = 16)


#we also plot the linear function of the extended GLM for age, to see how much the nonlinearity contributes:
partial_dependence_age_linear_function = partial_dependence(linear_function_pdp_adapter(linear_weights=pho_lin_weights), sm.add_constant(X, prepend=True), features=["age"], grid_resolution=100, method="brute", kind ="average", percentiles = (0,1))
axs[0].plot(partial_dependence_age_linear_function["grid_values"][0]*age_std+age_mean, 
        partial_dependence_age_linear_function["average"][0], color="black", linewidth=3, linestyle="dashed", label="PDP of linear component")

# we also insert vertical lines showing the 10th and 90th percentiles of the data
axs[0].axvline(np.percentile(X.loc[:, "age"], 10)*age_std+age_mean, color="black", linestyle="dotted", label="10th/90th percentiles")
axs[0].axvline(np.percentile(X.loc[:, "age"], 90)*age_std+age_mean, color="black", linestyle="dotted")

#Then, we do a PDP for "avg_glucose_level"
glucose_mean = original_means[stroke.columns.get_loc("avg_glucose_level")]
glucose_std = original_stds[stroke.columns.get_loc("avg_glucose_level")]
partial_dependence_glucose = partial_dependence(sklearn_logits_pdp_adapter(extended_glm=extended_glm, pdp_type="prediction"), X, features=["avg_glucose_level"], grid_resolution=100, method="brute", kind ="both", percentiles = (0,1))
axs[1].plot(partial_dependence_glucose["grid_values"][0]*glucose_std+glucose_mean, 
        partial_dependence_glucose["average"][0], color="blue", linewidth=3)
axs[1].set_xlabel("Average Glucose Level", fontsize = 16)


# we again insert vertical lines showing the 10th and 90th percentiles of the data
axs[1].axvline(np.percentile(X.loc[:, "avg_glucose_level"], 10)*glucose_std+glucose_mean, color="black", linestyle="dotted")
axs[1].axvline(np.percentile(X.loc[:, "avg_glucose_level"], 90)*glucose_std+glucose_mean, color="black", linestyle="dotted")

# we also plot the linear function of the extended GLM for avg_glucose_level, to see how much the nonlinearity contributes:
partial_dependence_glucose_linear_function = partial_dependence(linear_function_pdp_adapter(linear_weights=pho_lin_weights), sm.add_constant(X, prepend=True), features=["avg_glucose_level"], grid_resolution=100, method="brute", kind ="average", percentiles = (0,1))
axs[1].plot(partial_dependence_glucose_linear_function["grid_values"][0]*glucose_std+glucose_mean, 
        partial_dependence_age_linear_function["average"][0], color="black", linewidth=3, linestyle="dashed", label="PDP of linear component $\\textbf{X}\\beta$")

#last, we do a pdp for "bmi"
bmi_mean = original_means[stroke.columns.get_loc("bmi")]
bmi_std = original_stds[stroke.columns.get_loc("bmi")]
partial_dependence_bmi = partial_dependence(sklearn_logits_pdp_adapter(extended_glm=extended_glm, pdp_type="prediction"), X, features=["bmi"], grid_resolution=100, method="brute", kind ="both", percentiles= (0,1))
axs[2].plot(partial_dependence_bmi["grid_values"][0]*bmi_std+bmi_mean, 
        partial_dependence_bmi["average"][0], color="blue", linewidth=3)
axs[2].set_xlabel("BMI", fontsize = 16)

# we again insert vertical lines showing the 10th and 90th percentiles of the data
axs[2].axvline(np.percentile(X.loc[:, "bmi"], 10)*bmi_std+bmi_mean, color="black", linestyle="dotted")
axs[2].axvline(np.percentile(X.loc[:, "bmi"], 90)*bmi_std+bmi_mean, color="black", linestyle="dotted")

# we also plot the linear function of the extended GLM for bmi, to see how much the nonlinearity contributes:
partial_dependence_bmi_linear_function = partial_dependence(linear_function_pdp_adapter(linear_weights=pho_lin_weights), sm.add_constant(X, prepend=True), features=["bmi"], grid_resolution=100, method="brute", kind ="average", percentiles = (0,1))
axs[2].plot(partial_dependence_bmi_linear_function["grid_values"][0]*bmi_std+bmi_mean, 
        partial_dependence_age_linear_function["average"][0], color="black", linewidth=3, linestyle="dashed", label="PDP of linear component $X\\beta$")


axs[0].legend(fontsize=16, loc="lower left")
fig.tight_layout()
fig.savefig(f"{path}_pdp_overview.pdf", bbox_inches="tight")

#%% We compute the BMI-PDP three times, and split it each time w.r.t. the binary variables "hypertension", "heart_disease" or "ever_married"

fig, axs = plt.subplots(1,3, figsize=(18, 6), sharey=True)

variables_to_split = ["hypertension", "heart_disease", "sex"]
for i, variable in enumerate(variables_to_split):
    positive_subset = X.loc[X[variable] == 1, :]
    negative_subset = X.loc[X[variable] == 0, :]
    partial_dependence_bmi_positive = partial_dependence(sklearn_logits_pdp_adapter(extended_glm=extended_glm, pdp_type="prediction"), positive_subset, features=["bmi"], grid_resolution=100, method="brute", kind ="average", percentiles = (0,1))
    partial_dependence_bmi_negative = partial_dependence(sklearn_logits_pdp_adapter(extended_glm=extended_glm, pdp_type="prediction"), negative_subset, features=["bmi"], grid_resolution=100, method="brute", kind ="average", percentiles = (0,1))
    axs[i].plot(partial_dependence_bmi_positive["grid_values"][0]*bmi_std+bmi_mean,
            partial_dependence_bmi_positive["average"][0], color="blue", linewidth=3, label=f"{variable} = 1")
    axs[i].plot(partial_dependence_bmi_negative["grid_values"][0]*bmi_std+bmi_mean,
            partial_dependence_bmi_negative["average"][0], color="orange", linewidth=3, label=f"{variable} = 0")
    axs[i].set_xlabel("BMI", fontsize = 16)
    axs[i].set_title(f"PDP of BMI split by {variable}", fontsize = 18)
    axs[i].legend(fontsize=12)
axs[0].set_ylabel("Partial dependence", fontsize = 16)
fig.tight_layout()


#%% we compute PDPs of all binary transformed covariates "sex", "hypertension", "heart_disease", "ever_married", "Residence_type" and "smoking_status" w.r.t. bmi

target_variables  = ["sex", "hypertension", "heart_disease", "ever_married", "Residence_type", "smoking_status"]
target_variable_labels = ["Sex", "Hypertension", "Heart disease", "Ever married", "Residence type", "Smoking status"]
transformed_train = extended_glm.NN_transform(torch.tensor(np.array(X), dtype=torch.float32, device=extended_glm.used_device)).detach().cpu().numpy().mean(axis=0)

fig, axs = plt.subplots(2,3, figsize=(12, 7), sharex=True, sharey=False)
for i, target_variable in enumerate(target_variables):
    partial_dependence_binary = partial_dependence(sklearn_logits_pdp_adapter(extended_glm=extended_glm, pdp_type="T_p", target_variable=X.columns.get_loc(target_variable)), X, features=["bmi"], grid_resolution=100, method="brute", kind ="average", percentiles = (0,1))
    axs[i//3, i%3].plot(partial_dependence_binary["grid_values"][0]*bmi_std+bmi_mean,
            partial_dependence_binary["average"][0], color="blue", linewidth=3, label= "PDP plot")
    # to compare, we also plot the average value of the transformed covariate (i.e. the "T_p" function) for the same variable, as a function of bmi:
    transformed_variable = transformed_train[X.columns.get_loc(target_variable)]
    axs[i//3, i%3].axhline(transformed_variable, color="red", linestyle="dashed", label=f"Average of transformed variable")


    axs[i//3, i%3].set_xlabel("BMI", fontsize = 12)
    axs[i//3, i%3].set_title(f"{target_variable_labels[i]}", fontsize = 18)
    axs[i//3, i%3].axvline(np.percentile(X.loc[:, "bmi"], 10)*bmi_std+bmi_mean, color="black", linestyle="dotted")
    axs[i//3, i%3].axvline(np.percentile(X.loc[:, "bmi"], 90)*bmi_std+bmi_mean, color="black", linestyle="dotted")
    if i%3 == 0:
        axs[i//3, i%3].set_ylabel("Partial dependence", fontsize = 12)
    
axs[1, 2].legend(loc = (0.45, 0.01), fontsize = 14, framealpha = 1)
fig.savefig(f"{path}_bmi_pdps.pdf", bbox_inches="tight")
fig.tight_layout()
