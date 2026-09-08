#In this script, a single LiD-GLM is fit to the "Cars" data and various interpretation methods are applied

#import standard packages:
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple

# import custom models:
from lidglm import TransformerUtils, ExtendedGLM_Utils
from lidglm.extended_glm.transformer.Custom_Lipschitz.Custom_Lipschitz import Custom_InducedNormLinear
from lidglm.extended_glm.extended_glm_eval_utls import Extended_glm_eval_utils, sklearn_pdp_adapter, sklearn_statsmodels_pdp_adapter

#import data loading function
from ucimlrepo import fetch_ucirepo 

torch.set_num_threads(1)
seed = 42 # random seed we will use later

#%% Load the dataset, and do first preprocessing:
output_folder = "./outputs/"
# fetch dataset 
auto_mpg = fetch_ucirepo(id=9) 
  
# data (as pandas dataframes) 
X = auto_mpg.data.features 
y = auto_mpg.data.targets 

# check for missing values:
print("Number of missing values:", X.isnull().sum())
# 6 rows with missing values in the 'horsepower' column
# an index is created for later removal of these entries
removal_index = X["horsepower"].isnull()

# We check the feature "cylinders" for rare subclasses:
print("cylinder counts:", X["cylinders"].value_counts())
# we add entries with 3 or 5 cylinders to the removal index, as there are only
# a handful of entries for each of these classes - potentially not enough for
# a model to learn a meaningful relation.
removal_index = removal_index | (X["cylinders"] == 3) | (X["cylinders"] == 5)
print(f"{removal_index.sum()} entries were removed out of {len(y)} total")
# We remove the selected entries from the design matrix and the target vector:
X = X[~removal_index]
y = y[~removal_index]

# the origin column is categorical, so we convert it to one-hot encoding
X = pd.get_dummies(X, columns=["origin"], drop_first=True, dtype=int)

# update the index:
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
#%% Check the data for correlated features:
    
# compute a correlation matrix:
corr_matrix = X.corr()
# plot the matrix as a heatmap:
fig, ax = plt.subplots(figsize=(10, 8))
#plt.imshow(corr_matrix, cmap='YlGnBu', interpolation='nearest')
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='YlGnBu', ax=ax, cbar = False, annot_kws={"size":18})
ax.set_xticklabels(ax.get_xticklabels(), fontsize = 12, rotation = 20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize = 12, rotation = 45)

fig.savefig(f"{output_folder}cars_correlation_matrix.pdf")

# print out the 4 most highly correlated combinations with resp. correlations
# get all pairs with correlation values:
corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False)
# delete self correlations and double entries:
corr_pairs_index = corr_pairs.index
unnecessary_entries_index = [entry for entry in corr_pairs_index if entry[1]> entry[0] or entry[0]==entry[1]]
corr_pairs = corr_pairs.drop(index=unnecessary_entries_index)
print("Most highly correlated features: \n", corr_pairs.head(4))

# We drop the features "displacement" and "weight" as they are highly
# correlated with "horsepower" and "cylinders"
X = X.drop(columns=["displacement", "weight"])

#%% Preparation for model training:
    
# We normalize the covariates and do a train/validation split. As this workbook
# is only for interpretation, we do not create a separate test set

# determine the number of features; needed later
num_params = X.shape[1]



# Train/Validation split:
X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(X, y, test_size=0.2, random_state=seed)
# normalize the data:
non_binary_indices =  np.array([i for i in range(X.shape[1]) if len(np.unique(X.iloc[:, i])) > 2])
original_means_contin = np.zeros(X_train_df.shape[1])
original_stds_contin = np.ones(X_train_df.shape[1])

original_means_contin[non_binary_indices] = X_train_df.iloc[:, non_binary_indices].mean(axis=0)
original_stds_contin[non_binary_indices] = X_train_df.iloc[:, non_binary_indices].std(axis=0)    

X_train_df = (X_train_df - original_means_contin)/original_stds_contin
X_val_df = (X_val_df - original_means_contin)/original_stds_contin
X = (X- original_means_contin)/original_stds_contin
original_means_contin = pd.Series(original_means_contin, index = X.columns)
original_stds_contin = pd.Series(original_stds_contin, index = X.columns)
# convert data to numpy arrays, which are needed for some models
X_train = X_train_df.to_numpy()
X_val = X_val_df.to_numpy()
y_train = y_train_df.to_numpy().reshape(-1)
y_val = y_val_df.to_numpy().reshape(-1)





#%% First Model: traditional (Generalized) Linear Model:

sm_glm = sm.GLM(endog=y_train, exog=sm.add_constant(X_train, prepend = True), family=sm.families.Gaussian(sm.families.links.Identity()))
sm_results = sm_glm.fit(maxiter=10**4, method="newton", tol = 1e-10, disp = True)

#summarize the results:
print(sm_results.summary())


#%% Second model: LiD-GLM:
    
# Maximum Lipschitz constant of nu_p and nu_d, evenly distributed among their respective blocks. 
# Each block has a Lipschitz constant in (1,2) and (block_Lip_const)^{num_blocks} = Lip_const
total_lipschitz_const_nu_p = 1.99
p_norm_used = 2

# initialize nu_p:
depth =  2
width = 12
nu_p = TransformerUtils().initialize_transformer(num_params= num_params,
                                                        lipschitz_const=total_lipschitz_const_nu_p, width=width, depth=depth,
                                                        num_blocks=1, domain_codomain=p_norm_used, activation_string="group_sort", group_size=2)


#convert the traditional GLM into a LiD-GLM by converting it to TorchGLM and prepending the transformer.
# X_train and y_train contained in the statsmodels GLM are automatically converted to tensors:
util = ExtendedGLM_Utils()
extended_glm, y_tensor, X_tensor = util.convert_sm_glm(glm=sm_results, net_transform=nu_p, bias = "first")


# We keep our (small) model on the CPU in this example
extended_glm = extended_glm.to("cpu")
extended_glm.used_device = "cpu"
y_tensor = y_tensor.to("cpu")
X_tensor = X_tensor.to("cpu")

#train the LiD-GLM:
max_epochs = 2000
train_model=False
path = f"{output_folder}cars_extended_glm_trained_y_normalized"

# We implemented the ability to only train selected parts (for example freezing the GLM weights), and also to do early stopping
if train_model:
    loss_hist, val_loss_hist = util.train_model(extended_glm = extended_glm, exog=X_tensor, endog=y_tensor,epochs= max_epochs, part_to_train="full", lr=1e-3, loss_crit="log_like", num_batches=1, 
                                              X_test=torch.tensor(X_val, dtype=torch.float32, device="cpu"), y_test= torch.tensor(y_val, dtype=torch.float32, device="cpu"), early_stopping=True, patience=20)
    #we save the loss histories for reproducability:
    #torch.save(extended_glm.state_dict(), f"{path}_state_dict.pth")
    extended_glm.save_model(path)
    # plot the training and validation loss histories:
    fig, ax = plt.subplots(2,1, figsize=(10, 8))
    ax[0].plot(loss_hist, label="Training Loss")
    ax[1].plot(val_loss_hist,c="g", label="Validation Loss")
    ax[1].set_xlabel("Epochs")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Training Loss")
    ax[1].set_ylabel("Validation Loss")
    fig.savefig(f"{output_folder}LiDGLM_loss_hist.pdf")
else:
    extended_glm = util.load_from_param_dict_and_state(torch.load(f"{path}_params.pth"), torch.load(f"{path}_state_dict.pth"))
    
    


#%%
# we compute the validation loglike
extended_glm.eval()

sm_val_loglike = np.log(sm_results.get_distribution(exog = sm.add_constant(X_val, prepend = True, has_constant="add")).pdf(y_val)).mean()
print("Mean validation loglike of GLM:", sm_val_loglike)

val_log_like = extended_glm.log_like_obs(exog = torch.tensor(X_val, dtype=torch.float32),endog= torch.tensor(y_val, dtype=torch.float32)).detach().cpu().numpy().mean()
print("Mean validation Loglike of Extended GLM:", val_log_like)

# we also compute the validation MSE
with torch.no_grad():
    y_val_pred = extended_glm.predict(torch.tensor(X_val, dtype=torch.float32)).detach().cpu().numpy().reshape(-1)
val_mse = np.mean((y_val_pred - y_val)**2)
print("Validation MSE of Extended GLM:", val_mse)
    

#%% For comparison: train an unconstrained NN on the data with the same architecture as nu_p using sklearn:
# solver lbfgs results in erratic PDPs
nn_model_sklearn = MLPRegressor(hidden_layer_sizes=(12,)*3, activation='relu', 
                                max_iter=max_epochs, random_state=seed,
                                early_stopping=True, n_iter_no_change=20, validation_fraction=0.2, solver = "adam")
nn_model_sklearn.fit(X, y)  # the MLPRegressor function internally also uses train_test_split with the specified seed
                            # even if no custom validation split can be defined in their API, 
                            # it therefore still uses the same train/val split as in our split above

# we print the mse on the validation set for comparison:
y_val_pred_nn = nn_model_sklearn.predict(X_val_df)
val_mse_nn = np.mean((y_val_pred_nn - y_val)**2)
print("Validation MSE of NN:", val_mse_nn)





#%% For comparison: train a traditional Linear Regression model on the data using sklearn:

linear_model_sklearn = LinearRegression()
linear_model_sklearn.fit(X_train_df, y_train.reshape(-1))


#%% Start of evaluation
# Initialize some utilities
extended_pdp_adapter = sklearn_pdp_adapter(extended_glm, pdp_type="prediction")
statsmodels_adapter = sklearn_statsmodels_pdp_adapter(sm_results)
eval_util = Extended_glm_eval_utils()

# We will do a single PDP for each feature except horsepower:
# First "cylinders"
fig, axs = plt.subplots(1, 3, figsize= (6, 7), width_ratios=(2.5,2,2), sharey=True)
fig.tight_layout()
partial_dependence_cylinders = partial_dependence(extended_pdp_adapter, X, features=["cylinders"], kind = "average", grid_resolution=100)
axs[0].bar(range(len(partial_dependence_cylinders["grid_values"][0])), 
       partial_dependence_cylinders["average"][0], width=0.4, align='center', color="darkgreen", edgecolor="black")
axs[0].set_xticks(range(len(partial_dependence_cylinders["grid_values"][0])))
axs[0].set_xticklabels(np.array(partial_dependence_cylinders["grid_values"][0]*original_stds_contin["cylinders"]+original_means_contin["cylinders"]).astype(int).astype(str), fontsize=18)
axs[0].set_xlabel("Cylinders", fontsize = 18)
axs[0].set_yticklabels(axs[0].get_yticklabels(), fontsize = 16)
axs[0].set_ylabel("Partial dependence", fontsize=18)

#Then, "origin_2":

partial_dependence_origin_2 = partial_dependence(extended_pdp_adapter, X, features=["origin_2"], kind = "average", grid_resolution=100)
axs[1].bar(range(len(partial_dependence_origin_2["grid_values"][0])), 
       partial_dependence_origin_2["average"][0], width=0.3, align='center', color="blue", edgecolor="black", hatch = "/")
axs[1].set_xticks(range(len(partial_dependence_origin_2["grid_values"][0])))
axs[1].set_xticklabels(np.array(partial_dependence_origin_2["grid_values"][0]).astype(int).astype(str), fontsize=18)
axs[1].set_xlabel("Origin 2", fontsize = 18)

#Then, "origin_3":
partial_dependence_origin_3 = partial_dependence(extended_pdp_adapter, X, features=["origin_3"], kind = "average", grid_resolution=100)
axs[2].bar(range(len(partial_dependence_origin_3["grid_values"][0])),
        partial_dependence_origin_3["average"][0], width=0.3, align='center', color="red", edgecolor="black", hatch = "\\")
axs[2].set_xticks(range(len(partial_dependence_origin_3["grid_values"][0])))
axs[2].set_xticklabels(np.array(partial_dependence_origin_3["grid_values"][0]).astype(int).astype(str), fontsize=18)
axs[2].set_xlabel("Origin 3", fontsize = 18)
fig.tight_layout()
fig.savefig(f"{output_folder}cars_categorical_pdp.pdf", bbox_inches="tight")

#%% 

#Then, "model_year":
fig, ax = plt.subplots(figsize=(6, 7))
partial_dependence_model_year = partial_dependence(extended_pdp_adapter, X, features=["model_year"], kind = "average", grid_resolution=100)
ax.plot(partial_dependence_model_year["grid_values"][0]*original_stds_contin["model_year"]+original_means_contin["model_year"], 
        partial_dependence_model_year["average"][0], color="blue", linewidth=4)
ax.scatter(X["model_year"]*original_stds_contin["model_year"]+original_means_contin["model_year"], y, alpha=0.3, label="Data points", color="green")
ax.set_xlabel("Model Year", fontsize = 18)
ax.set_ylabel("Partial dependence", fontsize = 16)
ax.set_xticklabels(ax.get_xticklabels(), fontsize = 16)
ax.set_yticklabels(ax.get_yticklabels(), fontsize = 16)
fig.tight_layout()
fig.savefig(f"{output_folder}cars_model_year_pdp.pdf", bbox_inches="tight")

#Then, "acceleration":
fig, ax = plt.subplots(figsize=(6, 7))
partial_dependence_acceleration = partial_dependence(extended_pdp_adapter, X, features=["acceleration"], kind = "average", grid_resolution=100)
ax.plot(partial_dependence_acceleration["grid_values"][0]*original_stds_contin["acceleration"]+original_means_contin["acceleration"], 
        partial_dependence_acceleration["average"][0], color="blue", linewidth=4)
ax.scatter(X["acceleration"]*original_stds_contin["acceleration"]+original_means_contin["acceleration"], y, alpha=0.3, label="Data points", color="green")
ax.set_xlabel("Acceleration", fontsize = 18)
ax.set_ylabel("Partial dependence", fontsize = 16)
ax.set_xticklabels(ax.get_xticklabels(), fontsize = 16)
ax.set_yticklabels(ax.get_yticklabels(), fontsize = 16)
fig.tight_layout()
fig.savefig(f"{output_folder}cars_acceleration_pdp.pdf", bbox_inches="tight")



#%% We will do 3 PDPs for horsepower, one each for the LiD-GLM, the NN and the traditional Linear Regression model.


cylinder_labels = (X["cylinders"]*original_stds_contin["cylinders"] + original_means_contin["cylinders"]).round(0).unique()
cylinder_labels = [str(int(label)) + " cylinders" for label in sorted(cylinder_labels, reverse=False)]
colorlist = ["red", "purple", "blue"]
shapelist = ["-.", "--", ":"]
linewidth = 3
shapeslist = ["s", "x", "o"]
horsepower_series = X["horsepower"]*original_stds_contin["horsepower"]+original_means_contin["horsepower"]
# get the indices for each class of feature "cylinders" in the data:
indices = [(X["cylinders"]*original_stds_contin["cylinders"] + original_means_contin["cylinders"]).round(0)== number for number in [4,6,8]]
alpha = 0.3


# We separate the data into three dataframes, one for each cylinder class:
X_sep_list = [X[X["cylinders"]== i] for i in np.sort(X["cylinders"].unique())]
colorlist = ["red", "darkviolet", "mediumblue"]
shapelist = ["-.", "-", "--"]
fig, axs=plt.subplots(1, 3, figsize = (14, 5), sharey=True)
alpha = 0.25
ax1 = axs[1]
ax2 = axs[2]
ax3 = axs[0]

lines = []
for i, X_sep in enumerate(X_sep_list):
    # LiD-GLM PDP:
    partial_dependence_sep = partial_dependence(extended_pdp_adapter, X_sep, features=["horsepower"], kind="average", grid_resolution=100)
    ax1.plot(partial_dependence_sep["grid_values"][0]*original_stds_contin["horsepower"]+original_means_contin["horsepower"], partial_dependence_sep["average"][0],
              label=cylinder_labels[i], c= colorlist[i], linewidth=3, linestyle = shapelist[i])
    ax1.set_title("LiD-GLM", fontsize = 20)
    
    # NN PDP:
    partial_dependence_sep_nn = partial_dependence(nn_model_sklearn, X_sep, features=["horsepower"], kind="average", grid_resolution=100)
    ax2.plot(partial_dependence_sep_nn["grid_values"][0]*original_stds_contin["horsepower"]+original_means_contin["horsepower"], partial_dependence_sep_nn["average"][0],
              label=cylinder_labels[i], c= colorlist[i], linewidth=3, linestyle = shapelist[i])
    ax2.set_title("Neural Network", fontsize = 20)
    
    # Linear Regression PDP:
    ax3.set_title("Linear Model", fontsize = 20)
    partial_dependence_sep_lr = partial_dependence(linear_model_sklearn, X_sep, features=["horsepower"], kind="average", grid_resolution=100)
    line, = ax3.plot(partial_dependence_sep_lr["grid_values"][0]*original_stds_contin["horsepower"]+original_means_contin["horsepower"], partial_dependence_sep_lr["average"][0],
              label=cylinder_labels[i], c= colorlist[i], linewidth=3, linestyle = shapelist[i])
    lines.append(line)

ax1.scatter(horsepower_series[indices[0]], y[indices[0]], alpha= alpha/2, marker = shapeslist[0], c = colorlist[0])
ax1.scatter(horsepower_series[indices[1]], y[indices[1]], alpha= alpha/1.5, marker = shapeslist[1], c = colorlist[1])
ax1.scatter(horsepower_series[indices[2]], y[indices[2]], alpha= alpha/2, marker = shapeslist[2], c = colorlist[2])
ax2.scatter(horsepower_series[indices[0]], y[indices[0]], alpha= alpha/2, marker = shapeslist[0], c = colorlist[0])
ax2.scatter(horsepower_series[indices[1]], y[indices[1]], alpha= alpha/1.5, marker = shapeslist[1], c = colorlist[1])
ax2.scatter(horsepower_series[indices[2]], y[indices[2]], alpha= alpha/2, marker = shapeslist[2], c = colorlist[2])
dots_zero = ax3.scatter(horsepower_series[indices[0]], y[indices[0]], alpha= alpha/2, marker = shapeslist[0], c = colorlist[0])
dots_one = ax3.scatter(horsepower_series[indices[1]], y[indices[1]], alpha= alpha/1.5, marker = shapeslist[1], c = colorlist[1])
dots_two = ax3.scatter(horsepower_series[indices[2]], y[indices[2]], alpha= alpha/2, marker = shapeslist[2], c = colorlist[2])

ax1.set_xlabel("Horsepower", fontsize = 16)
ax2.set_xlabel("Horsepower", fontsize = 16)
ax3.set_xlabel("Horsepower", fontsize = 16)
ax3.set_ylabel("Partial dependence", fontsize = 18)
leg = ax2.legend([(lines[0], dots_zero), (lines[1], dots_one), (lines[2], dots_two)], cylinder_labels, handler_map={tuple: HandlerTuple(None)}, handlelength = 5, fontsize = 16, markerscale =1.5)

fig.tight_layout()
fig.savefig(f"{output_folder}cars_comparative_horsepower_pdps.pdf")

#%% Generate a coefficient table:
coeff_table = eval_util.coefficient_table(extended_glm=extended_glm, exog=X_tensor, coefficient_names=["bias"]+list(X.columns), original_glm = sm_results)
coeff_table.to_csv(f"{output_folder}CarsCoefficientTable.csv")


#%% we again re-compute the R^2 implementation to test the results for correctness:

# first: apply PHO, get new \tilde{\nu_p}
beta_tilde , tilde_nu_1, Gamma = eval_util.custom_PHO(extended_glm, X_tensor)

#check that PHO works correctly:
tilde_nu_1_output = torch.tensor(tilde_nu_1(X_tensor))
transformed_covars = X_tensor + tilde_nu_1_output
max_transf_covar_diff = np.abs((transformed_covars-extended_glm.NN_transform(X_tensor)).detach().cpu().numpy()).max()
print("Maximum transformed exog deviation: ", max_transf_covar_diff)


# test that the mean of \tilde{\nu_p} is zero (to the degree of float value accuracy):
print(tilde_nu_1_output.mean(axis=0))

x_distance = torch.pow(X_tensor - X_tensor.mean(axis=0), 2).sum(axis=0)
z_distance = torch.pow(transformed_covars - transformed_covars.mean(axis=0), 2).sum(axis = 0 )
nu_norm = torch.pow(tilde_nu_1_output, 2).sum(axis = 0 )

print(z_distance-x_distance-nu_norm)
print("Manual $R^2$", x_distance/z_distance)

implemented_R_sq = eval_util.one_dimensional_R_squared(extended_glm, X_tensor, pho=True)
print("implemented R^2 = ", implemented_R_sq)

# check that the output is the same:
mean_difference = np.abs(extended_glm.predict(X_tensor).detach().cpu().numpy()-
                        (sm.add_constant(transformed_covars.detach().cpu().numpy(), prepend=True)@beta_tilde)).mean()
print("mean difference in predictions:", mean_difference)
    


#%% 

# we load the weights and biases of T_p for further analysis:
nu_p_weights, nu_p_biases = [], []
for layer in extended_glm.nu_1.blocks[0].nnet.net:
    if isinstance(layer, nn.Linear) or isinstance(layer, Custom_InducedNormLinear):
        nu_p_weights.append(layer.normalized_weight.detach().cpu().numpy())
        nu_p_biases.append(layer.bias.detach().cpu().numpy())
 
np.save(f'{output_folder}cars_weights.npy', np.array(nu_p_weights, dtype= object))
np.save(f'{output_folder}cars_biases.npy', np.array(nu_p_biases, dtype= object))
