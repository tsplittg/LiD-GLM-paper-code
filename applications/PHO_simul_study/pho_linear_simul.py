# A simulation study to reconstruct linear effects with several models to test the effect of orthogonalization.
# Based on the setup described in: Rügamer (2023)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import itertools
import statsmodels.api as sm
from multiprocessing import Pool
from numpy.random import SeedSequence
from sklearn.model_selection import train_test_split
import copy
from lidglm.application_utilities.custom_sddr import CustomSddr
from lidglm import ExtendedGLM, ExtendedGLM_Utils
from lidglm.extended_glm.extended_glm_eval_utls import Extended_glm_eval_utils
from lidglm import TorchGLM
from lidglm import TransformerUtils

# create a data frame with all combinations of the settings for which simulations should be run
settings = pd.DataFrame(
    list(itertools.product(
    [50, 100, 1000],  # n
    [1, 3, 10],   # p
    )),columns=['n', 'p'])

n_reps = 20     # number of repititions for each simulation
n_cores = 8    # number of cores used
max_epochs = 5000   # maximum number of epochs used during gradient descent
output_dir = './outputs'


# create random normal data in the shape needed for the maximum setting; for smaller settings, only a subset of the data will be used
data = pd.DataFrame(
    np.random.normal(size=(max(settings['n']), max(settings['p']))),
    columns=[f"x{i+1}" for i in range(max(settings['p']))]
)

#%% ----------------------------------------------------------------

def create_formula(p, orthog):
    """
    Creates the formula strings for the SDDR model based on the number of linear and NN parameters.

    The offical SDDR python implementation always uses orthogonalization. However, as only the variables of the nonlinear terms are orthogonalized that also are used in the linear component,
    we can work around this by copying the dataset, renaming the columns to t_i and passing them into
    the neural net as pseudo-new features. The x_i are then given as input to the linear component and the t_i to the nonlinear one.
    This tricks the model into not orthogonalizing.

    """
    lin_formula = " + ".join([f"x{i+1}" for i in range(p)])
    if orthog:
        nonlin_inputs = " , ".join([f"x{i+1}" for i in range(p)])
    else:
        nonlin_inputs = " , ".join([f"t{i+1}" for i in range(p)])

    return {'loc': f"~ -1 + {lin_formula} + deep_model({nonlin_inputs})", 'scale': "~ 1"}

def additive_predictor_fun(relevant_data, p, coefs= np.linspace(-2.5, 2.5, 11)[np.arange(11) != 6]):
    return np.dot(relevant_data.iloc[:, :p], coefs[:p])


#%% ----------------------------------------------------------------

def run_one_simul(sim_nr, eta, relevant_data, n, p, max_epochs, true_coefs, seed):
    random_generator = np.random.default_rng(seed)
    outcome = eta + random_generator.normal(size=eta.shape)
    
    
    outcome = pd.DataFrame((outcome - np.mean(outcome)))
    outcome = pd.DataFrame(outcome)
    #copy all columns x_i into new columns t_i for the neural net to use as "new" features, see create_formula function
    for i in range(p):
        relevant_data[f"t{i+1}"] = relevant_data[f"x{i+1}"].values

    # Do a train-val split for early stopping
    X_train, X_val, y_train, y_val = train_test_split(relevant_data, outcome, test_size=0.2, random_state=seed)

    ######## fit a linear model as baseline ########

    sm_glm = sm.GLM(endog=y_train, exog=X_train.iloc[:, :p].values, family=sm.families.Gaussian(sm.families.links.Identity()))
    sm_results = sm_glm.fit(maxiter=10**4, method="newton", tol = 1e-10, disp = True)
    lm_coefs = np.array(sm_results.params)

    ######## fit an SDDR model ########

    form_w_orthog = create_formula(p, orthog = True)

    # define the Neural net used:
    deep_models_dict = {
    'deep_model': {
        'model': nn.Sequential(nn.Linear(p, 100), nn.ReLU(),
                                nn.Linear( 100, 50), nn.ReLU(),
                                nn.Linear(50, 1)),
        'output_shape': 1}
    }

    #define training parameters
    train_parameters_w_oz = {
    'batch_size': 50,
    'epochs': max_epochs,
    'degrees_of_freedom': {'loc': 0, 'scale': 0},
    'optimizer' : optim.Adam,
    "lr":1e-3, 
    'dropout_rate': 0.,
    'early_stop_epochs': 20
    }

    #model with orthogonalization:
    mod_w_oz = CustomSddr(distribution="Normal",
        formulas=form_w_orthog,
        deep_models_dict=deep_models_dict,
        train_parameters=train_parameters_w_oz,
        output_dir=output_dir)
    hist_w_oz, val_hist_w_oz = mod_w_oz.custom_train(y_train, X_train, y_val, X_val)

    train_parameters_wo_oz = train_parameters_w_oz.copy()
    form_wo_orthog = create_formula(p, orthog = False)
    mod_wo_oz = CustomSddr(distribution="Normal",
        formulas=form_wo_orthog,
        deep_models_dict=deep_models_dict,
        train_parameters=train_parameters_wo_oz,
        output_dir=output_dir)
    hist_wo_oz, val_hist_wo_oz = mod_wo_oz.custom_train(y_train, X_train, y_val, X_val)

    #PHO on SDDR, following algorithm 2 in the PHO paper with 1 batch
    sddr_coefs = mod_wo_oz.coeff("loc")
    linear_pred = np.zeros_like(y_train.values).reshape(-1)
    for i in range(p):
        linear_pred += sddr_coefs[f"x{i+1}"] * X_train.iloc[:, i].values
    zeta = (mod_wo_oz.predict(X_train)[0].loc.detach().cpu().numpy()).reshape(-1) - linear_pred
    
    X = X_train.iloc[:, :p].values
    H = X.T @ X
    s = X.T @ zeta
    alpha = np.linalg.solve(H, s)
    alpha = alpha.reshape(-1)
    pho_orthog_coeffs = copy.deepcopy(mod_wo_oz.coeff("loc"))
    for i in range(p):
        pho_orthog_coeffs[f"x{i+1}"] = pho_orthog_coeffs[f"x{i+1}"] + alpha[i]

    

    #Fit a LiD-GLM
    total_lipschitz_const = 1.99
    norm_used = 2

    #initialize the transformer (i-ResNet) for the covariates:
    transformer = TransformerUtils().initialize_transformer(num_params= p, lipschitz_const=total_lipschitz_const, width=75,
                                                             depth=2, num_blocks=1, domain_codomain=norm_used, activation_string="relu")

    #convert the traditional GLM into a deep GLM by converting it to TorchGLM and prepending the transformer:
    util = ExtendedGLM_Utils()
    
    extended_glm, target_tensor, X_tensor = util.convert_sm_glm(glm=sm_results, net_transform=transformer, bias = "none")
    #train the model:

    loss_hist, test_error_hist = util.train_model(extended_glm = extended_glm, exog=X_tensor, endog=target_tensor,epochs= max_epochs, part_to_train="full", lr=4e-4, loss_crit="log_like", num_batches=1,
                                                  X_test= torch.tensor(X_val.iloc[:, :p].values, dtype=torch.float32, device= X_tensor.device),
                                                  y_test= torch.tensor(y_val.values, dtype=torch.float32, device= X_tensor.device),
                                                  patience=50, early_stopping = True)

    #PHO on LiD-GLM
    eval_util = Extended_glm_eval_utils()
    lidglm_pho_beta, _, _ = eval_util.custom_PHO(extended_glm, X_tensor)

    # We also test a LiD-Glm that has not been initialized with the linear model coefficients
    uninit_transformer = TransformerUtils().initialize_transformer(num_params= p, lipschitz_const=total_lipschitz_const, width=75,
                                                            depth=2, num_blocks=1, domain_codomain=norm_used, activation_string="relu", group_size=2)
    torch_glm = TorchGLM(num_params=p, family_str="gaussian", link_str="identity", bias=False)

    uninit_lidglm = ExtendedGLM(regular_glm=torch_glm, net_transform=uninit_transformer)
    #train the uninitialized LiD-GLM
    loss_hist2, test_error_hist2 = util.train_model(extended_glm = uninit_lidglm, exog=X_tensor, endog=target_tensor,epochs= 2000, part_to_train="full", lr=1e-1, loss_crit="log_like", num_batches=1,
                                                    X_test= torch.tensor(X_val.iloc[:, :p].values, dtype=torch.float32, device= X_tensor.device),
                                                    y_test= torch.tensor(y_val.values, dtype=torch.float32, device= X_tensor.device),
                                                    patience=50, early_stopping = True)
    uninit_lidglm_pho_beta, _, _ = eval_util.custom_PHO(uninit_lidglm, X_tensor)
    uninit_lidglm_coeffs = uninit_lidglm.read_linear_weights()



    run_result = []
    coeff_keys = list(mod_w_oz.coeff("loc").keys())[:p]

    for coeff_index in range(p):
        coefs = {
            "coeff_index": coeff_index,
            "true_mod" : true_coefs[coeff_index],
            "lm" : lm_coefs[coeff_index],
            "sddr_w_oz" : mod_w_oz.coeff("loc")[coeff_keys[coeff_index]][0],
            "sddr_wo_oz" : mod_wo_oz.coeff("loc")[coeff_keys[coeff_index]][0],
            "pho_orthog_sddr" : pho_orthog_coeffs[coeff_keys[coeff_index]][0],
            "lidglm_coeffs": extended_glm.read_linear_weights()[coeff_index],
            "pho_lidglm": lidglm_pho_beta[coeff_index+1],  # +1 to account for intercept, 
            "uninit_lidglm_coeffs": uninit_lidglm_coeffs[coeff_index],
            "pho_uninit_lidglm": uninit_lidglm_pho_beta[coeff_index+1]
        }
        run_result.append(coefs|{'n': n, 'p': p, 'sim_nr': sim_nr})

    print(f"Finished simulation nr {sim_nr} of setting n={n}, p={p}", flush =True)
    return run_result

def run_specific_setting_run(setting, data, sim_nr, seed):
    torch.set_num_threads(1)
    n = int(setting['n'])
    p = int(setting['p'])

    # extract relevant data
    ind_cols = list(range(p))
    relevant_data = data.iloc[:n, ind_cols].copy()
    
    # generate the true additive predictor:
    true_coefs = np.linspace(-2.5, 2.5, 11)[np.arange(11) != 6]
    eta = additive_predictor_fun(relevant_data, p, coefs=true_coefs)

    run_result = run_one_simul(sim_nr, eta, relevant_data, n, p, max_epochs, true_coefs, seed)
    return run_result

def run_entire_simulation(data):
    numpy_seed_seq = SeedSequence(42).spawn(len(settings)*n_reps)
    setting_run_list = [(settings.iloc[i, :], data, sim_nr, numpy_seed_seq[i*n_reps + sim_nr].entropy) for i in range(len(settings)) for sim_nr in range(n_reps)]

    with Pool(processes=n_cores) as pool:
        results = pool.starmap(run_specific_setting_run, setting_run_list)
    results = pd.DataFrame(itertools.chain(*results) )
    torch.save(results, f"{output_dir}/pho_linear_simulation_results_dataframe.pt")
    return results

if __name__ == "__main__":
    res_list = run_entire_simulation(data)

         
