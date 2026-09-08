import torch
from matplotlib import pyplot as plt
import seaborn as sns


#%% load simulation data computed in pho_linear_simul.py
loaded_df = torch.load("./outputs/pho_linear_simulation_results_dataframe.pt")

#  for each combination of n and p we do a boxplot of the test log-likelihoods:
n_values = loaded_df["n"].unique()
p_values = loaded_df["p"].unique()
models = ["lm", "sddr_wo_oz", "sddr_w_oz", "pho_orthog_sddr", "uninit_lidglm_coeffs", "pho_uninit_lidglm", "lidglm_coeffs", "pho_lidglm"]

# add squared error columns:
for model_string in models:
    loaded_df[model_string + "_sq_error"] = (loaded_df[model_string]-loaded_df["true_mod"])**2

#summarize over covariates:
mse_df = loaded_df.groupby(["n", "p", "sim_nr"])[[model + "_sq_error" for model in models]].mean().reset_index()

    


fig, ax = plt.subplots(len(n_values), len(p_values), figsize=(10, 5), sharey=True, sharex=True)
fig.supylabel("MSE")
for i, n in enumerate(n_values):
    for j, p in enumerate(p_values):
        subset_df = mse_df[(mse_df["n"] == n) & (mse_df["p"] == p)].drop(columns=["n", "p"])
        subset_df.set_index("sim_nr", inplace=True)
        sns.boxplot(data=subset_df, ax=ax[i, j], log_scale=False,
                    **{"boxprops": {"facecolor": "none", "edgecolor": "black"}, 
                     "medianprops": {"color": "black"},
                     "whiskerprops": {"color": "black"},
                     "capprops": {"color": "black"}})
        ax[i,j].set_xticklabels(["GLM", "SDDR (no OZ)", "SDDR (ONO)", "SDDR (PHO)", "LiDGLM (no OZ,NI)", "LiD-GLM (PHO, NI)", "LiDGLM (no OZ, I)", "LiDGLM (PHO, I)"], rotation = 45, ha="right",
                                rotation_mode="anchor")
        ax[i,j].set_ylabel(f"n = {n}")
        ax[i,j].set_xlabel(f"p={p}")
        ax[i,j].label_outer()
fig.tight_layout()
fig.savefig("reconstruct_linear_effects.pdf")
fig.show()

