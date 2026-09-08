# In this file, the saved networks from the cars_interpretation.py file are loaded and the ExLipbab computation is run.
# For the installation of exlipbab see readme
import numpy as np
import pandas as pd
from exlipbab.exlipbab_main import exlipbab_main
from exlipbab.helper_classes.piece_wise_linear_function import PWL_Relu, PWL_Identity, Componentwise_Relu, GroupSort
from exlipbab.helper_classes.polyhedron import Polyhedron, FullPolyhedron
from exlipbab.helper_classes.sub_problem import SubProblem
from exlipbab.helper_classes.custom_activation_functions import GroupSort as GroupSortActivation
from tqdm import tqdm
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import time

# remember to choose the right activation function and group size when constructing the network below


model_string = "insurance_application"
outputs_folder = "./"
# load the saved bike sharing network weights and biases
wts = np.load(f'{outputs_folder}insurance_weights.npy', allow_pickle=True)
# weights need to be transposed for our exlipbab implementation
for i in range(len(wts)):
    wts[i] = wts[i].T

print("wts shapes:", [w.shape for w in wts])

bs = np.load(f'{outputs_folder}insurance_biases.npy', allow_pickle=True)

# we also load the cars dataset to define the input polyhedron
#%% Load the dataset, and do first preprocessing:
    
# fetch dataset
insurance_data = pd.read_csv("../Datasets/insurance_data/insurance.csv")
output_folder = "./"
insurance_data.info()
insurance_data["sex"] = insurance_data["sex"].replace({"female":1, "male":0}).astype(int)
insurance_data["smoker"] = insurance_data["smoker"].replace({"yes":1, "no":0}).astype(int)

# region is a categorical variable with categories 'southwest', 'southeast', 'northwest', 'northeast'
# we'll do a dummy encoding and delete the reference
insurance_data = pd.get_dummies(insurance_data, columns=["region"], drop_first=True, dtype=int)

#finally, we pop the target column
target = insurance_data.pop("charges")

# to stabilize density estimation: normalize the target
target = (target-target.mean())/target.std()

X= np.array(insurance_data)
y = np.array(target)


print("Final dataset size:", X.shape)
# we compute the global Lipschitz constant and use the intire R^{input_dim} as input polyhedron
input_polyhedron = FullPolyhedron(X.shape[1])
# we do not use a lower bound for the ExLipbab computation
lower_bound = 0

# we build a network list of the expected form:
network = []
for i in range(len(wts)):
    network.append((wts[i].T, bs[i]))
    if i < len(wts)-1:
        #CHOOSE CORRECT ACTIVATION FUNCTION

        network.append(PWL_Relu(wts[i].shape[1]))
    else:
        network.append(PWL_Identity(wts[i].shape[1]))

start_time = time.time()
glb, gub, bound_hist = exlipbab_main(N = network, X=input_polyhedron, lower_bound = 0., record_ram = False, solver = "glpk", verbose= False, record_symprop= True)
print(f"LipBaB computation took {time.time()-start_time} seconds")
print("final lower bound", glb)
print("final upper bound", gub)

# we also compute the layerwise bounds for comparison
layerwise_time = time.time()
layerwise_norms = []
for W in wts:
    layerwise_norms.append(np.linalg.norm(W, ord=2))
print(f"Layerwise bound computation took {time.time()-layerwise_time} seconds")

print("layerwise norms", layerwise_norms)
print("layerwise bound", np.prod(layerwise_norms))

# we save the bound history
np.save(f'{outputs_folder}insurance_exlipbab_bound_history.npy', np.array(bound_hist, dtype=object))
# to be safe, we save the lower and upper bounds separately as well
lower_bound_history, upper_bound_history = bound_hist
np.save(f'{outputs_folder}insurance_exlipbab_lower_bound_history.npy', np.array(lower_bound_history, dtype=object))
np.save(f'{outputs_folder}insurance_exlipbab_upper_bound_history.npy', np.array(upper_bound_history, dtype=object))
plt.plot(lower_bound_history, label='Lower Bound')
plt.show()


# final upper bound 0.9844030539149026