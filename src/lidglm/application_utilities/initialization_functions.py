# Initialization functions to be used for model comparison utilizing the raytune package. For each model class,
# lists of hyperparameters are converted into lists of dicts holding all possible value combinations for that model so
# raytune can iterate over the grid:

import itertools
import copy


def trad_glm_create_params(families = ["gaussian"], links = ["identity"]):
    """
    Raytune hyperparameter initialization for the traditional GLM (without regularization)
    """
    
    parameterwise_values = {"family": families, "link": links}
    

    # In the following, the * unpacks the values-list, and the expression "itertools.product(*parameterwise_values.values())"
    # is therefore equivalent to being itertools.product(parameterwise_values["key1"], parameterwise_values["key2"], etc.)

    # Then, for the list of all possible value combinations; these are combined with the matching keys
    # and the model type is added as an additional parameter for easier usage later on

    return_list = [dict(zip(parameterwise_values.keys(), x))|{"model_type": "traditional_glm"} for x in itertools.product(*parameterwise_values.values())]
    
    return_list = [param_dict for param_dict in return_list if (param_dict["family"].lower() == "gaussian" or
                                                                 param_dict["link"].lower() != "identity")]

    return return_list


def elastic_net_create_params(families = ["gaussian"], links = ["identity"], alphas = [0.1]):
    """
        Raytune hyperparameter initialization for the traditional GLM (with elastic net regularization)
    """

    parameterwise_values = {"family": families, "link": links, "alpha": alphas}

    # see "trad_glm_create_params":
    return_list = [dict(zip(parameterwise_values.keys(), x))|{"model_type": "traditional_glm"} for x in itertools.product(*parameterwise_values.values())]
    
    return_list = [param_dict for param_dict in return_list if (param_dict["family"].lower() == "gaussian" or
                                                                 param_dict["link"].lower() != "identity")]

    return return_list


def lidglm_create_params(families = ["gaussian"], links = ["identity"], lipschitz_consts =[1.99], norms =[2],
                        lrs =[1e-4], epochs=[100], widths = [5], depths = [None], loss_crits =["log_like"],
                        parts_to_train = ["full"], num_blocks =[3], activations = ["swish"],
                        #for the second network:
                        use_nu_d = [False],
                        lipschitz_consts_2 = [None], widths_2 = [None], depths_2 = [None], num_blocks_2 = [None], activations_2 = [None], 
                        nums_batches = [10]):
    """
        Raytune hyperparameter initialization for LiD-GLM
    """
    parameterwise_values_no_nu_d = {"family": families, "link": links, "lipschitz_const":lipschitz_consts,"norm": norms,
                            "lr": lrs,"epochs": epochs, "width": widths, "depth": depths, "loss_crits": loss_crits,
                            "part_to_train": parts_to_train, "num_blocks": num_blocks, "activation": activations, "num_batches": nums_batches}

    # the training scheme "nu_p_then_nu_d" only makes sense for networks containing both components
    if "nu_p_then_nu_d" in parameterwise_values_no_nu_d["part_to_train"]:
        parameterwise_values_no_nu_d["part_to_train"] = copy.deepcopy(parameterwise_values_no_nu_d["part_to_train"])
        parameterwise_values_no_nu_d["part_to_train"].remove("nu_p_then_nu_d")
        if len(parameterwise_values_no_nu_d["part_to_train"]) == 0:
            parameterwise_values_no_nu_d["part_to_train"] = ["full"]
    
    parameterwise_values_with_nu_d = {"family": families, "link": links, "lipschitz_const":lipschitz_consts,"norm": norms,
                            "lr": lrs,"epochs": epochs, "width": widths, "depth": depths, "loss_crits": loss_crits,
                            "part_to_train": parts_to_train, "num_blocks": num_blocks, "activation": activations,
                            "lipschitz_const_2": lipschitz_consts_2, "width_2": widths_2, "depth_2": depths_2, "num_blocks_2": num_blocks_2, "activation_2": activations_2,
                            "num_batches": nums_batches}

    # see "trad_glm_create_params":
    return_list_no_nu_d = [dict(zip(parameterwise_values_no_nu_d.keys(), x))|{"model_type": "ldglm"} for x in itertools.product(*parameterwise_values_no_nu_d.values())]
    return_list_with_nu_d = [dict(zip(parameterwise_values_with_nu_d.keys(), x))|{"model_type": "ldglm"} for x in itertools.product(*parameterwise_values_with_nu_d.values())]

    # filter all entries such that depth, num_blocks and lipschitz_constant are compatible for each potential grid entry:
    return_list_no_nu_d = [param_dict for param_dict in return_list_no_nu_d if
                   (param_dict["depth"] >= param_dict["num_blocks"] and
                    2 ** param_dict["num_blocks"] >= param_dict["lipschitz_const"])]
    return_list_with_nu_d = [param_dict for param_dict in return_list_with_nu_d if (param_dict["depth"]>=param_dict["num_blocks"] and
                                                                2** param_dict["num_blocks"]>= param_dict["lipschitz_const"])]
    return_list_with_nu_d = [param_dict for param_dict in return_list_with_nu_d if param_dict["lipschitz_const_2"] is not None and
                             (param_dict["depth_2"] >= param_dict["num_blocks_2"] and
                              2 ** param_dict["num_blocks_2"] >= param_dict["lipschitz_const_2"])]
    
    # filter entries, such that nu_2 is only used for Gaussian families (obsolete for discrete ones):
    return_list_with_nu_d = [param_dict for param_dict in return_list_with_nu_d if (param_dict["family"].lower() == "gaussian" or
                                                                 all([param_dict["lipschitz_const_2"] is None, param_dict["width_2"] is None,
                                                                      param_dict["depth_2"] is None, param_dict["num_blocks_2"] is None, param_dict["activation_2"] is None]))]

    #filter entries such that the identity link is only used for the gaussian family:
    return_list_no_nu_d = [param_dict for param_dict in return_list_no_nu_d if
                             (param_dict["family"].lower() == "gaussian" or
                              param_dict["link"].lower() != "identity")]
    return_list_with_nu_d = [param_dict for param_dict in return_list_with_nu_d if (param_dict["family"].lower() == "gaussian" or
                                                                 param_dict["link"].lower() != "identity")]




    
    #filter out keys that just have None entries (mostly in cases where no second network exists)
    for i in range(len(return_list_no_nu_d)):
        return_list_no_nu_d[i] = {key:value for key, value in return_list_no_nu_d[i].items() if value is not None}
    for i in range(len(return_list_with_nu_d)):
        return_list_with_nu_d[i] = {key:value for key, value in return_list_with_nu_d[i].items() if value is not None}

    return_list = []
    if True in use_nu_d:
        return_list += return_list_with_nu_d
    if False in use_nu_d:
        return_list += return_list_no_nu_d
    
    return return_list


def basic_nn_create_params(lrs =[1e-4], epochs=[100], widths = [5], depths = [5], num_batches=[10]):
    """
        Raytune hyperparameter initialization for a basic neural network:
    """

    parameterwise_values = {"lr": lrs,"epochs": epochs, "width": widths, "depth": depths, "num_batches": num_batches}

    # see "trad_glm_create_params":
    return [dict(zip(parameterwise_values.keys(), x))|{"model_type": "basic_nn"} for x in itertools.product(*parameterwise_values.values())]


def custom_sddr_create_params(lrs =[1e-4], epochs=[100], widths = [5], depths = [5], num_batches=[10], depths_2 = [2], widths_2 = [16], dropout_rates = [0.1], 
                              families = ["gaussian"]):
    """
        Raytune hyperparameter initialization for a basic SDDR model with our custom implementation:
    """
    
    parameterwise_values = {"lr": lrs,"epochs": epochs, "width": widths, "depth": depths, "num_batches": num_batches, "width_2": widths_2, "depth_2": depths_2, 
                            "dropout_rate": dropout_rates, "family": families}

    # see "trad_glm_create_params":
    return [dict(zip(parameterwise_values.keys(), x))|{"model_type": "custom_sddr"} for x in itertools.product(*parameterwise_values.values())]


def sddr_create_params(families = ["normal"],
                        lrs =[1e-4], epochs=[100], widths = [5], depths = [None], dfs = [7], output_shape = [8], weight_decays = [0.],
                        #for the second network:
                        widths_2 = [None], depths_2 = [None], dfs_2 =[None], output_shape_2 = [None],dropout_rates = [0.1],
                        batch_size = [1024], optimizers = ["adam"], early_stop_epochs = [0], spline_terms = [None], orthog = [False]):
    """
         Raytune hyperparameter initialization for an SDDR model using the official python implementation:
    """

    parameterwise_values = {"family": families, "lr": lrs,"epochs": epochs, "width": widths, "depth": depths, "df": dfs, "output_shape": output_shape,
                            "width_2": widths_2, "depth_2": depths_2, "df_2": dfs_2, "output_shape_2": output_shape_2, "dropout_rate": dropout_rates, 
                            "batch_size": batch_size, "optimizer": optimizers, "weight_decay": weight_decays, "early_stop_epochs": early_stop_epochs, "spline_terms": spline_terms, "orthog": orthog}

    # see "trad_glm_create_params":
    return_list = [dict(zip(parameterwise_values.keys(), x))|{"model_type": "sddr"} for x in itertools.product(*parameterwise_values.values())]
    
    #filter out keys that just have None entries (mostly in cases where no second network exists)
    for i in range(len(return_list)):
        #filter out keys that just have None entries (mostly in cases where no second network exists)
        return_list[i] = {key:value for key, value in return_list[i].items() if value is not None}
    
    return return_list