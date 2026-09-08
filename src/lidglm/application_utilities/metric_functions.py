"""
Script containing metrics to evaluate and compare various types of models. Mostly used for trials and model comparison on data.
"""

import torch
import numpy as np
import sklearn.metrics as metrics
from ray import train
import scipy
import statsmodels.api as sm

class LidglmMetrics():
    
    @classmethod
    def accuracy(cls, extended_glm, X_test, y_test, **kwargs):
        """ 
        Adapter for the sklearn accuracy_score function
        """
        pred = torch.round(extended_glm.predict(X_test)).detach().cpu().numpy()
        return metrics.accuracy_score(y_true=y_test.detach().cpu().numpy(), y_pred=pred)

    @classmethod
    def precision(cls, extended_glm, X_test, y_test, **kwargs):
        """ 
        Adapter for the sklearn precision_score function
        """
        pred = torch.round(extended_glm.predict(X_test)).detach().cpu().numpy()
        return metrics.precision_score(y_true=y_test.detach().cpu().numpy(), y_pred=pred)

    @classmethod
    def recall(cls, extended_glm, X_test, y_test, **kwargs):
        """ 
        Adapter for the sklearn recall_score function
        """
        pred = torch.round(extended_glm.predict(X_test)).detach().cpu().numpy()
        return metrics.recall_score(y_true=y_test.detach().cpu().numpy(), y_pred=pred)

    @classmethod
    def f1_score(cls, extended_glm, X_test, y_test, **kwargs):
        """ 
        Adapter for the sklearn f1_score function
        """
        pred = torch.round(extended_glm.predict(X_test)).detach().cpu().numpy()
        return metrics.f1_score(y_true=y_test.detach().cpu().numpy(), y_pred=pred)
    
    @classmethod
    def roc_auc(cls, extended_glm, X_test, y_test, **kwargs):
        """ 
        Adapter for the sklearn roc_auc_score function
        """
        pred = extended_glm.predict(X_test).detach().cpu().numpy()
        return metrics.roc_auc_score(y_true=y_test.detach().cpu().numpy(), y_score=pred)
    
    @classmethod
    def brier_score(cls, extended_glm, X_test, y_test, **kwargs):
        """ 
        Adapter for the sklearn brier_score_loss function
        """
        pred = extended_glm.predict(X_test).detach().cpu().numpy()
        return metrics.brier_score_loss(y_true=y_test.detach().cpu().numpy(), y_prob=pred)

    @classmethod
    def mse(cls, extended_glm, X_test, y_test, **kwargs):
        """ 
        Adapter for the sklearn mean_squared_error function
        """
        pred = extended_glm.predict(X_test).detach().cpu().numpy().reshape(y_test.shape)
        return metrics.root_mean_squared_error(y_true=y_test.detach().cpu().numpy(), y_pred=pred) **2 
    
    @classmethod
    def negative_loglike(cls, extended_glm, X_test, y_test, **kwargs):
        """ 
        Calls the internal log_likelihood function
        """
        return -extended_glm.log_like_obs(y_test, X_test).detach().cpu().numpy().sum()

    @classmethod
    def mean_difference_to_base_glm(cls, extended_glm, X_test, y_test, **kwargs):
        extended_beta = np.append(extended_glm.read_linear_weights(), extended_glm.read_linear_bias()).reshape(-1)
        return (np.abs(kwargs["traditional_beta"]-extended_beta)).mean()
    
    @classmethod
    def compare_to_true_beta(cls, true_beta):
        def compare_to_true_beta_helper(true_beta, estimated_beta):
            return np.mean((true_beta - estimated_beta)**2)
        
        return lambda extended_glm, X_test, y_test, **kwargs: compare_to_true_beta_helper(true_beta, 
                                                                                          np.append(extended_glm.read_linear_bias().detach().cpu().numpy().reshape(-1),
                                                                                                    extended_glm.read_linear_weights().detach().cpu().numpy().reshape(-1)))

    @classmethod
    def true_log_like(cls, dist:scipy.stats.rv_continuous, function):
        def true_log_like_helper(dist, function, X_test, y_test):
            means = function(X_test)
            log_like = np.log(dist.pdf(y_test, loc=means))
            return -np.sum(log_like)
        
        return lambda extended_glm, X_test, y_test, **kwargs: true_log_like_helper(dist=dist, function=function, X_test=X_test, y_test=y_test)

    @classmethod
    def numpy_based_metrics(cls, prediction, y_test, model_string, metrics_list, other_computed_metrics, log_like = None, model =None, other_specifications=None,  checkpoint = None, **kwargs):
        """
        This function evaluates the accuracy, precision, recall and f1-score of the given model on the given data and appends them to the hyperparameter optimizer.
        
        -------------
        Parameters
        metrics: list of strings
        """
                
        reported_metrics = other_computed_metrics
        if log_like is not None:
            reported_metrics = reported_metrics |{"negative log_like": (-log_like)}
        
        for metric_string in metrics_list:
            match metric_string:
                case "accuracy": computed_metric = metrics.accuracy_score(y_true=y_test, y_pred=np.round(prediction))
                case "precision": computed_metric = metrics.precision_score(y_true=y_test, y_pred=np.round(prediction)) 
                case "recall": computed_metric =  metrics.recall_score(y_true=y_test, y_pred=np.round(prediction))
                case "f1_score": computed_metric = metrics.f1_score(y_true=y_test, y_pred=np.round(prediction))
                case "mse": computed_metric = (metrics.root_mean_squared_error(y_true= y_test , y_pred = prediction)**2)
                case "mse_true_beta": computed_metric = (metrics.root_mean_squared_error(model.params,other_specifications["true_beta"]))**2
                case _: continue
            reported_metrics = reported_metrics | {metric_string: computed_metric}
        if checkpoint is not None:
            train.report(metrics=reported_metrics, checkpoint=checkpoint)
        else:
            train.report(metrics=reported_metrics)    
        


# We also implemented some helper functions for traditional GLM models

class traditional_glm_metrics():

    @classmethod
    def negative_loglike(cls, trad_glm, X_test, y_test, **kwargs):
        """ 
        Computes the negative log-likelihood of the traditional GLM model.
        """
        return -trad_glm.family.loglike(y_test, trad_glm.predict(sm.add_constant(X_test, prepend=True)), scale =trad_glm.scale).sum()
    
    @classmethod
    def roc_auc(cls, trad_glm, X_test, y_test, **kwargs):
        """ 
        Wrapper for the sklearn roc_auc_score function
        """
        pred = trad_glm.predict(sm.add_constant(X_test, prepend=True)).reshape(-1)
        return metrics.roc_auc_score(y_true=y_test.reshape(-1), y_score=pred)
    
    @classmethod
    def brier_score(cls, trad_glm, X_test, y_test, **kwargs):
        """ 
        Wrapper for the sklearn brier_score_loss function
        """
        pred = trad_glm.predict(sm.add_constant(X_test, prepend=True)).reshape(-1)
        return metrics.brier_score_loss(y_true=y_test.reshape(-1), y_prob=pred)
    
    @classmethod
    def accuracy(cls, trad_glm, X_test, y_test, **kwargs):
        """ 
        Wrapper for the sklearn accuracy_score function
        """
        pred = np.round(trad_glm.predict(sm.add_constant(X_test, prepend=True))).reshape(-1)
        return metrics.accuracy_score(y_true=y_test.reshape(-1), y_pred=pred)
    
    @classmethod
    def f1_score(cls, trad_glm, X_test, y_test, **kwargs):
        """ 
        Wrapper for the sklearn f1_score function
        """
        pred = np.round(trad_glm.predict(sm.add_constant(X_test, prepend=True))).reshape(-1)
        return metrics.f1_score(y_true=y_test.reshape(-1), y_pred=pred)
    
    @classmethod
    def mse(cls, trad_glm, X_test, y_test, **kwargs):
        """ 
        Wrapper for the sklearn mean_squared_error function
        """
        pred = trad_glm.predict(sm.add_constant(X_test, prepend=True)).reshape(-1)
        return metrics.mean_squared_error(y_true=y_test.reshape(-1), y_pred=pred)