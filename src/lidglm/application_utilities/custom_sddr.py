# A modified version of the training function from the official SDDR implementation which allows a custom train/Val split for early stopping
# The PySddr package needs to be installed (from https://github.com/HelmholtzAI-Consultants-Munich/PySDDR) for this model to function
import sys
import os
import numpy as np
import copy
from sddr import Sddr
from sddr.sddrnetwork import SddrNet
from sddr.utils.dataset import SddrDataset
import torch
from torch.utils.data import DataLoader
from ray import train as ray_train
from ray.train import Checkpoint
from sklearn.metrics import roc_auc_score, log_loss
import tempfile

class CustomSddr(Sddr):
    #  We implement a custom training function that can accept a set validation split and is also compatible with raytune (and sets checkpoints).
    # By inherinting from Sddr, all other functions, properties etc. are kept intact
    # For this, we adjust the training function of the original SDDR model:
    def custom_train(self, train_target, train_structured_data, val_target, val_structured_data,
                     train_unstructured_data=dict(), val_unstructured_data=dict(),
                     resume=False, plot=False, hyperparam_opt=False):
        '''
        Trains the SddrNet for a number of epochs

        Parameters
        ----------
            train_target: str / Pandas.DataFrame
                target (Y), given as:
                - string: file path pointing to the target column in csv format. This file must contain a column header
                          that corresponds to the name of the target variable. If target variable (Y) is given as file path,
                          the input matrix (X) should also be given as file path.
                - string: name of the target variable that will be extracted from the input matrix 'data'.
                          In this case the taget variable must be contained in the input matrix 'data'.
                - pandas dataframe: the target variable as pandas dataframe column.
                          In this case the target variable must be excluded from the input matrix 'data'.
            train_structured_data: str / Pandas.DataFrame
                input dataset (X), given as:
                - string: file path pointing to the input matrix in csv format. This file must contain column headers
                          that correspond to the names used in the formula. If input matrix (X) is given as file path,
                          the target variable (Y) should also be given as file path.
                - pandas dataframe: input matrix as pandas object with columns names that correspond to the names used in the
                          formula.
            train_unstructured_data: dictionary - default empty dict
                The information of unstructured data, including file paths of the unstructured data and the type of it (image...)
            val_target:
            val_structured_data: str / Pandas.DataFrame
            val_unstructured_data: dictionary - default empty dict
            resume: bool - default False
                If true, the model could be continue trained based on the loaded results.
            plot: boolean - default False
                If true, the training loss vs epochs could be plotted
        '''

        epoch_print_interval = max(1, int(self.config['train_parameters']['epochs'] / 10))
        target = train_target
        structured_data = train_structured_data
        unstructured_data = train_unstructured_data
        if resume:
            self.dataset = SddrDataset(structured_data, self.prepare_data, target, unstructured_data, fit=False)
        else:
            self.dataset = SddrDataset(structured_data, self.prepare_data, target, unstructured_data)
            self.val_dataset = SddrDataset(val_structured_data, self.prepare_data, val_target, val_unstructured_data)
            self.net = SddrNet(self.family, self.prepare_data.network_info_dict, self.p)
            self.net = self.net.to(self.device)
            self.P = self.prepare_data.get_penalty_matrix(self.device)
            self._setup_optim()
            self.cur_epoch = 0

        # get number of train and validation samples
        # document val_split is 0.2 for e.g. 20% holdout
        if 'val_split' in self.config['train_parameters'].keys():
            # is ignored
            pass

        n_val = int(len(self.dataset))
        n_train = len(self.dataset)
        # split the dataset randomly to train and val
        train, val = self.dataset, self.val_dataset

        # load train and val data with data loader
        self.train_loader = DataLoader(train,
                                       batch_size=self.config['train_parameters']['batch_size'])
        self.val_loader = DataLoader(val,
                                     batch_size=self.config['train_parameters']['batch_size'])

        train_loss_list = []
        val_loss_list = []

        if 'early_stop_epochs' in self.config['train_parameters'].keys():
            early_stop_counter = 0
            if not resume:
                self.cur_best_loss = sys.maxsize
            if 'early_stop_epsilon' in self.config['train_parameters'].keys():
                eps = self.config['train_parameters']['early_stop_epsilon']
            else:
                eps = 0.001
        cooldown = 0
        best_model_state = self.net.state_dict()
        # print('Beginning training ...')
        for epoch in range(self.cur_epoch, self.config['train_parameters']['epochs']):
            computed_metrics = {}
            self.net.train()
            self.epoch_train_loss = 0
            for batch in self.train_loader:
                # for each batch
                target = batch['target'].float().to(self.device)
                datadict = batch['datadict']

                # send each input batch to the current device
                for param in datadict.keys():
                    for data_part in datadict[param].keys():
                        datadict[param][data_part] = datadict[param][data_part].to(self.device)

                # get the network output
                self.optimizer.zero_grad()
                output = self.net(datadict)

                # compute the loss and add regularization
                loss = torch.mean(self.net.get_log_loss(target))
                loss += self.net.get_regularization(self.P).squeeze_()

                # and backprobagate
                loss.backward()
                self.optimizer.step()
                self.epoch_train_loss += loss.item()

            # compute the avg loss over all batches in the epoch
            self.epoch_train_loss = self.epoch_train_loss / len(self.train_loader)

            # and save it in a list in case we want to print later
            train_loss_list.append(self.epoch_train_loss)
            loss_dict = {"average_loss": self.epoch_train_loss}
            # after each epoch of training evaluate performance on validation set
            with torch.no_grad():
                self.net.eval()
                self.epoch_val_loss = 0
                for batch in self.val_loader:
                    # for each batch
                    target = batch['target'].float().to(self.device)
                    datadict = batch['datadict']

                    # send each input batch to the current device
                    for param in datadict.keys():
                        for data_part in datadict[param].keys():
                            datadict[param][data_part] = datadict[param][data_part].to(self.device)
                    _ = self.net(datadict)
                    # compute the loss and add regularization
                    val_batch_loss = torch.mean(self.net.get_log_loss(target))
                    val_batch_loss += self.net.get_regularization(self.P).squeeze_()
                    self.epoch_val_loss += val_batch_loss.item()
                if len(self.val_loader) != 0:
                    self.epoch_val_loss = self.epoch_val_loss / len(self.val_loader)
                val_loss_list.append(self.epoch_val_loss)
                loss_dict["validation_loss"] = self.epoch_val_loss

                # check if early stopping has been set
                if 'early_stop_epochs' in self.config['train_parameters'].keys():
                    # if model performance improves dif will be positive
                    dif = self.cur_best_loss - self.epoch_val_loss
                    if dif > eps:
                        self.cur_best_loss = self.epoch_val_loss
                        early_stop_counter = 0
                        if hyperparam_opt:
                            if self.config['distribution'] == "Bernoulli":
                                predictions = self.predict(val_structured_data)[0].logits.cpu().detach().numpy()
                                computed_metrics["brier_score"] = np.mean((predictions - np.array(val_target)) ** 2)
                                # compute ROC AUC:
                                roc_auc = roc_auc_score(np.array(val_target), predictions)
                                computed_metrics["roc_auc"] = roc_auc
                                validation_loglike = log_loss(np.array(val_target), predictions)
                                computed_metrics["validation_loglike"] = validation_loglike
                            if cooldown == 0:
                                with tempfile.TemporaryDirectory() as tempdir:
                                    torch.save({"epoch": epoch, "model_state": self.net.state_dict(),
                                                "working_dir": os.getcwd()},
                                               os.path.join(tempdir, "checkpoint.pt"))
                                    ray_train.report(metrics=loss_dict | computed_metrics,
                                                     checkpoint=Checkpoint.from_directory(tempdir))
                                cooldown = 10
                            else:
                                cooldown -= 1
                        best_model_state = copy.deepcopy(self.net.state_dict())
                    else:
                        early_stop_counter += 1

            if hyperparam_opt and (epoch % epoch_print_interval == 0 and len(self.val_loader) != 0):
                if self.config['distribution'] == "Bernoulli":
                    predictions = self.predict(val_structured_data)[0].logits.cpu().detach().numpy()
                    computed_metrics["brier_score"] = np.mean((predictions - np.array(val_target)) ** 2)
                    # compute ROC AUC:
                    roc_auc = roc_auc_score(np.array(val_target), predictions)
                    computed_metrics["roc_auc"] = roc_auc
                    validation_loglike = log_loss(np.array(val_target), predictions)
                    computed_metrics["validation_loglike"] = validation_loglike
                # save checkpoint in temporary file; will be permanent after being reported
                with tempfile.TemporaryDirectory() as tempdir:
                    torch.save({"epoch": epoch, "model_state": self.net.state_dict(), "working_dir": os.getcwd()},
                               os.path.join(tempdir, "checkpoint.pt"))
                    self.save()
                    ray_train.report(metrics=loss_dict | computed_metrics,
                                     checkpoint=Checkpoint.from_directory(tempdir))

            if 'early_stop_epochs' in self.config['train_parameters'].keys() and early_stop_counter == \
                    self.config['train_parameters']['early_stop_epochs']:
                print(
                    'Validation loss has not improved for the last %s epochs! To avoid overfitting we are going to stop training now' % (
                        early_stop_counter))
                self.net.load_state_dict(best_model_state)
                # we will also compute one last checkpoint with the best model:
                if hyperparam_opt:
                    with torch.no_grad():
                        self.net.eval()
                        train_datadict = self.dataset[:]['datadict']
                        # move each input batch to the current device
                        for param in train_datadict.keys():
                            for data_part in train_datadict[param].keys():
                                train_datadict[param][data_part] = train_datadict[param][data_part].to(self.device)
                        _ = self.net(train_datadict)
                        average_loss = self.net.get_log_loss(
                            torch.tensor(train_target.values, device=self.device, dtype=torch.float32)).mean().item()
                        loss_dict = {"average_loss": average_loss}
                        val_datadict = self.val_dataset[:]['datadict']
                        # move each input batch to the current device
                        for param in val_datadict.keys():
                            for data_part in val_datadict[param].keys():
                                val_datadict[param][data_part] = val_datadict[param][data_part].to(self.device)
                        _ = self.net(val_datadict)
                        validation_loss = self.net.get_log_loss(
                            torch.tensor(val_target.values, device=self.device, dtype=torch.float32)).mean().item()
                        loss_dict["validation_loss"] = validation_loss
                        predictions = self.predict(val_structured_data)[0].logits.cpu().detach().numpy() if self.config[
                                                                                                                'distribution'] == "Bernoulli" else \
                        self.predict(val_structured_data)[0].loc.cpu().detach().numpy()
                        computed_metrics = {}
                        if self.config['distribution'] == "Bernoulli":
                            computed_metrics["brier_score"] = np.mean((predictions - np.array(val_target)) ** 2)
                            # compute ROC AUC:
                            roc_auc = roc_auc_score(np.array(val_target), predictions)
                            computed_metrics["roc_auc"] = roc_auc
                            validation_loglike = log_loss(np.array(val_target), predictions)
                        else:
                            validation_loglike = - self.predict(val_structured_data, clipping=True)[0].log_prob(
                                torch.tensor(val_target.values, dtype=torch.float32).to(self.device).reshape(-1,
                                                                                                             1)).mean().item()
                        computed_metrics["validation_loglike"] = validation_loglike
                        with tempfile.TemporaryDirectory() as tempdir:
                            torch.save(
                                {"epoch": epoch, "model_state": self.net.state_dict(), "working_dir": os.getcwd()},
                                os.path.join(tempdir, "checkpoint.pt"))
                            ray_train.report(metrics=loss_dict | computed_metrics,
                                             checkpoint=Checkpoint.from_directory(tempdir))

                break
        return train_loss_list, val_loss_list