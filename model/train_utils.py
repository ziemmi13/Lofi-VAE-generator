import torch
import numpy as np
from comet_ml import Experiment
import os

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, path='checkpoints/checkpoint.pt', verbose=False):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.Inf
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f"Validation loss decreased. Saving model to {self.path}")


def setup_commet_loger(experiment_name, project_name="LOFI-VAE-generator"):
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=project_name
    )

    experiment.set_name(experiment_name)

    return experiment


