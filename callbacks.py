import torch
import os
import time
class EarlyStopping():
    def __init__(self,path=None, name=None, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.path = os.getcwd() if None else path
        self.name = time.time() if None else name
        self.early_stop = False

    def __call__(self, Model, val_loss):

        if self.best_loss == None:
            self.best_loss = val_loss

        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            print("Saving the best Model State Dict")
            torch.save(Model.state_dict(),os.path.join(self.path,f"{self.name}_{self.best_loss}"))
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"Early Stopping Counter {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("EARLY STOPPING")
                self.early_stop = True
