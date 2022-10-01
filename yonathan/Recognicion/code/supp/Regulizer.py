import pickle
from functools import reduce
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
from supp.data_functions import preprocess
import time
import os
import pickle

class Regulizer_base():
    def __init__(self, lamda, opts):

        # Configurations.
        self.lamda = lamda
        self.opts = opts
        self.model_old = opts.model_old
        self.model = opts.model
        self.old_params = {}
        self.fisher = {}

    def estimate_fisher_second_method(self,data_loader, batch_size = 10):
        fisher = {}
        for n,p in self.model_old.named_parameters():
            fisher[n] = 0.0
        time1 =time.time()
        estimate = True
        if estimate:
            for inputs in data_loader:
                inputs = preprocess(inputs)
                outs = self.model_old(inputs)  # Compute the model output.
                loss = self.opts.loss_fun(self.model_old, inputs, outs)  # Compute the loss.
    #            opts.optimizer.zero_grad()  # Reset the optimizer.
                loss.backward()  # Do a backward pass.
                for n, p in self.model_old.named_parameters():
                    if p.grad != None:
                     fisher[n] += p.grad ** 2 /len(data_loader)
        time2 = time.time()

        print(time2 - time1)

        return fisher  # Return the loss and the output.


    def consolidate(self, fisher):
        for n, p in self.model_old.named_parameters():
            self.old_params[n] = p.clone()
        self.fisher = fisher

    def ewc_loss(self, cuda=False):

        losses = []
        fisher = self.fisher
        for n, p in self.model.named_parameters():
            # retrieve the consolidated mean and fisher information.

            old_param = self.old_params[n]

            # wrap mean and fisher in variables.
            # calculate a ewc loss. (assumes the parameter's prior as
            # gaussian distribution with the estimated mean and the
            # estimated cramer-rao lower bound variance, which is
            # equivalent to the inverse of fisher information)
            losses.append((fisher[n] * (p - old_param)**2).sum())
        return self.lamda * sum(losses) 


    def _is_on_cuda(self):
        return next(self.model.parameters()).is_cuda

class Regulizer():
    def __init__(self, lambd, opts, data_loader):
        self.lamb = lambd
        self.model_old = opts.model_old
        path = os.path.join(opts.model_dir,'Fisher_dict')
        self.regulizer = Regulizer_base(lambd, opts)
        if not os.path.exists(path):
          self.fisher = self.regulizer.estimate_fisher_second_method(data_loader, batch_size=10)
          path = open(path,'wb')
          pickle.dump(self.fisher,path)

        else:
         with open(path, "rb") as new_data_file:
           self.fisher = pickle.load(new_data_file)
        self.regulizer.consolidate(self.fisher)

    def loss_step(self):
        return self.regulizer.ewc_loss()