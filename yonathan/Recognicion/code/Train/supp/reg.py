from functools import reduce
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
from supp.data_functions import preprocess
import time

class Regulizer_base():
    def __init__(self, lamda, opts):

        # Configurations.
        self.lamda = lamda
        self.opts = opts
        self.model = opts.model
        self.old_params = {}
        self.fisher = {}

    def estimate_fisher_second_method(self,data_loader, batch_size = 10):
        fisher = {}
        for n,p in self.model.named_parameters():
            fisher[n] = 0.0
        time1 =time.time()
        estimate = True
        if estimate:
            for inputs in data_loader:
                inputs = preprocess(inputs)
                outs = self.model(inputs)  # Compute the model output.
                loss = self.opts.loss_fun(self.opts, inputs, outs)  # Compute the loss.
    #            opts.optimizer.zero_grad()  # Reset the optimizer.
                loss.backward()  # Do a backward pass.
                for n, p in self.model.named_parameters():
                    if p.grad != None:
                     fisher[n] += p.grad ** 2 /len(data_loader)
        time2 = time.time()

        print(time2 - time1)

        return fisher  # Return the loss and the output.

    def estimate_fisher(self, data_loader, batch_size = 10):
        # sample loglikelihoods from the dataset.
       # data_loader = utils.get_data_loader(dataset, batch_size)
        loglikelihoods = []
        for idx,inputs in enumerate(data_loader):
        #    x = x.view(batch_size, -1)
            y = inputs[1]
            print(idx)
            outs = self.model(inputs)
          #  inputs = Variable(inputs).cuda() if self._is_on_cuda() else Variable(x)
            y = Variable(y).cuda() if self._is_on_cuda() else Variable(y)
            loglikelihoods.append( F.log_softmax(outs[1], dim=1)[range(batch_size), y.data])
            if idx>5:
                break
          #  torch.cuda.empty_cachee()

        # estimate the fisher information of the parameters.
        loglikelihoods = torch.cat(loglikelihoods).unbind()
        loglikelihood_grads = zip(*[autograd.grad(
            l, self.model.parameters(),
            retain_graph=(i < len(loglikelihoods))
        ) for i, l in enumerate(loglikelihoods, 1)])
        loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]
        param_names = [
            n.replace('.', '__') for n, p in self.named_parameters()
        ]
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    def consolidate(self, fisher):
        for n, p in self.model.named_parameters():
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
        return (self.lamda/2)*sum(losses)


    def _is_on_cuda(self):
        return next(self.model.parameters()).is_cuda

class Regulizer():
    def __init__(self, lambd, model, data_loader):
        self.lamb = lambd
        self.model_old = model
        self.regulizer = Regulizer_base(lambd,model )
        self.fisher = self.regulizer.estimate_fisher_second_method(data_loader, batch_size = 10)
        self.regulizer.consolidate(self.fisher)

    def loss_step(self):
        return self.regulizer.ewc_loss()