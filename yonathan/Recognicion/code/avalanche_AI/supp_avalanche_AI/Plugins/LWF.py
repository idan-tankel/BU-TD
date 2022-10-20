import sys
sys.path.append(r'/home/sverkip/data/BU-TD/yonathan/Recognicion/code')
import copy
import argparse
import torch
from avalanche.training.plugins import LwFPlugin

class MyLWFPlugin(LwFPlugin):
    # TODO - CHANGE THE NAMES.
    """
    A Learning without Forgetting plugin.
    LwF uses distillation to regularize the current loss with soft targets
    taken from a previous version of the model.
    This plugin does not use task identities.
    When used with multi-headed models, all heads are distilled.
    """

    def __init__(self, parser:argparse,prev_model = None):
        """
        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each experience.
        :param temperature: softmax temperature for distillation
        """

        super().__init__()

        self.alpha = parser.alpha_LWF
        self.temperature =parser.temperature_LWF
        self.prev_model = copy.deepcopy(prev_model) if prev_model != None else None

        self.prev_classes = {"0": set()}

        """ 
        In Avalanche, targets of different experiences are not ordered. 
        As a result, some units may be allocated even though their 
        corresponding class has never been seen by the model.
        Knowledge distillation uses only units corresponding to old classes. 
        """

<<<<<<< HEAD
    def _distillation_loss(self, cur_out, prev_out, x):
=======
    def _distillation_loss(self, out, prev_out, x):
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d
        """
        Compute distillation loss between output of the current model and
        and output of the previous (saved) model.
        """
<<<<<<< HEAD
        loss_weight = (x[-1]).unsqueeze(dim = 2)
        cur_out = torch.transpose(cur_out, 2, 1)
        prev_out = torch.transpose(prev_out, 2, 1)
        cur_out_softmax = torch.log_softmax(cur_out / self.temperature, dim = 2 )
        prev_out_softmax = torch.softmax(prev_out / self.temperature, dim = 2)
        dist_loss = - cur_out_softmax * prev_out_softmax
        dist_loss = dist_loss * loss_weight
        dist_loss = dist_loss.sum() / loss_weight.size(0)
        return dist_loss
=======
        loss_weight= (x[-1]).unsqueeze(dim = 1)
     #   loss_weight = loss_weight.view(loss_weight.shape[0], 1, loss_weight.shape[-1])
      #  loss_weight = torch.zeros_like(loss_weight)
       # task_out = out
     #   prev_out_task = prev_out
        log_p = torch.log_softmax(out / self.temperature, dim=1)
        q = torch.softmax(prev_out[0] / self.temperature, dim=1)
        res = torch.nn.functional.kl_div(log_p, q, reduction='none')
        res = res * loss_weight
        res = res.sum() / loss_weight.size(0)
        return res
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d

    def penalty(self,model, out, x, alpha):
        """
        Compute weighted distillation loss.
        """

        if self.prev_model is None:
            return 0
        else:
<<<<<<< HEAD
                # TODO - CHANGE TO THE TASK IDENTITIES.
            y_prev = {"0": self.prev_model(x, head = 0)  }
            y_curr = {"0": model.forward_features(out[1], x, head = 0)  }
=======
            with torch.no_grad():
                '''
                if isinstance(self.prev_model, MultiTaskModule):
                    # output from previous output heads.
                    y_prev = avalanche_forward(self.prev_model, x, None)
                    # in a multitask scenario we need to compute the output
                    # from all the heads, so we need to call forward again.
                    # TODO: can we avoid this?
                    y_curr = avalanche_forward(curr_model, x, None)
                else:  # no task labels
                '''
                # TODO - CHANGE TO THE TASK IDENTITIES.
                y_prev = {"0": self.prev_model(x, head = 0)  }
                y_curr = {"0": model.forward_features(out[1], x, head = 0)  }
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d

            dist_loss = 0
            for task_id in y_prev.keys():
                # compute kd only for previous heads.
                if task_id in self.prev_classes:
                    yp = y_prev[task_id]
                    yc = y_curr[task_id]
<<<<<<< HEAD
                    dist_loss += self._distillation_loss(yc, yp[0], x)
=======
                    dist_loss += self._distillation_loss(yc, yp, x)
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d

            return alpha * dist_loss

    def before_backward(self, strategy, **kwargs):
        """
        Add distillation loss
        """
        alpha = (
            self.alpha[strategy.clock.train_exp_counter]
            if isinstance(self.alpha, (list, tuple))
            else self.alpha
        )
        penalty = self.penalty(strategy.model,
            strategy.mb_output, strategy.mb_x, alpha
        )
<<<<<<< HEAD
      #  print(penalty)
=======
     #   print(penalty)
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d
        strategy.loss += penalty

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned classes.
        """
        if strategy.EpochClock.train_exp_epochs == strategy.EpochClock.max_epochs:
            print("copy")
            self.prev_model = copy.deepcopy(strategy.model)
        task_ids = strategy.experience.dataset.task_set
        for task_id in task_ids:
            task_data = strategy.experience.dataset.task_set[task_id]
            pc = set(task_data.targets)

            if task_id not in self.prev_classes:
                self.prev_classes[str(task_id)] = pc
            else:
                self.prev_classes[str(task_id)] = self.prev_classes[
                    task_id
                ].union(pc)