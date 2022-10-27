from avalanche.training.plugins import AGEMPlugin
import warnings
import torch
from avalanche.models import avalanche_forward
from torch.utils.data import random_split
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.data_loader import ( GroupBalancedInfiniteDataLoader,)
from supp.utils import preprocess
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

class MyAGEMPlugin(AGEMPlugin):
    """Average Gradient Episodic Memory Plugin.

    AGEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous experiences. If the dot product
    between the current gradient and the (average) gradient of a randomly
    sampled set of memory examples is negative, the gradient is projected.
    This plugin does not use task identities.
    """

    def __init__(self,parser, old_dataset,  patterns_per_experience: int, sample_size: int):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param sample_size: number of patterns in memory sample when computing
            reference gradient.
        """

        super().__init__(patterns_per_experience = patterns_per_experience, sample_size = sample_size)


        self.prev_data = old_dataset
        self.parser = parser

        if parser.pretrained_model and self.prev_data != None:
          self.update_memory(self.prev_data, num_workers=0)

    def after_training_exp(self, strategy, **kwargs):
        """Update replay memory with patterns from current experience."""
        if strategy.EpochClock.train_exp_epochs == strategy.EpochClock.max_epochs:
            print("Update memory")
            self.update_memory(strategy.experience.dataset, **kwargs)

    def before_training_iteration(self, strategy, **kwargs):
        """
        Compute reference gradient on memory sample.
        """
        if len(self.buffers) > 0:
            strategy.model.train()
            strategy.optimizer.zero_grad()
            mb = self.sample_from_memory()
            mb = preprocess(mb, device=strategy.device)
          #  xref, yref, tid = mb[0], mb[1], mb[-1]
           # xref, yref = xref.to(strategy.device), yref.to(strategy.device)
            out = strategy.model(mb, head = 0) # TODO - SUPPORT ANY TASK ID OR OMIT THIS AS WE HAVE NO NEED FOR THIS.
            loss = strategy._criterion(self.parser, mb, out)
            loss.backward()
            # gradient can be None for some head on multi-headed models
            self.reference_gradients = [
                p.grad.view(-1)
                if p.grad is not None
                else torch.zeros(p.numel(), device=strategy.device)
                for n, p in strategy.model.named_parameters()
            ]
            self.reference_gradients = torch.cat(self.reference_gradients)
            strategy.optimizer.zero_grad()

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """
        if len(self.buffers) > 0:
            current_gradients = [
                p.grad.view(-1)
                if p.grad is not None
                else torch.zeros(p.numel(), device=strategy.device)
                for n, p in strategy.model.named_parameters()
            ]
            current_gradients = torch.cat(current_gradients)

            assert (
                current_gradients.shape == self.reference_gradients.shape
            ), "Different model parameters in AGEM projection"

            dotg = torch.dot(current_gradients, self.reference_gradients)
            if dotg < 0:
                alpha2 = dotg / torch.dot(
                    self.reference_gradients, self.reference_gradients
                )
                grad_proj = (
                    current_gradients - self.reference_gradients * alpha2
                )

                count = 0
                for n, p in strategy.model.named_parameters():
                    n_param = p.numel()
                    if p.grad is not None:
                        p.grad.copy_(
                            grad_proj[count : count + n_param].view_as(p)
                        )
                    count += n_param

    def sample_from_memory(self):
        """
        Sample a minibatch from memory.
        Return a tuple of patterns (tensor), targets (tensor).
        """
        return next(self.buffer_dliter)

    @torch.no_grad()
    def update_memory(self, dataset, num_workers=0, **kwargs):
        """
        Update replay memory with patterns from current experience.
        """
        if num_workers > 0:
            warnings.warn(
                "Num workers > 0 is known to cause heavy" "slowdowns in AGEM."
            )
        removed_els = len(dataset) - self.patterns_per_experience
        if removed_els > 0:
            dataset, _ = random_split(
                dataset, [self.patterns_per_experience, removed_els]
            )
        self.buffers.append(dataset)
        persistent_workers = num_workers > 0
        self.buffer_dataloader = GroupBalancedInfiniteDataLoader(
            self.buffers,
            batch_size=self.sample_size // len(self.buffers),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
        )
        self.buffer_dliter = iter(self.buffer_dataloader)
