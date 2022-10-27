from avalanche.training.plugins import AGEMPlugin
import warnings
import torch
from avalanche.models import avalanche_forward
from torch.utils.data import random_split
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.data_loader import ( GroupBalancedInfiniteDataLoader,)
from supp.utils import preprocess
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

class IL(SupervisedPlugin):
    """Average Gradient Episodic Memory Plugin.

    AGEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous experiences. If the dot product
    between the current gradient and the (average) gradient of a randomly
    sampled set of memory examples is negative, the gradient is projected.
    This plugin does not use task identities.
    """

    def __init__(self, parser, old_data,  patterns_per_experience: int, sample_size: int):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param sample_size: number of patterns in memory sample when computing
            reference gradient.
        """

        self.patterns_per_experience = int(patterns_per_experience)
        self.prev_data = old_data
        self.memory_x, self.memory_y, self.memory_tid = [[]], [[]], [[]]
        self.buffers = []
        self.sample_size = sample_size
        self.parser = parser
        if parser.pretrained_model and self.prev_data != None:
          self.update_memory(self.prev_data, num_workers=0)


    def after_training_exp(self, strategy, **kwargs):
        """Update replay memory with patterns from current experience."""
        if strategy.EpochClock.train_exp_epochs == strategy.EpochClock.max_epochs:
            print("Update memory")
            self.update_memory(strategy.experience.dataset, **kwargs)

    def before_backward(self, strategy, **kwargs):
        """
        Compute reference gradient on memory sample.
        """
        if len(self.buffers) > 0:
            strategy.model.train()
            strategy.optimizer.zero_grad()
            mb = self.sample_from_memory()
            mb = preprocess(mb, device=strategy.device)
            out = strategy.model(mb) # TODO - SUPPORT ANY TASK ID OR OMIT THIS AS WE HAVE NO NEED FOR THIS.
            loss_old_task = strategy._criterion(self.parser, mb, out)
            strategy.loss += loss_old_task
           # strategy.loss /= 2

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
            warnings.warn("Num workers > 0 is known to cause heavy" "slowdowns in IL.")
        removed_els = len(dataset) - self.patterns_per_experience
        if removed_els > 0:
            dataset, _ = random_split(dataset, [self.patterns_per_experience, removed_els])
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