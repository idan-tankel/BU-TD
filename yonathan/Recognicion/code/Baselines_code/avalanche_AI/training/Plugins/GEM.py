import numpy as np
import quadprog
import torch
from torch.utils.data import DataLoader
from supp.utils import preprocess
from avalanche.models import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class GEMPlugin(SupervisedPlugin):
    """
    Gradient Episodic Memory Plugin.
    GEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous experiences. The gradient on
    the current minibatch is projected so that the dot product with all the
    reference gradients of previous tasks remains positive.
    This plugin does not use task identities.
    """

    def __init__(self,parser,old_data, patterns_per_experience: int, memory_strength: float):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param memory_strength: offset to add to the projection direction
            in order to favour backward transfer (gamma in original paper).
        """

        super().__init__()

        self.patterns_per_experience = int(patterns_per_experience)
        self.memory_strength = memory_strength

        self.old_data = old_data

        self.memory_x, self.memory_y, self.memory_tid = [[]], [[]], [[]]

        self.G = None

        self.parser = parser

        if self.parser.pretrained_model:
            self.update_memory(
               old_data,
                0,
                64,
            )

    def before_training_iteration(self, strategy, **kwargs):
        """
        Compute gradient constraints on previous memory samples from all
        experiences.
        """

        if strategy.EpochClock.train_exp_counter > 0: # strategy.clock.train_exp_counter > 0:
            G = []
            strategy.model.train()
            for t in range(1): # strategy.clock.train_exp_counter
                strategy.model.train()
                strategy.optimizer.zero_grad()
                for input in self.memory_x[t] :
                    xref = preprocess(input, strategy.device)
                    yref = preprocess(input, strategy.device)
                    outs = strategy.model(xref, head = 0)
                    loss = strategy._criterion(self.parser, xref, outs)
                    loss.backward()

                G.append(
                    torch.cat(
                        [
                            p.grad.flatten()
                            if p.grad is not None
                            else torch.zeros(p.numel(), device=strategy.device)
                            for p in strategy.bumodel.model.parameters()
                        ],
                        dim=0,
                    )
                )

            self.G = torch.stack(G)  # (experiences, parameters)

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """
        if strategy.EpochClock.train_exp_counter > 0: # strategy.clock.train_exp_counter > 0:
           # print("Use rehersal")
            g = torch.cat(
                [
                    p.grad.flatten()
                    if p.grad is not None
                    else torch.zeros(p.numel(), device=strategy.device)
                    for p in strategy.model.parameters()
                ],
                dim=0,
            )

            to_project = (torch.mv(self.G, g) < 0).any()
        else:
            to_project = False

        if to_project:
            v_star = self.solve_quadprog(g).to(strategy.device)

            num_pars = 0  # reshape v_star into the parameter matrices
            for p in strategy.model.parameters():
                curr_pars = p.numel()
                if p.grad is not None:
                    p.grad.copy_(
                        v_star[num_pars: num_pars + curr_pars].view(p.size())
                    )
                num_pars += curr_pars

            assert num_pars == v_star.numel(), "Error in projecting gradient"

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience
        """
        if strategy.EpochClock.train_exp_epochs == strategy.EpochClock.max_epochs:
            self.update_memory(
                strategy.experience.dataset,
                strategy.clock.train_exp_counter,
                strategy.train_mb_size,
            )

    @torch.no_grad()
    def update_memory(self, dataset, t, batch_size):
        """
        Update replay memory with patterns from current experience.
        """
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") \
            else None
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                collate_fn=collate_fn)
        tot = 0
        for mbatch in dataloader:
            x = mbatch
            y = mbatch
            tid = 0

            if tot + x[0].size(0) <= self.patterns_per_experience:
                    self.memory_x[0].append([inp.clone() for inp in x])
                    self.memory_y[0].append([inp.clone() for inp in y])
                    self.memory_tid[0] = 0

            else:
                break
            tot += x[0].size(0)
     #   print(self.memory_x)

    def solve_quadprog(self, g):
        """
        Solve quadratic programming with current gradient g and
        gradients matrix on previous tasks G.
        Taken from original code:
        https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py
        """

        memories_np = self.G.cpu().double().numpy()
        gradient_np = g.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + self.memory_strength
        v = quadprog.solve_qp(P, q, G, h)[0]
        v_star = np.dot(v, memories_np) + gradient_np

        return torch.from_numpy(v_star).float()