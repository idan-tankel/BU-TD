from avalanche.training.plugins.clock import Clock

class EpochClock(Clock):
    """Counter for strategy events.

    WARNING: Clock needs to be the last plugin, otherwise counters will be
    wrong for plugins called after it.
    """

    def __init__(self, max_epoch, pretrained_model):
        """Init."""
        super().__init__()
        # train

        self.max_epochs = max_epoch
        self.pretrained_model = pretrained_model
        """

        """
        self.train_iterations = 0
        """ Total number of training iterations. """

        self.train_exp_counter = 1 if pretrained_model else 0
        """ Number of past training experiences. """

        self.train_exp_epochs = 0
        """ Number of training epochs for the current experience. """

        self.train_exp_iterations = 0
        """ Number of training iterations for the current experience. """

        self.train_epoch_iterations = 0
        """ Number of iterations for the current epoch. """

        self.total_iterations = 0
        """ Total number of iterations in training and eval mode. """

        self.just_initialized = True

    def before_training_epoch(self, strategy, **kwargs):
        self.train_epoch_iterations = 0

    def before_training_exp(self, strategy, **kwargs):
        pass

    def after_training_iteration(self, strategy, **kwargs):
        self.train_epoch_iterations += 1
        self.train_exp_iterations += 1
        self.train_iterations += 1
        self.total_iterations += 1
        self.just_initialized = False

    def after_training_epoch(self, strategy, **kwargs):
        self.train_exp_epochs += 1

    def after_training_exp(self, strategy, **kwargs):
        if self.train_exp_epochs == self.max_epochs:
            self.train_exp_counter += 1
            self.train_exp_epochs = 0

    def after_eval_iteration(self, strategy, **kwargs):
        self.total_iterations += 1