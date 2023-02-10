class TrainData:
    loss: float = None
    acc: float = None
    acc_all: float = None
    is_pretrain: bool = False

    def __str__(self):
        if self.loss is None or self.acc is None:
            return 'TrainData is not ready'
        if self.acc_all is None:
            return "loss: {:.3f}, acc: {:.3f}".format(self.loss, self.acc)
        return "loss: {:.3f}, acc: {:.3f}, acc all: {:.3f}".format(self.loss, self.acc, self.acc_all)
