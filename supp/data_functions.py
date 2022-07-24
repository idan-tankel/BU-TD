import torch
from torch.utils.data import DataLoader

# TODO - MOVE IT TO ANOTHER FILE
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(seed=0)


def pause_image(fig=None):
    plt.draw()
    plt.show(block=False)
    if fig is None:
        fig = plt.gcf()
    fig.waitforbuttonpress()


def redraw_fig(fig):
    if fig is None:
        return
    fig.canvas.draw_idle()
    try:
        fig.canvas.flush_events()
    except NotImplementedError:
        pass


def preprocess(inputs: torch) -> torch:
    # Moves the tensor into the device, usually to the cuda.
    inputs = [inp.to(dev) for inp in inputs]
    return inputs


class WrappedDataLoader:
    def __init__(self, data_loader: DataLoader, preprocess_func: staticmethod) -> None:
        """
        :param data_loader: The data-loader
        :param preprocess_func: The preprocessing function to apply on the input.
        """
        self.data_loader = data_loader
        self.preprocess_func = preprocess_func

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        """
        for each input in the data loader,  preprocesses and returns it.
        """
        batches = iter(self.data_loader)
        for b in batches:
            if b is not None:
                yield self.preprocess_func(b)
