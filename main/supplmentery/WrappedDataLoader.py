# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from torch.utils.data import DataLoader

from torch.utils.data import DataLoader


class WrappedDataLoader:
    """ Wrapper class for `torch.utils.data.DataLoader`
    """

    def __init__(self, dl: DataLoader, device):
        """__init__ _summary_

        Args:
            dl (torch.utils.data.DataLoader): The torch DataLoader itself
            func (_type_): preprocessing function. In our case - the called function would be a function to transfer all the inputs to a specific device
        """
        self.dl = dl
        self.device = device

    def __len__(self):
        """__len__ _summary_

        Returns:
            int: the length based on the dl
        """        
        return len(self.dl)

    def __iter__(self):
        """__iter__ iterate over the dataset

        Yields:
            _type_: return the preprocessing function on each of the inpts
        """
        batches = iter(self.dl)
        for b in batches:
            # if the batch is None do nothing (iterate over the next one)
            if b is not None:
                yield ([item.to(self.device) for item in b])
