from avalanche.models.dynamic_modules import MultiTaskModule
import torch

class MyMultiTaskModule(MultiTaskModule):
    def __init__(self, model):
        super(MyMultiTaskModule, self).__init__()
        self.model = model

        self.outs_to_struct = model.outs_to_struct
        self.inputs_to_struct = model.inputs_to_struct
        self.bumodel = model.bumodel # TODO change to FE.

    def forward_single_task(self, x: torch.Tensor, task_label: int ) -> torch.Tensor:
        return self.model(x, head =  task_label)

    def forward_and_out_to_struct(self,x, head = None):
        return self.model.forward_and_out_to_struct(x, head = head)


    def forward( self, x: torch.Tensor, task_labels: torch.Tensor) -> torch.Tensor:
        """compute the output given the input `x` and task labels.

        :param x:
        :param task_labels: task labels for each sample. if None, the
            computation will return all the possible outputs as a dictionary
            with task IDs as keys and the output of the corresponding task as
            output.
        :return:
        """
        if task_labels is None:
            return self.forward_all_tasks(x)

        if isinstance(task_labels, int): # TODO - SUPPORT MANY TASKS.
            # fast path. mini-batch is single task.
            return self.forward_single_task(x, task_labels)
        
        elif len(torch.unique(task_labels)) == 1:
            return self.forward_single_task(x, task_labels[0])
        else:
            unique_tasks = torch.unique(task_labels)
        '''
        out = torch.zeros(x[1].shape, device=x.device)
        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = x[task_mask]
            out_task = self.forward_single_task(x_task, task.item())
            assert len(out_task.shape) == 2, (
                "multi-head assumes mini-batches of 2 dimensions "
                "<batch, classes>"
            )
            n_labels_head = out_task.shape[1]
            out[task_mask, :n_labels_head] = out_task
        return out
        '''