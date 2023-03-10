# this is an extention to the VitEmbedding class
from torch import nn
from transformers.models.vit.modeling_vit import ViTEmbeddings

class TaskHead(ViTEmbeddings):
    def __init__(self,original_embedder,ntasks,hidden_dim=768) -> None:
        super().__init__(original_embedder.config)
        self.task_head = nn.Linear(ntasks, hidden_dim,bias=False)
    
    # @property
    # def task_head(self):
    #     return nn.Linear(TaskHead.ntasks, TaskHead.hidden_dim)

    def forward(self,dicted_input,*args,**kwargs):
        """The forward call of the head
        Note! the pass of `args`, `kwargs` is for compatibility with the VitEmbedding class

        Args:
            dicted_input (_type_): The composed task and image input

        Returns:
            _type_: _description_
        """        
        task_input = dicted_input.task
        image_input = dicted_input.image
        task_enc = self.task_head(task_input).unsqueeze(1)
        image_enc = super(TaskHead,self).forward(pixel_values = image_input)
        return image_enc + task_enc