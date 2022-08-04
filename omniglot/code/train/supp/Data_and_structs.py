import torch
import torch.nn as nn

class outs_to_struct_base:
    """
    Struct transforming the model output list to struct.
    """

    def __init__(self, model: nn.Module) -> None:
        """
        :param model:Containing the flags to create the model according to.
        """
        self.use_td_loss = model.use_td_loss
        self.use_td_flag = model.use_td_flag
        self.occurrence_out = None
        self.task_out = None
        self.bu_out = None
        self.bu2_out = None
        self.td_head_out = None
        self.td_top_embe = None

    def __call__(self, outs: list[torch]) -> object:
        """
        :param outs: List of all model streams output.
        :return: Struct instance.
        """
        occurrence_out, task_out, bu_out, bu2_out, *rest = outs
        self.occurrence_out = occurrence_out
        self.task = task_out
        self.bu = bu_out
        self.bu2 = bu2_out
        if self.use_td_loss:
            td_head_out, *rest = rest
            self.td_head = td_head_out
        if self.use_td_flag:
            td_top_embed = rest[0]
            self.td_top_embed = td_top_embed
        return self

class cyclicOutToStruct:
    def __init__(self,opts):
        self.outs = [ ]
        self.stages = opts.stages
        self.use_td_loss = opts.model.module.use_td_loss
        self.use_td_flag = opts.model.module.use_td_flag
        for  _ in self.stages:
         out = outs_to_struct_base(opts)
         self.outs.append(out)

    def __call__(self, model_outs: list[torch]) -> object:
        outs = {}
        for idx, stage in enumerate(self.stages):
            out = self.outs[idx](model_outs[stage])
            outs[stage] = out
        return outs

class cyclic_inputs_to_strcut:
    def __init__(self,inputs,stage):
        img, label_all, label_existence,flag, flag_stage_1, flag_stage_2, flag_stage_3 ,label_task_stage_1, label_task_stage_2, label_task_stage_3  = inputs
        self.image = img
        self.label_all = label_all
        self.label_existence = label_existence
        self.general_flag = flag
        if stage == 0:
         self.label_task = label_task_stage_1
         self.flag = flag_stage_1
        if stage == 1:
         self.label_task = label_task_stage_2
         self.flag = flag_stage_2
        if stage == 2:
         self.label_task = label_task_stage_3
         self.flag = flag_stage_3