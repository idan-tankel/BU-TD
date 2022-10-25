from types import SimpleNamespace

from torch import nn

from .Heads import OccurrenceHead, MultiLabelHead, ImageHead, MultiLabelHeadOnlyTask


class BUModel(nn.Module):

    def __init__(self, opts):
        super(BUModel, self).__init__()
        bu_shared = ResNetLatSharedBase(opts)
        self.trunk = ResNetLatShared(opts, bu_shared)

    def forward(self, inputs):
        trunk_out, laterals_out = self.trunk(inputs)
        return trunk_out, laterals_out


class TDModel(nn.Module):

    def __init__(self, opts):
        super(TDModel, self).__init__()
        self.trunk = ResNetTDLat(opts)

    def forward(self, bu_out, flag, laterals_in):
        # bu_out, flag, laterals_in = inputs
        td_outs = self.trunk(bu_out, flag, laterals_in)
        return td_outs


class BUTDModel(nn.Module):

    def forward(self, inputs):
        # Here we got all the flags/features - check where input is more than 1
        samples = self.inputs_to_struct(inputs)
        images = samples.image  # TODO torch.Size([10, 3, 224, 448])
        # TODO Those flags - we need to change - to be more than 1(on inference time)
        flags = samples.flag
        # flags shape is (10,13=7+6)
        model_inputs = [images, flags, None]
        bu_out, bu_laterals_out = self.bumodel1(
            x=images, flags=flags, laterals_in=None)
        # this is forward pass for bu model
        # Although the recipe for forward pass needs to be defined within this function,
        #  one should call the Module instance afterwards instead of this
        # since the former takes care of running the registered hooks while the latter silently ignores them
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        if self.use_bu1_loss:
            occurence_out = self.occhead(bu_out)
        else:
            occurence_out = None
        model_inputs = [bu_out, flags]
        if self.use_lateral_butd:
            model_inputs += [bu_laterals_out]
        else:
            model_inputs += [None]
        td_outs = self.tdmodel(bu_out=bu_out, flag=flags,
                               laterals_in=bu_laterals_out)
        td_out, td_laterals_out, *td_rest = td_outs
        if self.use_td_loss:
            td_head_out = self.imagehead(td_out)
        model_inputs = [images, flags]
        if self.use_lateral_tdbu:
            bu2_out, bu2_laterals_out = self.bumodel2(
                x=images, flags=flags, laterals_in=td_laterals_out)
        else:
            # when not using laterals we only use td_out as a lateral
            bu2_out, bu2_laterals_out = self.bumodel2(
                x=images, flags=flags, laterals_in=[td_out])
            # TODO - This output - we need to check diff from all flags
        # TODO - check input and output
        task_out = self.taskhead(bu2_out, flags)
        if False:  # TODO itsik - change this to be from config - itsik_flag
            self.zero_non_task_layers(flags, task_out)

        outs = [occurence_out, task_out, bu_out, bu2_out]
        if self.use_td_loss:
            outs += [td_head_out]
        if self.tdmodel.trunk.use_td_flag:
            td_top_embed, td_top = td_rest
            outs += [td_top_embed]

        # TODO itsik - multi - freeze all that not in task, unfreeze all that in task
        # How ever we cant do it - because in each batch we can have different flags - and the loss is fr all of them - and we can't seperate them
        # for layer in self.taskhead.layers:
        #     layer.weight.requires_grad = True # check if we need to unfreeze
        #     layer.bias.requires_grad = True # check if we need to unfreeze - in flag
        return outs

    def zero_non_task_layers(self, flags, task_out):
        loss_weight_by_task = activated_tasks(task_out.shape[2], flags)
        for index_layer in list(range(task_out.shape[2])):
            if loss_weight_by_task is not None and type(loss_weight_by_task) is list and len(
                    loss_weight_by_task) > index_layer:
                task_out[:, :, index_layer] = task_out[:, :,
                                                       index_layer] * loss_weight_by_task[index_layer]

    def outs_to_struct(self, outs):
        """
        outs_to_struct 

        Args:
            outs (List): The unstructured model outs

        Returns:
            `SimpleNamespace`: The structured model outs
        """        
        occurence_out, task_out, bu_out, bu2_out, *rest = outs
        outs_ns = SimpleNamespace(
            occurence=occurence_out, task=task_out, bu=bu_out, bu2=bu2_out)
        if self.use_td_loss:
            td_head_out, *rest = rest
            outs_ns.td_head = td_head_out
        if self.tdmodel.trunk.use_td_flag:
            td_top_embed = rest[0]
            outs_ns.td_top_embed = td_top_embed
        return outs_ns


class BUTDModelShared(BUTDModel):

    def __init__(self, opts):
        super(BUTDModelShared, self).__init__()
        self.use_bu1_loss = opts.Losses.use_bu1_loss
        if self.use_bu1_loss:
            self.occhead = OccurrenceHead(opts)
        # here should use instead MultiLabelHeadOnlyTask
        self.taskhead = MultiLabelHeadOnlyTask(opts)
        # self.taskhead = MultiLabelHead(opts) # here should use instead MultiLabelHeadOnlyTask
        self.use_td_loss = opts.Losses.use_td_loss
        if self.use_td_loss:
            self.imagehead = ImageHead(opts)
        bu_shared = ResNetLatSharedBase(opts)
        self.bumodel1 = ResNetLatShared(opts, bu_shared)
        opts.use_top_flag = opts.use_bu2_flag
        self.bumodel2 = ResNetLatShared(opts, bu_shared)

        pre_top_shape = bu_shared.inshapes[-2][-1]
        opts.avg_pool_size = tuple(pre_top_shape[1:].tolist())
        #        opts.avg_pool_size = (7,14)
        self.tdmodel = TDModel(opts)
        self.use_bu1_flag = opts.use_bu1_flag
        self.use_lateral_butd = opts.Models.use_lateral_butd
        self.use_lateral_tdbu = opts.Models.use_lateral_tdbu
        self.inputs_to_struct = opts.inputs_to_struct


class BUTDModelDuplicate(BUTDModel):
    def __init__(self, opts):
        super(BUTDModelDuplicate, self).__init__()
        self.occhead = OccurrenceHead(opts)
        self.taskhead = MultiLabelHead(opts)
        self.use_td_loss = opts.use_td_loss
        if self.use_td_loss:
            self.imagehead = ImageHead(opts)
        bu_shared = ResNetLatSharedBase(opts)
        self.bumodel1 = ResNetLatShared(opts, bu_shared)
        # TODO: fix this as this will not work in duplicate
        opts.use_top_flag = opts.use_bu2_flag
        self.bumodel2 = self.bumodel1

        pre_top_shape = bu_shared.inshapes[-2][-1]
        opts.avg_pool_size = tuple(pre_top_shape[1:].tolist())
        #        opts.avg_pool_size = (7,14)
        self.tdmodel = TDModel(opts)
        self.use_bu1_flag = opts.use_bu1_flag
        self.use_lateral_butd = opts.use_lateral_butd
        self.use_lateral_tdbu = opts.use_lateral_tdbu
        self.inputs_to_struct = opts.inputs_to_struct


class BUTDModelSeparate(BUTDModel):
    def __init__(self, opts):
        super(BUTDModelSeparate, self).__init__()
        self.occhead = OccurrenceHead(opts)
        self.taskhead = MultiLabelHead(opts)
        self.use_td_loss = opts.use_td_loss
        if self.use_td_loss:
            self.imagehead = ImageHead(opts)
        bu_shared1 = ResNetLatSharedBase(opts)
        self.bumodel1 = ResNetLatShared(opts, bu_shared1)
        opts.use_top_flag = opts.use_bu2_flag
        bu_shared2 = ResNetLatSharedBase(opts)
        self.bumodel2 = ResNetLatShared(opts, bu_shared2)

        pre_top_shape = bu_shared1.inshapes[-2][-1]
        opts.avg_pool_size = tuple(pre_top_shape[1:].tolist())
        #        opts.avg_pool_size = (7,14)
        self.tdmodel = TDModel(opts)
        self.use_bu1_flag = opts.use_bu1_flag
        self.use_lateral_butd = opts.use_lateral_butd
        self.use_lateral_tdbu = opts.use_lateral_tdbu
        self.inputs_to_struct = opts.inputs_to_struct


class BUModelSimple(nn.Module):

    def __init__(self, opts):
        super(BUModelSimple, self).__init__()
        self.occhead = OccurrenceHead(opts)
        self.taskhead = MultiLabelHead(opts)
        self.bumodel = BUModel(opts)
        pre_top_shape = self.bumodel.trunk.inshapes[-2][-1]
        opts.avg_pool_size = tuple(pre_top_shape[1:].tolist())
        self.inputs_to_struct = opts.inputs_to_struct

    def forward(self, inputs):
        samples = self.inputs_to_struct(inputs)
        images = samples.image
        flags = samples.flag
        model_inputs = [images, flags, None]
        bu_out, _ = self.bumodel(model_inputs)
        occurence_out = self.occhead(bu_out)
        task_out = self.taskhead(bu_out)
        return occurence_out, task_out, bu_out

    def outs_to_struct(self, outs):
        occurence_out, task_out, bu_out = outs
        outs_ns = SimpleNamespace(
            occurence=occurence_out, task=task_out, bu=bu_out)
        return outs_ns


##########################################
#    Baseline functions - not really used
##########################################
class BUModelRawOcc(nn.Module):

    def __init__(self, opts):
        super(BUModelRawOcc, self).__init__()
        self.occhead = OccurrenceHead(opts)
        self.bumodel = BUModelRaw(self.occhead, opts)

    def forward(self, inputs):
        images, labels, flags, segs, y_adjs, ids = inputs
        model_inputs = images
        task_out = self.bumodel(model_inputs)
        return task_out


class BUModelRaw(nn.Module):

    def __init__(self, head, opts):
        super(BUModelRaw, self).__init__()
        self.trunk = ResNet(BasicBlock, opts)
        self.head = head

    def forward(self, inputs):
        x = self.trunk(inputs)
        x = self.head(x)
        return x




