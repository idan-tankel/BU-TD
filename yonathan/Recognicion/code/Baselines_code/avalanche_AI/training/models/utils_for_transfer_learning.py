import torch
from supp.utils import num_params as NP
from supp.models import ResNet
from supp.Parser import GetParser

def Get_learned_params_for_trnafer_learning(model, p):
    params = []
    num_params = 0
    is_break = False
    model_np = NP(model.bumodel.parameters())
    bit_list = [[] for _ in range(len(list(enumerate(model.bumodel.trunk.alllayers))))]
    for layer_id, layer in reversed(list(enumerate(model.bumodel.trunk.alllayers))):
        bit_list[layer_id] = [0 for _ in range(len(list(enumerate(layer))))]

        for block_id, block in reversed(list(enumerate(layer))):

            num_params += NP(block.parameters())
            if p * model_np < num_params:
                is_break =True
                break
            else:
                params.extend(list(block.parameters()))
                bit_list[layer_id][block_id] = 1
            if is_break:
                break
    return params, bit_list

parser = GetParser(0,0,model_type=ResNet)
print(Get_learned_params_for_trnafer_learning(parser.model, 0.5))
params = Get_learned_params_for_trnafer_learning(parser.model, 0.5)
print(params[1])
print(NP(params[0]))