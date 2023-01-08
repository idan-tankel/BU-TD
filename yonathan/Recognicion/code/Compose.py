import torch

import os

from pathlib import Path

from torch import dtype

from training.Data.Data_params import Flag, DsType
from training.Data.Get_dataset import get_dataset_for_spatial_relations
from training.Data.Parser import GetParser
from training.Modules.Create_Models import create_model
from training.Modules.Models import *

import torch.nn as nn
import argparse
import numpy as np

from training.Utils import preprocess, tuple_direction_to_index, load_model

class CompositiveModel(nn.Module):
    def __init__(self, opts:argparse, model:nn):
        super(CompositiveModel, self).__init__()
        self.opts = opts
        self.model = model
        self.flag = opts.model_flag
        self.edge_class = 47 if self.flag is DsType.Emnist else 10
        self.inputs_to_struct = opts.inputs_to_struct
        self.ndirections = opts.ndirections

    def forward(self, batch):
        self.model.eval()
        pred = self.model(batch)[-1].argmax(dim=1)
        return pred

    def Create_new_flag(self, pred, direction, char):
        try:
          B = pred.size(0)
        except AttributeError:
          B = char.size(0)
        direction_index, _ = tuple_direction_to_index(self.opts.num_x_axis, self.opts.num_y_axis, direction, self.opts.ndirections, task_id = 0) # TODO - CHANGE.
        task_type_ohe = torch.nn.functional.one_hot(torch.zeros((B), dtype=torch.long), 1).cuda()
        # Getting the direction embedding, telling which direction we solve now.
        direction_type_ohe = torch.nn.functional.one_hot(torch.ones((B), dtype=torch.long) * direction_index, self.opts.ndirections).cuda()
        # Getting the character embedding, which character we query about.
        char_type_one = torch.nn.functional.one_hot(pred, self.edge_class) if pred is not None else char
        # Concatenating all three flags into one flag.
     #   print(direction_type_ohe.shape, task_type_ohe.shape, char_type_one.shape)

        flag = torch.concat([direction_type_ohe, task_type_ohe, char_type_one], dim=1).float()
        return flag

    def compose_tasks(self, batch, directions):
        preds = []
        for direction_idx, direction in enumerate(directions):
            load_model(model, parser.results_dir, f'Model_{direction}_single_base/BUTDModel_best_direction=[{direction}].pt')
            if direction_idx > 0:
                new_flag = self.Create_new_flag(pred, direction, None)
                batch.flag = new_flag
            else:
                char = batch.flag[:, self.opts.ndirections + self.opts.ntasks:]
                new_flag = self.Create_new_flag(None, direction, char)
                batch.flag = new_flag
            pred = self(batch)
            preds.append(pred.clone())
            mask = (pred != self.edge_class)
            pred = pred * mask


        return preds

    def compose_tasks_full_data_loader(self,dl, directions):
        acc = 0.0
        for batch in dl:
            batch = preprocess(batch, self.opts.device)
            batch = self.inputs_to_struct(batch)
            pred = self.compose_tasks(batch, directions)
            pred = self.final_prediction(pred)
            acc += (pred == batch.label_task).float().sum() / pred.size(0)

        return acc / len(dl)

    def final_prediction(self, preds):
        all_preds = torch.stack(preds,dim = -1)
        mask = (all_preds == self.edge_class).sum(dim=-1) > 0
        pred = preds[-1] *( ~ mask) + self.edge_class * mask
        return pred

ds_type = DsType.Fashionmnist

parser = GetParser(model_flag=Flag.CL, ds_type=ds_type, model_type=BUTDModel)
model = create_model(parser)
load_model(model,parser.results_dir, 'Model_(1, 0)_single_base/BUTDModel_best_direction=[(1, 0)].pt')
comp_model = CompositiveModel(opts=parser, model=model)
project_path = Path(__file__).parents[1]
data_path = os.path.join(project_path, f'data/{str(ds_type)}/samples/(3,3)_Image_Matrix')
#
DataLoaders = get_dataset_for_spatial_relations(parser, data_path, lang_idx=0, direction_tuple=[(-1, -1)])
print(comp_model.compose_tasks_full_data_loader(DataLoaders['test_dl'],[ (-1,0), (0,-1), ]))