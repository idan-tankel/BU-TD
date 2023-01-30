"""
Here we define and perform  a composition of tasks sequentially as BU-TD model allows us.
"""
import os
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader
from training.Data.Get_dataset import get_dataset_for_spatial_relations
from training.Data.Parser import GetParser
from training.Data.Structs import Task_to_struct
from training.Modules.Create_Models import create_model
from training.Modules.Models import *
from training.Utils import preprocess, tuple_direction_to_index


class ComposeModel(nn.Module):
    """
    Here we define composite model, to solve all spatial relations using sequential tasks.
    """

    def __init__(self, opts: argparse, butd_model: BUTDModel):
        super(ComposeModel, self).__init__()
        self.opts = opts  # The model opts.
        self.model: BUTDModel = butd_model  # The model.
        self.flag = opts.model_flag  # The model flag.
        # Compute the edge class, needed for early stopping.
        self.edge_class = 47 if self.flag is DsType.Emnist else 10
        # Input to struct object.
        self.inputs_to_struct = opts.inputs_to_struct
        # The number of directions.
        self.ndirections = opts.ndirections

    def forward(self, batch: inputs_to_struct) -> Tensor:
        """
        Here we just forward the model and return the prediction.
        Args:
            batch: The input

        Returns: The prediction.

        """
        self.model.eval()
        prediction = self.model.forward_and_out_to_struct(batch)
        prediction = prediction.classifier.argmax(dim=1)
        return prediction

    def Create_new_flag(self, prediction: Optional[Tensor], direction: tuple, char: Optional[Tensor],
                        direction_id: int) -> Tensor:
        """
        Create the new flag.
        Args:
            prediction: The prediction, first cycle None.
            direction: The task.
            char: The characters, needed for first cycle only as no prediction is computed.
            direction_id: The task id.

        Returns: The New flag instruction for next phase.

        """

        if direction_id == 0:
            B = char.size(0)
        else:
            B = prediction.size(0)
        direction_index, _ = tuple_direction_to_index(self.opts.num_x_axis, self.opts.num_y_axis, direction,
                                                      self.opts.ndirections)
        task_type_ohe = torch.nn.functional.one_hot(torch.zeros(B, dtype=torch.long), 1).cuda()
        # Getting the task embedding, telling which task we solve now.
        direction_type_ohe = torch.nn.functional.one_hot(torch.ones(B, dtype=torch.long) * direction_index,
                                                         self.opts.ndirections).cuda()
        # Getting the character embedding, which character we query about.
        char_type_one = torch.nn.functional.one_hot(prediction, self.edge_class) if prediction is not None else char
        # Concatenating all three flags into one flag.
        flag = torch.concat([direction_type_ohe, task_type_ohe, char_type_one], dim=1).float()
        return flag

    def compose_tasks(self, batch: inputs_to_struct, directions: list[tuple]) -> list[Tensor]:
        """
        Args:
            batch: The batch input.
            directions: The list of all directions.

        Returns: The final prediction list(including all time steps).

        """
        preds = []
        prediction = None
        for direction_idx, direction in enumerate(directions):
            # TODO - THROW AWAY.
            #            load_model(model, opts.results_dir,
            #             f'Model_{direction}_single_base/BUTDModel_best_direction=[{direction}].pt')

            if direction_idx == 0:
                char = batch.flags[:, self.opts.ndirections + self.opts.ntasks:]
                new_flag = self.Create_new_flag(None, direction, char, direction_id=direction_idx)
                batch.flags = new_flag
            else:
                new_flag = self.Create_new_flag(prediction, direction, None, direction_id=direction_idx)
                batch.flags = new_flag
            prediction = self(batch)  # The prediction.
            preds.append(prediction)  # Add the prediction
            mask = (prediction != self.edge_class)  # The masked prediction to avoid spatial relation of 'border'
            prediction = prediction * mask

        return preds

    def compose_tasks_full_data_loader(self, dl: DataLoader, directions: list[tuple]) -> torch.float:
        """
        Compose all tasks on full data-loader.
        Args:
            dl: The data-loader.
            directions: The list of directions.

        Returns: Full accuracy.

        """
        acc = 0.0
        for batch in dl:
            batch = preprocess(batch, self.opts.device)
            batch = self.inputs_to_struct(batch)
            prediction = self.compose_tasks(batch, directions)
            prediction = self.final_prediction(prediction)
            acc += (prediction == batch.label_task).float().sum() / prediction.size(0)

        return acc / len(dl)

    def final_prediction(self, preds: list[Tensor]) -> Tensor:
        """
        Compute final prediction.
        Args:
            preds: all time steps predictions.

        Returns:

        """
        all_preds = torch.stack(preds, dim=-1)
        mask = (all_preds != self.edge_class).sum(dim=-1) > 0
        prediction = preds[-1] * mask + self.edge_class * mask
        return prediction


ds_type = DsType.Fashionmnist

parser = GetParser(model_flag=Flag.CL, ds_type=ds_type)
model: BUTDModel = create_model(parser)
# load_model(model, opts.results_dir, 'Model_(1, 0)_single_base/BUTDModel_best_direction=[(1, 0)].pt')
comp_model = ComposeModel(opts=parser, butd_model=model)
project_path = Path(__file__).parents[3]
data_path = os.path.join(project_path, f'data/{str(ds_type)}/samples/(3,3)_Image_Matrix')
#
DataLoaders = get_dataset_for_spatial_relations(parser, data_path, task=[Task_to_struct(task=0,
                                                                                        direction=(-1, 0))])

print(comp_model.compose_tasks_full_data_loader(DataLoaders['test_dl'], [(0, 1), (-1, 0)]))
