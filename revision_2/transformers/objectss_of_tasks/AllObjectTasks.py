from typing import List

from transformers.objectss_of_tasks.InstructionsObjectTask import InstructionsObjectTask
from transformers.objectss_of_tasks.ObjectTaskInterface import ObjectTaskInterface
from transformers.objectss_of_tasks.RightOfInstructionsObjectTask import RightOfInstructionsObjectTask
from transformers.objectss_of_tasks.RightOfObjectTask import RightOfObjectTask


class AllObjectTasks:
    all_object_tasks: List[ObjectTaskInterface] = []

    @classmethod
    def create_all_object_tasks(cls):
        if not cls.all_object_tasks:
            cls.all_object_tasks.append(RightOfObjectTask())
            cls.all_object_tasks.append(RightOfInstructionsObjectTask())
            cls.all_object_tasks.append(InstructionsObjectTask())

    @classmethod
    def get_wanted_object_task(cls, task_name: str, task_dataset: str):
        AllObjectTasks.create_all_object_tasks()
        for object_task in cls.all_object_tasks:
            if object_task.validate(task_name, task_dataset):
                return object_task
        return None
