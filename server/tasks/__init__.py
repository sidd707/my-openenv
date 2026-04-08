"""Task registry — maps task_id to task class."""

from server.tasks.easy import EasyTask
from server.tasks.medium import MediumTask
from server.tasks.hard import HardTask

TASK_REGISTRY = {
    "easy": EasyTask,
    "medium": MediumTask,
    "hard": HardTask,
}

ALL_TASKS = [EasyTask(), MediumTask(), HardTask()]
