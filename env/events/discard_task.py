from env.fog.fog import FogNode 
from env.fog.task import Task

from utils.custom_exceptions import InvalidValueError

from .events import Event 

class Discard_Task(Event):
    def __init__(self, time, node, k, task):
        super(Discard_Task, self).__init__(time, "Discard_Task")
        self.node = node 
        self.k = k 
        self.task = task 

        if not isinstance(node, FogNode) or k >= node.max_slice or k < 0 or not isinstance(task, Task):
            raise InvalidValueError("Verify arguments of Discard_task creation")

    def execute(self, event_queue):
        return self.node.remove_task_on_slice(self.k, self.task)


