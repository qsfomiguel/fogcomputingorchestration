from env.fog.fog import FogNode 
from env.fog.task import Task 
from utils.custom_exceptions import InvalidValueError

from .events import Event 

class Stop_Processing(Event):

    def __init__(self, time, node, k, task):
        super(Stop_Processing, self).__init__(time, "Stop_Processing")
        self.node = node 
        self.k = k
        self.task = task 

        if not isinstance(node, FogNode) or k >= node.max_slice or k < 0 or not isinstance(task, Task):
            raise InvalidValueError("Verify arguments of Stop_Processing creation")

    def execute(self, event_queue):
        self.node.stop_processing_in_slice(self.k, self.task, self.time)
        return self.node.remove_task_on_slice(self.k, self.task) if self.task.is_completed() else None 