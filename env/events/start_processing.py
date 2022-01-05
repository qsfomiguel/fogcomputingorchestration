import numpy as np 

from env.fog.fog import FogNode 
from utils.custom_exceptions import InvalidValueError

from .events import Event
from .discard_task import Discard_Task
from .stop_processing import Stop_Processing 

class Start_Processing(Event):
    def __init__(self, time, node, k, w):
        super(Start_Processing, self).__init__(time, "Start_Processing")
        self.node = node 
        self.k = k 
        self.w = w

        if not isinstance(node, FogNode) or k >= node.max_slice or k < 0 or w <= 0:
            raise InvalidValueError("Verify arguments of Start_Processing creation")

    def execute(self, event_queue):
        tasks_under_processing, discarded = self.node.start_processing_in_slice(self.k, self.w, self.time)
        for task in tasks_under_processing:
            task_finish = self.time + task.task_remaining_processing_time()
            if task.exceeded_constraint(task_finish):
                event_queue.add_event(Discard_Task(max(task.constraint_time(), self.time), self.node, self.k, task))
            else:
                event_queue.add_event(Stop_Processing(task_finish, self.node, self.k, task))

        for task in discarded:
            event_queue.add_event(Discard_Task(max(task.constraint_time(), self.time), self.node, self.k, task))
        
        return None 