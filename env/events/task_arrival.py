from env.fog.fog import FogNode
from env.fog.task import Task
from utils.custom_exceptions import InvalidValueError

from .events import Event 

class Task_Arrival(Event):
    def __init__(self, time, node, k, task):
        super(Task_Arrival, self).__init__(time, "Task_Arrival")
        self.node = node 
        self.k = k 
        self.task = task 

        if not isinstance(node, FogNode) or k >= node.max_slice or k < 0 or not isinstance(task, Task):
            raise InvalidValueError("Verify arguments of Task_Arrival creation")
        if time < task.task_time():
            raise InvalidValueError("Cannot recieve a task from the future")
    
    def execute(self, event_queue):
        return self.node.add_task_on_slice(self.k, self.task)

def is_arrival_on_slice(ev, node, k):
    return (ev.classtype == "Task_Arrival" and ev.node == node and ev.k == k)

def is_offload_arrival_event(ev):
    return (ev.classtype == "Task_Arrival" and ev.task.task_time() < ev.time)