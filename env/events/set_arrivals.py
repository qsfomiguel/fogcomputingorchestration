from .events import Event 
from .task_arrival import Task_Arrival 

from env.fog.fog_tools import bernoulli_arrival
from env.fog.task import Task 

class Set_Arrivals(Event):
    def __init__(self, time, timestep, nodes, case):
        super(Set_Arrivals, self).__init__(time, "Set_Arrivals")
        self.timestep = timestep
        self.nodes = nodes 
        self.case = case 

    def execute(self, event_queue):
        for n in self.nodes:
            for i in range(n.max_slice):
                if bernoulli_arrival(self.case["arrivals"][i]):
                    t = Task(self.time+self.timestep, task_type = self.case["task_type"][i])
                    event = Task_Arrival(self.time+self.timestep, n, i, t)
                    event_queue.add_event(event)
        event_queue.add_event(Set_Arrivals(self.time+self.timestep, self.timestep, self.nodes, self.case))
        return None 