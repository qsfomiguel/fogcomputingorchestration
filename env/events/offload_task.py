from env.fog.fog import FogNode, point_to_point_transmission_rate
from env.fog.task import task_communication_time
from env import configs as cfg 
from utils.custom_exceptions import InvalidValueError
from .events import Event 
from .task_arrival import Task_Arrival
from .start_transmitting import Start_Transmitting 

class Offload_Task(Event):
    def __init__(self, time, node, k, destination, concurr):
        super(Offload_Task, self).__init__(time, "Offload_Task")
        self.node = node 
        self.k = k 
        self.destination = destination 
        self.concurr = concurr 

        if not isinstance(node, FogNode) or k >= node.max_slice or k < 0 or not isinstance(destination, FogNode) or concurr < 1:
            raise InvalidValueError("Verify arguments for Ofload_Task")

    def execute(self, event_queue):
        bw = int(self.node.available_bandwidth()/self.concurr)
        if bw < cfg.NODE_BANDWIDTH_UNIT:
            return None 

        t = self.node.pop_task_to_send(self.k, self.time)

        if t == None:
            return None 

        arrival_time = self.time + task_communication_time(t.packet_size, point_to_point_transmission_rate(self.node.distances[self.destination.id], bw ))

        event_queue.add_event(Task_Arrival(arrival_time, self.destination, self.k, t))
        event_queue.add_event(Start_Transmitting(self.time, self.node, arrival_time, bw))

        return None 

