from .events import Event 
from .stop_transmitting import Stop_Transmitting 

class Start_Transmitting(Event):

    def __init__(self, time, node, arrival_time, bw):
        super(Start_Transmitting, self).__init__(time, "Start_Transmitting")
        self.node = node 
        self.arrival_time = arrival_time 
        self.bw = bw 

    def execute(self, event_queue):
        self.node.start_transmitting(self.bw)
        event_queue.add_event(Stop_Transmitting(self.arrival_time, self.node, self.bw))
        return None 