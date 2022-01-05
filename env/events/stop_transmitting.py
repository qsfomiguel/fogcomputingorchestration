from env.fog.fog import FogNode
from utils.custom_exceptions import InvalidValueError

from .events import Event 

class Stop_Transmitting(Event):

    def __init__(self, time, node, bw):
        super(Stop_Transmitting, self).__init__(time, "Stop_Transmitting")
        self.node = node 
        self.bw = bw 

        if not isinstance(node, FogNode):
            raise InvalidValueError("Verify arguments of Stop_Transmitting creation")   

    def execute(self, event_queue):
        self.node.finished_transmitting(self.bw)
        return None 