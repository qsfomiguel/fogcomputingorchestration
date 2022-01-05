from collections import deque 
from abc import ABC, abstractmethod 

from utils.custom_exceptions import InvalidStateError, InvalidValueError

class Event(ABC):
    def __init__(self, time, classtype=None):
        if time < 0:
            raise InvalidValueError("an event execution time cannot be negative")
        self.time = time 
        self.classtype= classtype

    @abstractmethod 
    def execute(self, eventqueue = None):
    
        pass

class Event_Queue(object):
    def __init__(self):
        self.q = deque()
        self.current_time = 0 
    
    def __str__(self):
        return "evq"+ str(len(self.q))
    
    def add_event(self, e:Event):
        if not isinstance(e, Event):
            return 
        if e.time < self.current_time:
            raise InvalidStateError("Tried to insert an event on the past")
        if len(self.q) == 0:
            self.q.append(e)
            return 
        for ev in self.q:
            if ev.time > e.time:
                continue 
            ind = self.q.index(ev)
            self.q.insert(ind,e)
            return 
        self.q.append(e)
    
    def pop_event(self):
        if self.is_empty():
            return None 
        ev = self.q.pop()
        self.current_time = ev.time 
        return ev 
    
    def queue_size(self):
        return len(self.q)
    
    def is_empty(self):
        return len(self.q) == 0 
    
    def first_time(self):
        if self.is_empty():
            return -1 
        return self.q[-1].time 
    
    def queue(self):
        return self.q
    
    def reset(self):
        self.current_time = 0
        self.q.clear()