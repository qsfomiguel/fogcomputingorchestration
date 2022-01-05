from env import configs as cfg 
from utils.custom_exceptions import InvalidValueError, InvalidStateError 

def task_processing_time(task, cpu_units=1):
    if task.is_processing():
        cpu_units = task._cpu_units
    total_cycles = task.cpu_demand*task.packet_size
    total_time = total_cycles / (cpu_units*cfg.CPU_UNIT)

    return total_time 

def task_communication_time(packet_size, bit_rate):
    if bit_rate <= 0 or packet_size <=0:
        raise InvalidValueError("task_communication_time must have a valid task with a positive bitrate")

    return float(float(packet_size)/bit_rate)

class Task(object):
    def __init__(self, timestamp, packet_size = cfg.PACKET_SIZE, delay_constraint=10, cpu_demand=400, ram_demand=400, task_type=None):
        super(Task, self).__init__()

        if delay_constraint < 0 or cpu_demand < 0 or ram_demand < 0 or packet_size < 0 or timestamp < 0:
            raise InvalidValueError("No arguments on Task object can be negative")

        self._timestamp = timestamp
        self.packet_size = packet_size 

        if not task_type == None and len(task_type) == 3: #BUG : task_type can be "hacked"
            self.delay_constainst = task_type[0]
            self.cpu_demand = task_type[1]
            self.ram_demand = task_type[2]
        else:
            self.delay_constainst = delay_constraint
            self.cpu_demand = cpu_demand
            self.ram_demand = ram_demand 
        
        self.processing = False 
        self._memory_units = 0
        self._cpu_units = 0
        self.total_delay = -1 

        self.started_processing = -1 
        self.expected_delay = -1 

    def __str__(self):
        return "[task:"+str(self._timestamp)+"s] is processing" + str(self.processing)
    
    def is_processing(self):
        return self.processing
    
    def is_completed(self):
        return False if self.total_delay == -1 else True 
    
    def start_processing(self, cpu_units, memory_units, start_time):
        if cpu_units < 0 or memory_units < 0 or start_time < self._timestamp:
            raise InvalidValueError("Task do not meet requirements to start processing")
        self.processing = True 
        self._cpu_units = cpu_units
        self._memory_units = memory_units
        self.started_processing = start_time 
        
        if self.expected_delay == -1:
            self.expected_delay = task_processing_time(self)
        
    def stop_processing(self, finish_time):
        if finish_time < self._timestamp:
            raise InvalidValueError("Task cannot stop before creation")
        if self.processing:
            self.processing = False 
            self._cpu_units = 0
            self._memory_units = 0

            if round(finish_time-self.started_processing,5) == round(self.expected_delay,5):
                self.total_delay = finish_time-self._timestamp
                self.expected_delay = 0 

            else:
                self.expected_delay -= finish_time-self.started_processing
    
    def task_remaining_processing_time(self):
        return self.expected_delay

    def task_delay(self):
        return self.total_delay
    
    def task_time(self):
        return self._timestamp
    
    def constraint_time(self):
        return self._timestamp+0.001*self.delay_constainst
    
    def exceeded_constraint(self, current_time):
        return self.constraint_time() < current_time
    