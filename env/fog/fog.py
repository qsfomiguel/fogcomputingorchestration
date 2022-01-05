from os import X_OK
import numpy as np
from collections import deque

# sim_env imports
from env import configs as cfg
from .fog_tools import euclidean_distance, channel_gain, shannon_hartley, db_to_linear
from .task import Task

# utils
from utils.tools import uniform_rand_choice, uniform_rand_int
from utils.custom_exceptions import InvalidValueError

def point_to_point_transmission_rate(d, bw = cfg.NODE_BANDWIDTH):
    if bw <= 0:
        raise InvalidValueError("Available bandwidth has to be positive", "[0,+inf[")
    g = channel_gain(d, cfg.PATH_LOSS_CONSTANT, cfg.PATH_LOSS_EXPONENT)
    p_mw = db_to_linear(cfg.TRANSMISSION_POWER)
    n0_mw = db_to_linear(cfg.THERMAL_NOISE_DENSITY)
    return shannon_hartley(g, p_mw, bw, n0_mw)

def create_random_node(id = 0):
    [x, y] = [uniform_rand_int(low=0, high=cfg.AREA[0]), uniform_rand_int(low=0, high=cfg.AREA[1])]
    n_slices = cfg.DEFAULT_SLICES    

    cpu = uniform_rand_choice(cfg.CPU_CLOCKS)
    ram = uniform_rand_choice(cfg.RAM_SIZES)

    if cfg.DEBUG:
        print("[DE8BUG] Node", id, "created at (x,y)=", (x,y), "cpu=", cpu, "ram=", ram)
    return FogNode(id,x,y,cpu,ram,n_slices)


class FogNode():
    def __init__(self, id, x, y, cpu_freq, ram_size, n_slices):
        super(FogNode, self).__init__()

        if id < 0 or x < 0 or y < 0 or cpu_freq < 0 or ram_size < 0 or n_slices < 0:
            raise InvalidValueError("No arguments on FogNode object can be negative")

        self.id = id
        self.name = "node_"+str(id)
        self.x = x 
        self.y = y 
        self.distances = {}
        self.bw = cfg.NODE_BANDWIDTH 
        self.cpu_freq = cpu_freq 
        self.ram_size = ram_size 
        self._available_cpu_units = cpu_freq / cfg.CPU_UNIT 
        self._available_ram_units = int(ram_size/cfg.RAM_UNIT)

        self.max_slice = n_slices
        self.buffers = [deque(maxlen=cfg.MAX_QUEUE) for _ in range(n_slices)]
        self.dealt_tasks = np.zeros(n_slices, dtype= np.uint64)
        self.total_time_intervals = 0
        self.service_rate = np.zeros(n_slices, dtype = np.float32)
        self.being_processed = np.zeros(n_slices, dtype = np.uint8)

    def __str__(self):
        return self.name 

    def slice_buffer_len(self, i):

        if i< 0 or i >= self.max_slice:
            raise InvalidValueError("Invalid slice number", "[0,"+str(self.max_slice)+"[")
        return len(self.buffers[i])

    def being_processed_on_slice(self, i):

        if i< 0 or i >= self.max_slice:
            raise InvalidValueError("Invalid slice number", "[0,"+str(self.max_slice)+"[")
        return self.being_processed[i]

    def add_task_on_slice(self, i, task):

        if i < 0 or i >= self.max_slice or not isinstance(task, Task):
            raise InvalidValueError("Invalid arguments for add_task_on_slice")

        if len(self.buffers[i]) == self.buffers[i].maxlen:
            return task 

        self.buffers[i].append(task)
        return None 
    
    def remove_task_on_slice(self, i, task):
        if i < 0 or i >= self.max_slice or not isinstance(task, Task):
            raise InvalidValueError("Invalid arguments for remove_task_on_slice")

        try: 
            self.buffers[i].remove(task)
            if task.is_processing():
                #print("removed task  on slice ",i)
                self.being_processed[i] -= 1
            self.dealt_tasks[i] += 1

            self._available_cpu_units += task._cpu_units
            self._available_ram_units += task._memory_units
        except:
            return None 
        return task 
    
    def stop_processing_in_slice(self, i, task, time):
        if i < 0 or i >= self.max_slice or not isinstance(task, Task):
            raise InvalidValueError("Invalid arguments for stop_processing_in_slice")

        if task in self.buffers[i] and task.is_processing():
            #print("task stopped processing in slice ",i)
            self.being_processed[i] -= 1
            self._available_cpu_units += task._cpu_units
            self._available_ram_units += task._memory_units
            task.stop_processing(time)
    
    def start_processing_in_slice(self, k, w, time):
        """Try to start processing w task, if any task has already exceeded time limit, discard it.

		On the slice k, starting at time, try to queue w tasks to processing. It depends
		on the bottleneck (cpu or ram) the amount that actually will start processing. If a task that
		would start processing has already exceeded its constraint, discard it instead.

		Parameters:
			k: int - the slice index
			w: int - number of tasks to attempt to process
			time: float - current simulation time
		"""
        if k < 0 or k >= self.max_slice or w <=0 or time < 0:
            raise InvalidValueError("Invalid arguments for start_processing_in_slice")
        
        under_processing = []; discarded = [];
        #print("size buffer: ", self.slice_buffer_len(k))

		# only process if there is a task on the buffer
        for task in self.buffers[k]:            
            #print(task.is_processing(), w, self._available_cpu_units, self._available_ram_units, np.ceil(task.ram_demand / cfg.RAM_UNIT))
			# only process if has cores, memory and an action request W for it
            if not task.is_processing() and w > 0 and self._available_cpu_units > 0 and self._available_ram_units >= np.ceil(task.ram_demand/cfg.RAM_UNIT):
				# if processor tries to load them and they exceeded constraint, move on
                if task.exceeded_constraint(time):
                    discarded.append(task)
                    continue
				# one cpu unit per task and the ram demand they require
                n_cpu_units = 1
                n_memory_units = np.ceil(task.ram_demand/cfg.RAM_UNIT)
				# and take them from the available pool
                self._available_cpu_units -= n_cpu_units
                self._available_ram_units -= n_memory_units
				# then start the processing
                task.start_processing(n_cpu_units, n_memory_units, time)
                under_processing.append(task)
				# reduce the number that we will still allocate
                w -= 1
                #print("Task being processed on slice",k)
                self.being_processed[k] += 1
        return under_processing, discarded

    def pop_task_to_send(self, i, time):
        if i < 0 or i >= self.max_slice:
            raise InvalidValueError("Invalid slice number", "[0,"+str(self.max_slice)+"[")

        if len(self.buffers[i]) == 0: return None 

        if self.buffers[i][-1].is_processing(): return None 

        if self.buffers[i][-1]._timestamp != time: return None 
        return self.buffers[i].pop()


    def finished_transmitting(self, bw):
        if bw < 0: 
            raise InvalidValueError("Bandwidth cannot be negative")
        self.bw += bw 
    
    def start_transmitting(self, bw):
        if self.bw < bw or bw < 0:
            raise InvalidValueError("Can't transmit in more bandwidth than the available")
        self.bw -= bw 
    
    def available_bandwidth(self):
        return self.bw 

    def set_distances(self, nodes):
        for n in nodes: 
            if n.id == self.id: continue 
            self.distances[n.id] = euclidean_distance(self.x, self.y, n.x, n.y)

    def new_interval_update_service_rate(self):
        self.total_time_intervals += 1
        for i in range(self.max_slice):
            self.service_rate[i] = self.dealt_tasks[i] / self.total_time_intervals 
            if self.service_rate[i] == 0:
                self.service_rate[i] = .1 
    
    def reset(self):
        for i in range(self.max_slice):
            self.buffers[i].clear()
        self._available_cpu_units = int(self.cpu_freq/cfg.CPU_UNIT)
        self._available_ram_units = int(self.ram_size / cfg.RAM_UNIT)
        #print("reset")
        self.being_processed = np.zeros(self.max_slice, dtype= np.uint8)
        self.bw = cfg.NODE_BANDWIDTH 