from collections import deque, namedtuple
from utils.custom_exceptions import InvalidValueError
from env.fog.fog_tools import shannon_hartley
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import csv

my_path = os.getcwd()+"/results"

OVERLOAD_WEIGHT = 0.2 # 2 used when no delay constraint

DEFAULT_SLICES = 1
NODE_BANDWIDTH = 1e6
NODE_BANDWIDTH_UNIT = 1e5
TRANSMISSION_POWER = 20 
PACKET_SIZE = 5000
MAX_QUEUE = 10

PATH_LOSS_EXPONENT = 4
PATH_LOSS_CONSTANT = 0.001
THERMAL_NOISE_DENSITY = -174

eps = np.finfo(np.float32).eps.item()

class replay_buffer(object):
    def __init__(self,max_size=-1):
        super(replay_buffer,self).__init__()
        self.max_size = max_size
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.next_state_buffer = []

    def __str__(self):
        return "maximum:"+str(self.max_size)+"current:"+str(len(self.state_buffer))

    def push(self, state, action, rw, next_state):
        if len(self.state_buffer) == self.max_size:
            self.state_buffer.pop(0) 
        if len(self.action_buffer) == self.max_size:
            self.action_buffer.pop(0) 
        if len(self.reward_buffer) == self.max_size:
            self.reward_buffer.pop(0) 
        if len(self.next_state_buffer) == self.max_size:
            self.next_state_buffer.pop(0) 

    def get_replay(self):
        return(self.state_buffer, self.action_buffer, self.reward_buffer, self.next_state_buffer)

    def size(self):
        return len(self.state_buffer)

def map_to_int(actions, vector = None):
    if vector is None:
        vector = [act-1 for act in actions]

    ret = 0 

    for v,a in zip(vector,actions):
        assert v < a 
        ret = ret*a + v 

    return ret

def map_int_to_int_vect(maxes, num):
    vect= []; retval = num; 
    for m in maxes[::-1]:
        vect.insert(0,retval%m)
        retval= int(retval/m)
    return vect

def db_to_linear(value):
    return 10**(0.1*float(value))
    
def channel_gain(distance, linear_coeff, exp_coeff):
    if distance <= 0 or linear_coeff <=0 or exp_coeff <=0:
        ## Raise error
        print("bruh")
        exit()
    return linear_coeff*distance**(-exp_coeff)

def point_to_point_transmission_rate (d, bw = NODE_BANDWIDTH):
    if bw <= 0:
        ## raise exception
        print("bruh")
        exit()
    gain = channel_gain(d, PATH_LOSS_CONSTANT, PATH_LOSS_EXPONENT)
    power_mw = db_to_linear(TRANSMISSION_POWER)
    n0_mw = db_to_linear(THERMAL_NOISE_DENSITY)
    return shannon_hartley(gain, power_mw, bw, n0_mw)

def get_next_state(state, new_value):
	state_list = tf.unstack(state)
	state = tf.stack((state[1:]))
	state = tf.concat((state, [new_value]), axis=0)
	return state


class TemporaryExperience(object):
    def __init__(self, state, action, reward, next_state, start_time, transmission_delays):
        super(TemporaryExperience, self).__init__()
        self.state = state
        self.action = action 
        self.reward = reward 
        self.next_state = next_state 

        self.start_time = start_time 
        self.times = []

        for node, node_times in enumerate(transmission_delays):
            for slice, slice_node_times in enumerate(node_times):
                if slice_node_times != 0.0:
                    self.times.append([node, slice, slice_node_times+start_time])

    def get_tuple(self):
        return (self.state, self.action, self.reward, self.next_state)

    def check_update(self, time, obs_n):
        done = False 
        for i in self.times: 
            j,k,t = i
            if t <= time: 
                self.times.remove(i)
                if obs_n[j][DEFAULT_SLICES+k]+1 >= MAX_QUEUE:
                    self.reward -= OVERLOAD_WEIGHT/DEFAULT_SLICES 

        if len(self.times) == 0:
            done = True 
        
        return done 


def apply_wrappers(env, list_of_wrappers):
    for name, params in list_of_wrappers:
        wrapper_class = load_wrapper_class(name)
        env = wrapper_class(env, **params)
    return env

def set_tools_seed(seed=None):
    global np_tools_random 
    np_tools_random = np.random.RandomState(seed=seed)

def set_tf_seed(seed=1):
    tf.random.set_seed(seed)

results_path = os.getcwd()+"/results/"
windows_path = os.getcwd()+"\\results\\"

def no_format(data):
    return data 

def write_dictionary_on_csvs(d, format_data = no_format):
    for key, value in d.items():
        write_to_csv(key+".csv",value, format_data)

def append_to_file(filename, data, format_data = no_format):
    with open(results_path+filename, "a", newline='') as f:
        d = format_data(data)
        f.write(str(d))

    return 

def write_to_csv(filename, data, format_data= no_format):
    with open(results_path+filename, "w", newline='') as f:
        d = format_data(data)
        wr = csv.writer(f)
        wr.writerow(d)
    return

def plt_line_plot(df, normalize=False, r=None, title="default_line_plt"):
    fig, ax = plt.subplots()
    for key, values in df.items():
        if normalize:
            if r is None:
                r = max(values) - min(values)
            values = [(v+eps)/(r+eps) for v in values]
        plt.plot(values, label=key)
    plt.legend(loc="upper left")
    fig.savefig(my_path+title+".png")

def info_gather(data, info):
    data["delay_sum"] += sum(info["delay_list"])
    data["success"] += len(info["delay_list"])
    data["overflow"] += info["overflow"]
    data["discard"] += info["discarded"]
    data["total"] = data["discard"] + data["success"] + data["overflow"]

    if data["total"] > 0:
        data["average_delay"] = data["delay_sum"]/data["total"]
        data["success_rate"] = data["success"]/data["total"]
        data["overflow_rate"] = data["overflow"]/data["total"]

    return data 

def data_display_init():
    keys = ["delay_sum", "success", "overflow", "discard"]
    return dict.fromkeys(keys, 0)

def info_logs(key, extime, data):
    print("[INFO LOG] Finished", key,"in",extime,"s")
    print("[1/4] total tasks", data["total"])
    print("[2/4] average delay", round(1000*data["average_delay"],5),"ms")
    print("[3/4] success rate", round(data["success_rate"],5))
    print("[4/4] overflow rate", round(data["overflow_rate"],5))


