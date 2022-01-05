from numpy import random
import gym
from gym import spaces
from gym.utils import seeding 

import numpy as np 

from env import configs as cfg
from env.fog.fog import create_random_node
from env.events.events import Event_Queue
from env.events.set_arrivals import Set_Arrivals
from env.events.stop_transmitting import Stop_Transmitting
from env.events.offload_task import Offload_Task
from env.events.start_transmitting import Start_Transmitting
from env.events.start_processing import Start_Processing
from utils.tools import set_tools_seed
from env.rewards import jbaek_reward_fun2


import numpy

class FogEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, case = cfg.BASE_SLICE_CHARS, random_seed = cfg.RANDOM_SEED, max_time = cfg.SIM_TIME, time_step = cfg.TIME_STEP, n_nodes = cfg.NODES, node_obj = None, reward= jbaek_reward_fun2):
        super(FogEnv, self).__init__()

        self.random_seed = random_seed
        set_tools_seed(random_seed)

        self.n_nodes = n_nodes 

        if node_obj is None:
            self.nodes = [create_random_node(i) for i in range(1, n_nodes+1)]
        else:
            self.nodes = node_obj 
        self.case = case 
        for n in self.nodes:
            n.set_distances(self.nodes)
        self.event_queue = Event_Queue()

        self.clock = 0 
        self.max_time = max_time 
        self.time_step = time_step 
        
        self.reward_func = jbaek_reward_fun2 
        self.saved_step_info = None 

        action_possibilities = [np.append([self.n_nodes+1 for _ in range(n.max_slice)], 
            [min(n._available_cpu_units, n._available_ram_units/(np.ceil(case["task_type"][k][2]/cfg.RAM_UNIT)))+1 for k in range(n.max_slice)]) for n in self.nodes]
        action_possibilities = np.array(action_possibilities, dtype= np.float32)

        self.action_space = spaces.MultiDiscrete(action_possibilities)

        state_possibilities = [np.concatenate(([2 for _ in range(n.max_slice)],[cfg.MAX_QUEUE+1 for _ in range (n.max_slice)],
            [min(n._available_cpu_units, n._available_ram_units/(np.ceil(case["task_type"][k][2]/cfg.RAM_UNIT)))+1 for k in range(n.max_slice)],
            [n._available_cpu_units+1], [n._available_ram_units+1])) for n in self.nodes] 
        state_possibilities = np.array(state_possibilities, dtype= np.float32)
        self.observation_space = spaces.MultiDiscrete(state_possibilities)

        self.event_queue.add_event(Set_Arrivals(0, time_step, self.nodes, self.case))
        
        self.seed(random_seed)

    def step(self, action_n):
        action_n = self.cap_action_n(action_n)
        assert self.action_space.contains(action_n)
        state_t = self.get_state_obs()

        obs_n = []
        rw = 0 
        info = {
                    "delay_list": [],
                    "overflow": 0,
                    "discarded": 0,
                };
        
        for n in self.nodes: 
            n.new_interval_update_service_rate()
            #print(n.id, n.service_rate)
        
        for i in range(self.n_nodes):
            self.set_agent_action(self.nodes[i], action_n[i])

        self.clock += self.time_step
        done = self.clock >= self.max_time

        while not self.event_queue.is_empty() and self.event_queue.first_time() <= self.clock:
            ev = self.event_queue.pop_event()
            t = ev.execute(self.event_queue)

            if t is not None:
                if t.is_completed():
                    info["delay_list"].append(t.task_delay())
                elif ev.classtype=="Discard_Task":
                    info["discarded"] += 1
                else:
                    info["overflow"] += 1

        obs_n = self.get_state_obs()

        rw = self.reward_func(self, state_t, action_n, obs_n, info)

        self.saved_step_info = [state_t, action_n, info]

        return obs_n, rw, done, info 

    def reset(self):
        self.event_queue.reset()
        self.event_queue.add_event(Set_Arrivals(0, self.time_step, self.nodes, self.case))
        for node in self.nodes:
            node.reset()
        self.clock = 0

        return self.get_state_obs()

    def get_state_obs(self):
        return np.array([self.get_agent_observation(n) for n in self.nodes], dtype=np.float32)

    def get_agent_observation(self, n):
        pobs = np.concatenate(([1 if len(n.buffers[k]) > 0 and self.clock == n.buffers[k][-1]._timestamp else 0 for  k in range(n.max_slice)],
                [len(n.buffers[k]) for k in range(n.max_slice)], [n.being_processed_on_slice(k) for k in range(n.max_slice)], 
                [n._available_cpu_units], [n._available_ram_units]))
        return np.array(pobs, dtype=np.float32)

    def cap_action_n(self, action_n):
        for n in range(len(action_n)):
            for i in range(len(action_n[n])):
                if action_n[n][i] >= self.action_space.nvec[n][i]:
                    action_n[n][i] = self.action_space.nvec[n][i]-1
        return action_n

    def set_agent_action(self, n, action):
        [fks, wks] = np.split(action, 2)
        concurr = sum([1 if fk!= n.id and fk!= 0 else 0 for fk in fks])
        for k in range(cfg.DEFAULT_SLICES):
            if wks[k]!=0:
                self.event_queue.add_event(Start_Processing(self.clock, n, k, wks[k]))
            if fks[k] != n.id and fks[k] != 0:
                self.event_queue.add_event(Offload_Task(self.clock, n, k, self.nodes[fks[k]-1], concurr))

    def render(self, mode="human", close=False):

        if self.saved_step_info is None: return 
        nodes_obs = self.saved_step_info[0]
        nodes_actions = self.saved_step_info[1]
        curr_obs = self.get_state_obs()
        print("-----", round(self.clock*1000,2),"ms ------")

        for i in range(self.n_nodes):
            print("----", self.nodes[i], "---")
            [a,b,be,rc,rm] = np.split(nodes_obs[i], [cfg.DEFAULT_SLICES, cfg.DEFAULT_SLICES*2, cfg.DEFAULT_SLICES*3, cfg.DEFAULT_SLICES*3+1])
            print(" obs[ a:", a,"b:", b, "be:", be, "rc:",rc, "rm:",rm,"]")
            [f, w] = np.split(nodes_actions[i], 2)
            print("act[ f:",f,"w:",w,"]")
            [a, b, be, rc, rm] = np.split(curr_obs[i], [cfg.DEFAULT_SLICES, cfg.DEFAULT_SLICES*2, cfg.DEFAULT_SLICES*3, cfg.DEFAULT_SLICES*3+1])
            print(" obs[ a:", a,"b:", b, "be:", be, "rc:",rc, "rm:",rm,"]")

        for ev in self.event_queue.queue():
            if ev.classtype != "Task_Arrival" and ev.classtype != "Task_Finished":
                print(ev.classtype+"["+str(round(ev.time*1000,2))+"ms]", end='-->')

        print(round(1000*(self.clock+self.time_step)),"ms")
        input("\n Enter to continue...")
        pass
    
    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def is_done(self):
        return self.clock >= self.max_time  

