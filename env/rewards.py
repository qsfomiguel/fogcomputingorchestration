import numpy as np 
import math 

from env.fog.fog import point_to_point_transmission_rate
from env.events.task_arrival import is_arrival_on_slice
from env import configs as cfg 

def jbaek_reward_fun2(env, state, action, next_state, info):
    case = env.case 
    reward = 0.0 

    for s, a, n in zip(state, action, env.nodes):
        node_reward = 0.0
        [fks, wks] = np.split(a,2)
        concurr = sum([1 if fk!= n.id and fk!= 0 else 0 for fk in fks])

        for k in range(cfg.DEFAULT_SLICES):
            if s[k] == 1 and fks[k] != 0:
                Dt_ik = 0.0
                if fks[k] != n.id:
                    bw = int(n.available_bandwidth()/concurr)
                    if bw >= cfg.NODE_BANDWIDTH_UNIT:
                        Dt_ik = cfg.PACKET_SIZE / (point_to_point_transmission_rate(n.distances[fks[k]],bw))

                dest_node = env.nodes[fks[k]-1]
                estimated_buffer_length = min(np.ceil(np.ceil(Dt_ik*1000)*(max(case["arrivals"][k]-dest_node.service_rate[k],0)) + state[fks[k]-1][cfg.DEFAULT_SLICES+k]), 10)

                D_ik = Dt_ik*1000
                D_ik += estimated_buffer_length/dest_node.service_rate[k]

                Dp_ik = 1000* cfg.PACKET_SIZE* case["task_type"][k][1] / (cfg.CPU_UNIT)
                D_ik += Dp_ik 

                if D_ik >= case["task_type"][k][0]:
                    coeficient = -1 
                else:
                    coeficient = 1 - ((D_ik - Dp_ik) / case["task_type"][k][0])

                if estimated_buffer_length + 1 >= cfg.MAX_QUEUE:
                    coeficient -= cfg.OVERLOAD_WEIGHT
                
                node_reward += s[k] * coeficient
        reward += node_reward/n.max_slice
    
    return reward

