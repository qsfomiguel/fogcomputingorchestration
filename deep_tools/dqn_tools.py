import numpy as np 

from tools import point_to_point_transmission_rate
from env import configs as cfg 


def instant_reward(env, state, action):

    case = env.case
    reward = 0.0 
    e_transmission_delays = []
    state = np.array(state)
    action= np.array(action)
    for s,a,n in zip(state, action, env.nodes):
        node_reward = 0.0
        [fks, wks] = np.split(a,2)
        concurr = sum([1 if fk!=n.id and fk!=0 else 0 for fk in fks])
        node_e_transmission_delays = []

        for k in range(cfg.DEFAULT_SLICES):
            if s[k] == 1 and fks[k] != 0:

                Dt_ik = 0.0
                if fks[k] != n.id:
                    bw = int(n.available_bandwidth()/concurr)
                    if bw >= cfg.NODE_BANDWIDTH_UNIT:
                        Dt_ik = cfg.PACKET_SIZE / point_to_point_transmission_rate(n.distances[fks[k]],bw)
                node_e_transmission_delays.append(Dt_ik)

                D_ik = Dt_ik*1000
                D_ik += s[cfg.DEFAULT_SLICES+k]/n.service_rate[k]
                D_ik += 1000* cfg.PACKET_SIZE* case["task_type"][k][1] / (cfg.CPU_UNIT)

                if D_ik >= case["task_type"][k][0]:
                    coeficient = -1 
                else:
                    coeficient = 1

                node_reward += s[k] * coeficient

        reward += node_reward /n.max_slice
        e_transmission_delays.append(node_e_transmission_delays)
    
    return reward, e_transmission_delays
