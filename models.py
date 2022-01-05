import tensorflow as tf
import numpy as np
import gym
import math
import copy
import time 

from tensorflow.python.keras.backend_config import epsilon
from tensorflow.python.ops.gen_batch_ops import batch
from tensorflow import keras
from tensorflow.keras import optimizers  
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber, Reduction
from gym import wrappers
from gym.spaces import Discrete
from copy import deepcopy
from configs import *
from collections import namedtuple
from gym.utils import colorize
from tools import TemporaryExperience, get_next_state, map_to_int, map_int_to_int_vect, replay_buffer, write_dictionary_on_csvs, plt_line_plot, set_tf_seed, info_gather, data_display_init, info_logs
from deep_tools.dqn_tools import instant_reward 
from frames import frame_1
from env import configs 
from env.env import FogEnv

class DQN(object):
    def __init__(self, env, dqn_frame = frame_1):
        super(DQN,self).__init__()

        self.used_frame = dqn_frame
        self.dqn_actors = [dqn_frame(map_to_int(action_space_n)+1, 11) for action_space_n in env.action_space.nvec]
        self.n_actors = len(env.nodes)

        self.name = env.case["case"]+"_rd"+str(env.random_seed)+"_dqn_orchestrator_"+dqn_frame.short_str()
        self.env = copy.deepcopy(env)

        self.n_action_spaces = env.action_space.nvec
        self.observation_spaces = env.observation_space.nvec   
        
        self.policy_random = np.random.RandomState(seed=ALGORITHM_SEED)
        self.epsilon= EPSILON_INITIAL

        self.actual_state = None
    
    def __str__(self):
        return self.name

    @staticmethod
    def short_str():
        return "dqn"
    
    def act(self, next_obs):
        if self.actual_state is None:
            x = tf.expand_dims(next_obs, 0)
            self.actual_state = tf.repeat(x, repeats= TIME_SEQUENCE_SIZE, axis = 0)
        
        self.actual_state = get_next_state(self.actual_state, next_obs)
        action = []
        for k in range(self.n_actors):
            obs = self.actual_state[:,k]
            dqn_actor = self.dqn_actors[k]
            action_space = self.n_action_spaces[k]

            obs = tf.expand_dims(obs, 0)
            q_values = dqn_actor(obs) 
            action_i = map_int_to_int_vect(action_space, self.get_action(tf.squeeze(q_values)))
            action.append(action_i)

        return np.array(action)
    
    def save_models(self, saved_models_path = DEFAULT_SAVE_MODELS_PATH):
        complete_path = saved_models_path + self.name
        for i, model in enumerate(self.dqn_actors):
            model.save(complete_path+"_node"+str(i))

    def get_action(self, q_values):
        rand = self.policy_random.rand()
        if rand < self.epsilon:
            action_int = self.policy_random.randint(len(q_values))
        else:
            action_int = tf.math.argmax(q_values)
        return action_int 

    def update_network(self, dqn_actors_target):
        for dqn_target, dqn in zip(dqn_actors_target, self.dqn_actors):
            dqn_target.set_weights(dqn.get_weights())
        return dqn_actors_target

    def train(self, batch_size: int = BATCH_SIZE):
        dqn_actors_target = [self.used_frame(map_to_int(action_space_n)+1, 11) for action_space_n in self.env.action_space.nvec]

        experience_replay_buffer = replay_buffer(REPLAY_BUFFER_START_SIZE)
        temporary_replay_buffer = []

        # for training steps    
        optimizers = {}
        for i in range(self.n_actors):
            lr = PolynomialDecay(DQN_LEARNING_RATE, MAX_DQN_ITERATIONS-REPLAY_BUFFER_START_SIZE, MIN_DQN_LEARNING_RATE)
            optimizers[i] = Adam(learning_rate=lr)
        huber_loss = Huber(reduction=Reduction.SUM)
        
        epsilon_start = EPSILON_INITIAL
        epsilon_it = 0
        epsilon_decay = math.log(epsilon_start)/EPSILON_RENEWAL_RATE - math.log(MIN_EPSILON)/EPSILON_RENEWAL_RATE

        renewal_it = EPSILON_RENEWAL_RATE
        r_buffer = tf.TensorArray(dtype=tf.float32, size=MAX_DQN_ITERATIONS)
        avg_instant_rw = 0; r_it = -renewal_it

        for ep in tf.range(MAX_DQN_ITERATIONS):
            if ep % renewal_it==0: 
                next_obs = self.env.reset()
                x = tf.expand_dims(next_obs, 0)
                state = tf.repeat(x, repeats= TIME_SEQUENCE_SIZE, axis=0)
                r_it += renewal_it
                print("[LOG] env reset")

            action= [] ; action_int = [];
                            
            for i in range(self.n_actors):
                obs = state[:,i]
                actor_dqn = self.dqn_actors[i]
                action_space = self.n_action_spaces[i]
                # just one batch
                obs = tf.expand_dims(obs, 0)
                # call its model
                q_values = actor_dqn(obs)
                # remove the batch and calculate the action
                action_i = self.get_action(tf.squeeze(q_values))
                action.append(map_int_to_int_vect(action_space, action_i))
                action_int.append(action_i)
            #print("state in ep:",next_obs)
            reward, estimated_transmission_delays = instant_reward(self.env, next_obs, action)
            #print("reward:",reward,"etd:", estimated_transmission_delays)
            start_time = self.env.clock 
            next_obs, _, _, _ = self.env.step(np.array(action))

            state_next = get_next_state(state, next_obs)

            exp = TemporaryExperience(state, action_int, reward, state_next, start_time, estimated_transmission_delays) 
            temporary_replay_buffer.append(exp) 

            for e in temporary_replay_buffer:
                done = e.check_update(self.env.clock, next_obs)
                if done: 
                    temporary_replay_buffer.remove(e)
                    state_e, action_e, reward_e, state_next_e = e.get_tuple()
                    r_buffer= r_buffer.write(r_it+int(e.start_time*1000), reward_e)
                    experience_replay_buffer.push(state_e, action_e, reward_e, state_next_e)

            state = tf.identity(state_next) 
            if experience_replay_buffer.size() < REPLAY_BUFFER_START_SIZE and ep% 100== 0 : 
                print("Iteration", ep.numpy(), "filling replay_buffer", flush = True)

            if experience_replay_buffer.size() == REPLAY_BUFFER_START_SIZE:
                epsilon_it += 1
                self.epsilon = max(math.exp(-epsilon_decay*(math.fmod(epsilon_it, EPSILON_RENEWAL_RATE))+ math.log(epsilon_start)), MIN_EPSILON) 

                dataset_for_training = tf.data.Dataset.from_tensor_slices(experience_replay_buffer.get_replay())
                dataset_for_training = dataset_for_training.shuffle(buffer_size=REPLAY_BUFFER_START_SIZE+REPLAY_BUFFER_START_SIZE).batch(batch_size)

                for data in dataset_for_training: 
                    (state_train, action_train, reward_train, next_state_train) = data
                    break 

                losses_buf = {}

                for k in tf.range(self.n_actors):
                    state_train_actor = state_train[:,:,k]
                    next_state_train_actor = next_state_train[:,:,k]

                    target = tf.TensorArray(dtype= tf.float32, size=batch_size)
                    for n in tf.range(batch_size):
                        x = tf.expand_dims(next_state_train_actor[n],0)
                        q_next_target_values = dqn_actors_target[n](x)
                        target_reward_n = tf.cast(reward_train[n], tf.float32)
                        target_t = target_reward_n + GAMMA*tf.reduce_max(tf.squeeze(q_next_target_values))                                
                        target = target.write(n, target_t)
                    target = target.stack()
                    #print(target.shape())

                    with tf.GradientTape() as tape: 
                        q_values = self.dqn_actors[k](state_train_actor)

                        actual_q_values = tf.TensorArray(dtype=tf.float32, size=batch_size)
                        for n in tf.range(batch_size):
                            actual_q_values= actual_q_values.write(n, q_values[n][int(action_train[n,k])])
                        actual_q_values = actual_q_values.stack()
                            
                        loss = huber_loss(actual_q_values, target)

                    gradients = tape.gradient(loss, self.dqn_actors[k].trainable_weights)
                    optimizers[k.numpy()].apply_gradients(zip(gradients, self.dqn_actors[k].trainable_weights))

                    del tape 

                    losses_buf[k.numpy()] = loss.numpy()
                    
                if ep%TARGET_NETWORK_UPDATE_RATE == 0:
                    dqn_actors_target = self.update_network(dqn_actors_target)

                if ep%EPSILON_RENEWAL_RATE == 0:
                    epsilon_start = epsilon_start * EPSILON_RENEWAL_FACTOR
                    epsilon_decay = math.log(epsilon_start)/EPSILON_RENEWAL_RATE - math.log(MIN_EPSILON) / EPSILON_RENEWAL_RATE
                    
                avg_instant_rw += reward

                if ep%10 == 0:
                    print("Iteration",ep.numpy()," [avg instant rw:", avg_instant_rw/10, "][epsilon:", round(self.epsilon,3),"] dqn it losses:", losses_buf, flush=True)
                    avg_instant_rw = 0

        r_buffer = r_buffer.stack().numpy()
        average_total_reward = r_buffer[0] 
        iter_rewards = []
        for i in range(0, len(r_buffer)):
            average_total_reward = (1-RW_EPS)*average_total_reward + RW_EPS*r_buffer[i]
            iter_rewards.append(average_total_reward)
            if i%1000 == 0:
                print(int(100*i/len(r_buffer)), "%", "complete...", flush = True)
        print("Done!")

        self.epsilon = MIN_EPSILON
        return iter_rewards

class ConfigManager:
    def __init__(self, env_name, policy_name, policy_params=None, train_params=None,
                 wrappers=None):
        self.policy_name = policy_name
        self.policy_params = policy_params or {}
        self.train_params = train_params or {}
        self.wrappers = wrappers or []

        self.env = env_name()
        device_name = tf.test.gpu_device_name()
        if not device_name:
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))


    def run_rl_alg_on_env(self, alg, case, compiled_info = None, debug= False, train = False, save = False, load = False):
        set_tf_seed(ALGORITHM_SEED)
        orchestrator = alg(self.env)

        print("\n==================================================")
        print("Loaded gym.env:", self.env)
        print("Wrappers:", self.wrappers)
        #print("Loaded policy:", policy.__class__)
        print("Policy params:", self.policy_params)
        print("Train params:", self.train_params)
        print("==================================================\n")

        training_time = time.time()
        iteration_rewards = orchestrator.train()
        print("Finished training in",round(time.time()-training_time,2),"s")
        d = {"rw_"+str(orchestrator): iteration_rewards}
        write_dictionary_on_csvs(d)
        plt_line_plot(d, title="avg_rw_"+str(orchestrator))

        start_time = time.time()
        next_obs = self.env.reset()
        done = False;

        while not done:
            action_n = np.array(orchestrator.act(next_obs), dtype=np.uint8)
            next_obs, rw_n, done, info = self.env.step(action_n)
            if debug: self.env.render()
            if compiled_info is not None: compiled_info = info_gather(compiled_info, info)

        if compiled_info is not None: info_logs(str(orchestrator), round(time.time()-start_time,2), compiled_info)
        key = str(orchestrator)

        if save:
            orchestrator.save_models()
        tf.keras.backend.clear_session()

        return compiled_info, key


config = ConfigManager(FogEnv, DQN)
config.run_rl_alg_on_env(DQN, configs.OFF_CASE_2, data_display_init(), debug = False, train = True, save = False, load = False)
