# experimental case
from tools import DEFAULT_SLICES, OVERLOAD_WEIGHT

DEBUG = False

BASE_SLICE_CHARS = {
	"case": "base",
	"arrivals" :  [0.6, 0.6], #[0.6, 0.6, 0.6], # [0.6],
	"task_type" : [[10, 600, 800], [20, 1200, 400]]# [[5, 600, 400], [10, 600, 400], [10, 400, 800]] # [[15, 1200, 800]] 
}

RANDOM_SEED = 2**19-1 # mersenne prime seeds at 2, 3, 5, 7, 13, 17, 19, 31
RANDOM_SEED_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


TIME_STEP = 0.001 # seconds
TOTAL_TIME_STEPS = 1024
SIM_TIME = TOTAL_TIME_STEPS*TIME_STEP

DEFAULT_SLICES = 1
NODES = 5 #default 5

AREA = [100, 100] 
PATH_LOSS_EXPONENT = 4
PATH_LOSS_CONSTANT = 0.001
THERMAL_NOISE_DENSITY = -174 # dBm/Hz

MAX_QUEUE = 10
CPU_CLOCKS = [5e9, 6e9, 7e9, 8e9, 9e9, 10e9] # [5, 6, 7, 8, 9, 10] GHz
RAM_SIZES = [2400, 4000, 8000] # MB = [6, 10, 20] units
CPU_UNIT = 1e9 # 1 GHz
RAM_UNIT = 400 # MB

NODE_BANDWIDTH = 1e6 # Hz
NODE_BANDWIDTH_UNIT = 1e5 # Hz , i.e., 10 concurrent transmissions is the maximum
TRANSMISSION_POWER = 20 # dBm

PACKET_SIZE = 5000

OVERLOAD_WEIGHT = 0.2

OFF_CASE_2 = {
    "case": "ofc2",
    "arrivals" : [0.6],
    "task_type" : [[15, 1200, 800]]
}