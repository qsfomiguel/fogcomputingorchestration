import numpy as np 
from utils.tools import uniform_rand_float

eps = np.finfo(np.float32).eps.item()

def channel_gain(distance, linear_coeff, exp_coeff):
    if distance <=0 or linear_coeff <= 0 or exp_coeff <=0:
        raise InvalidValueError("Channel gain function arguments must be positive")
    return linear_coeff*distance**(-exp_coeff)

def shannon_hartley(gain, power, bandwidth, noise_density):
    if gain <= 0 or gain > 1 or bandwidth <=0 or power <0 or noise_density <=0:
        raise InvalidValueError("Shannon_hartley function arguments must be positive and channel gain between [0,1]")
    return float(bandwidth)*np.log2(1+((float(gain)*float(power))/(float(bandwidth)*float(noise_density)+eps))) 

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

def db_to_linear(value):
    return 10**(0.1*float(value))

def linear_to_deb(value):
    if value <=0:
        raise InvalidValueError("linear values must be positive", "[0, +inf[")
    return 10*np.log10(float(value))

def bernoulli_arrival(p):

    if not (0<=p<=1):
        return 0 
    return True if uniform_rand_float() < p else False 

