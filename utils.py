import cv2
import numpy as np
from collections import deque
import torch
import math
import psutil

def exponential_epsilon_decay(step_idx, epsilon_start=1, epsilon_end=0.01, epsilon_decay=100_000):
    """
    Calculates the value of epsilon for a given step index using exponential decay and the specified parameters.

    Parameters:
    step_idx (int): The index of the current step.
    epsilon_start (float): The starting value of epsilon.
    epsilon_end (float): The minimum value of epsilon.
    epsilon_decay (float): The rate at which epsilon decays.

    Returns:
    float: The value of epsilon for the given step index.
    """
    return epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * step_idx / epsilon_decay)

def linear_epsilon_decay(step_idx, epsilon_start=1, epsilon_end=0.01, epsilon_decay=100_000):
    """
    Calculates the value of epsilon for a given step index using linear decay and the specified parameters.

    Parameters:
    step_idx (int): The index of the current step.
    epsilon_start (float): The starting value of epsilon.
    epsilon_end (float): The minimum value of epsilon.
    epsilon_decay (float): The total number of steps over which epsilon will decay from epsilon_start to epsilon_end.

    Returns:
    float: The value of epsilon for the given step index.
    """
    return epsilon_end + (epsilon_start - epsilon_end) * max((1 - step_idx / epsilon_decay), 0)

def preprocessing(raw_atari_frame):
    raw_atari_frame = np.dot(raw_atari_frame[...,:3], [0.299, 0.587, 0.114])
    image = cv2.resize(raw_atari_frame, dsize=(110, 84))
    height, width = image.shape
    crop_size = 84
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    cropped_image = image[start_y:start_y + crop_size, start_x:start_x + crop_size]
    return cropped_image/255.0

def frame_stack(observation, stack=None):
    if stack is None:
        stack = deque(maxlen=4)
        for _ in range(4):
            stack.append(preprocessing(observation))
    else:
        stack.append(preprocessing(observation))
    stacked_frames = np.stack(stack, axis=0)
    stacked_frames = torch.from_numpy(stacked_frames.astype(np.float32)).unsqueeze(0)
    return stacked_frames, stack

def get_action(env, q_values, epsilon):
    if np.random.rand() < epsilon:
        action = env.action_space.sample()  # Explore
    else:
        action = torch.argmax(q_values).item() # Exploit
    return action

def get_system_info():
    # Lấy thông tin CPU, RAM và GPU
    cpu_usage = psutil.cpu_percent(interval=1)

    # RAM usage (đã dùng và tổng)
    ram = psutil.virtual_memory()
    ram_used = ram.used / (1024 ** 3)  # GB
    ram_total = ram.total / (1024 ** 3)  # GB
    ram_usage_percent = ram.percent

    # GPU memory usage (đã dùng và tổng)
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated() / 1024 ** 2  # MB
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2  # MB
        gpu_usage_percent = (gpu_memory_used / gpu_memory_total) * 100
    else:
        gpu_memory_used = 0
        gpu_memory_total = 0
        gpu_usage_percent = 0

    return cpu_usage, ram_used, ram_total, ram_usage_percent, gpu_memory_used, gpu_memory_total, gpu_usage_percent