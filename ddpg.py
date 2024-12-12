import torch
import torch.nn.functional as f
import torch.nn as nn
from colorama import Fore

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_features=state_dim, out_features=400)
        self.layer2 = nn.Linear(in_features=400, out_features=300)
        self.layer3 = nn.Linear(in_features=300, out_features= action_dim)

    def forward(self, x):
        x = f.relu(self.layer1(x))
        x = f.relu(self.layer2(x))
        x = f.tanh(self.layer3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_features=state_dim, out_features=400)
        self.layer2 = nn.Linear(in_features=400 + action_dim, out_features=300)
        self.layer3 = nn.Linear(in_features=300, out_features=1)

    def forward(self, state, action):
        state_out = f.relu(self.layer1(state))
        combined = torch.cat([state_out, action], dim=1)
        x = f.relu(self.layer2(combined))
        q_value = self.layer3(x)

        return q_value

def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

if __name__ == '__main__':
    state_dim = 24
    action_dim = 4
    actor = Actor(state_dim=state_dim, action_dim=action_dim)
    actor.apply(init_weights)
    print(Fore.CYAN + f"{actor}")
    state_dim_random = torch.randn(1, state_dim)
    output_actor = actor(state_dim_random)
    print(Fore.GREEN + f"{output_actor.shape}")
    print(output_actor)
    critic = Critic(state_dim=state_dim, action_dim=action_dim)
    critic.apply(init_weights)
    print(Fore.CYAN + f"{critic}")
    state = torch.randn(1, state_dim)
    action = torch.randn(1,action_dim)
    output_critic = critic(state, action)
    print(Fore.GREEN + f"{output_critic.shape}")
    print(output_critic)



