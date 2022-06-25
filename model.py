import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super (ActorCritic, self).__init__()

        self.linear1 = nn.Linear(num_inputs, 256)
        self.layer_norm1 = nn.LayerNorm(256)


        self.linear2 = nn.Linear(256, 128)
        self.layer_norm2 = nn.LayerNorm(128)


        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, action_space)
        self.sigma_layer = nn.Linear(128, action_space)
    
    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.layer_norm1(x)
        x = F.relu(x)
        x = x.view(-1, 256)
        x = self.linear2(x)
        x = self.layer_norm2(x)
        x = F.relu(x)
        return self.critic_linear(x), self.actor_linear(x), F.softplus(self.sigma_layer(x))

