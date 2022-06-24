import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super (ActorCritic, self).__init__()

        self.linear1 = nn.Linear(num_inputs, 256)

        self.lstm = nn.LSTMCell(256, 128)

        num_outputs = action_space

        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, num_outputs)
        self.sigma_layer = nn.Linear(128, num_outputs)
    
    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.relu(self.linear1(inputs))
        x = x.view(-1, 256)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), F.softplus(self.sigma_layer(x)), (hx, cx)

