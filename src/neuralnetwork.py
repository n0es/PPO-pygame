import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPO(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(PPO, self).__init__()
        self.device = device
        self.pol_eval = self._build_eval_nn(input_size, hidden_size, output_size)
        self.old_pol_eval = self._build_eval_nn(input_size, hidden_size, output_size)
        self.value_function = self._build_value_function(input_size, hidden_size)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.MseLoss = nn.MSELoss()

    def _clear_grad(self):
        self.pol_eval.zero_grad()
        self.value_function.zero_grad()

    def _build_eval_nn(self, input_size, hidden_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)  # softmax to obtain the probabilities for all actions
        ).to(self.device)

    def _build_value_function(self, input_size, hidden_size):
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(self.device)

    def calculate_outputs(self, inputs):
        inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        output = self.pol_eval(inputs)
        return output.cpu().detach().numpy()

    def update(self, states, actions, log_probs_old, returns, advantages, clip_epsilon):
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        # Update policy
        log_probs = self.pol_eval(states).gather(1, actions.unsqueeze(1)).log()
        ratio = (log_probs - log_probs_old).exp()
        L_clip_obj = advantages * ratio

        L_clip_obj_clamped = advantages * torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon)
        L_clip = -torch.min(L_clip_obj, L_clip_obj_clamped).mean()

        # Update value function
        L_value = self.MseLoss(self.value_function(states), returns.unsqueeze(1))

        loss = L_clip + L_value

        self._clear_grad()
        loss.backward()
        self.optimizer.step()

        self.old_pol_eval.load_state_dict(self.pol_eval.state_dict())
