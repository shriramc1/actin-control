import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
import warnings


class QFixed(nn.Module):
    """A "network" for a fixed protocol
    """
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", weights)
    def forward(self, time):
        """
        Arguments:
            time {int} -- time step to get q values for
        """
        q_values =  torch.ones_like(self.weights[time]) * 1000.0
        action = torch.multinomial(input=self.weights[time], 
                                   num_samples=1)
        q_values[action] = 0.0

        return q_values


class QModelFixed:
    """A model for a fixed protocol
    """
    def __init__(self, weights, device=torch.device("cuda:0")):
        self.q_1 = QFixed(weights=weights).to(device)

    def save_networks(self, folder_name):
        """Empty function
        """
        pass


class Q(nn.Module):
    def __init__(self, state_dim=1, action_dim=10, out_scaling=80):
        """
        Arguments:
            state_dim {int} -- dimension of state
            action_dim {int} -- dimension of action
            out_scaling {float} -- amount to scale output of network by
        """
        super().__init__()
        self.out_scaling = out_scaling
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, s):
        """
        Arguments:
            s {torch.tensor} -- state
        Returns:
            torch.tensor -- q values
        """
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        # q_a = self.sigmoid(self.fc3(s) / 10) * 80
        q_a = self.sigmoid(self.fc3(s) / 10) * self.out_scaling
        return q_a


class QModel:
    def __init__(self, state_dim=1, action_dim=10,
                 tau=0.005, out_scaling = 80,
                 device=torch.device("cuda:0")):
        """
        Arguments:
            state_dim {int} -- dimension of state
            action_dim {int} -- dimension of action
            tau {float} -- tau for soft update
            out_scaling {float} -- amount to scale output of network by
            device {torch.device} -- device to use for network
        """
        self.device = device
        self.q_1 = Q(state_dim=state_dim,
                     action_dim=action_dim, out_scaling=out_scaling).to(device)

        self.q_target_1 = Q(state_dim=state_dim,
                            action_dim=action_dim, out_scaling=out_scaling).to(device)

        self.q_2 = Q(state_dim=state_dim,
                     action_dim=action_dim, out_scaling=out_scaling).to(device)

        self.q_target_2 = Q(state_dim=state_dim,
                            action_dim=action_dim, out_scaling=out_scaling).to(device)

        self.q_target_1.eval()
        self.q_target_2.eval()

        self.q_optimizer_1 = Adam(self.q_1.parameters(), lr=3e-4)
        self.q_optimizer_2 = Adam(self.q_2.parameters(), lr=3e-4)

        self._update(self.q_target_1, self.q_1)
        self._update(self.q_target_2, self.q_2)
        self.tau = tau

    def _smaller_weights_last_layer(self, network, scale):
        """Updates the last layer with smaller weights
        Arguments:
            network {torch.nn.Module} -- network to update
            scale {float} -- amount to scale weights by
        """
        last_layers = list(network.state_dict().keys())[-2:]
        for layer in last_layers:
            network.state_dict()[layer] /= scale

    def _update(self, target, local):
        """Set the parametrs of target network to be that of local network
        Arguments:
            target {torch.nn.Module} -- target network
            local {torch.nn.Module} -- local network
        """
        target.load_state_dict(local.state_dict())

    def _soft_update(self, target, local):
        """Soft update of parameters in target Networks
        Arguments:
            target {torch.nn.Module} -- target network
            local {torch.nn.Module} -- local network 
        """
        for target_param, param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau)
                                    + param.data * self.tau)

    def update_target_nn(self):
        """Updates target networks
        """
        self._soft_update(self.q_target_1, self.q_1)
        self._soft_update(self.q_target_2, self.q_2)

    def save_networks(self, folder_name):
        """Saves networks to folder_name
        Arguments:
            folder_name {str} -- folder to save networks to
        """
        torch.save({"model_state_dict": self.q_1.state_dict(),
                    "optimizer_state_dict": self.q_optimizer_1.state_dict()
                    }, folder_name + "q_1")

        torch.save({"model_state_dict": self.q_2.state_dict(),
                    "optimizer_state_dict": self.q_optimizer_2.state_dict()
                    }, folder_name + "q_2")

        torch.save({"model_state_dict": self.q_target_1.state_dict()},
                   folder_name + "q_target_1")

        torch.save({"model_state_dict": self.q_target_2.state_dict()},
                   folder_name + "q_target_2")

    def load_networks(self, folder_name="./"):
        """Loads networks from folder_name
        Arguments:
            folder_name {str} -- folder to load networks from
        """
        q_checkpoint_1 = torch.load(folder_name + "q_1",
                                    map_location=self.device)
        self.q_1.load_state_dict(q_checkpoint_1["model_state_dict"])
        self.q_optimizer_1.load_state_dict(q_checkpoint_1[
            "optimizer_state_dict"])

        q_checkpoint_2 = torch.load(folder_name + "q_2",
                                    map_location=self.device)
        self.q_2.load_state_dict(q_checkpoint_2["model_state_dict"])
        self.q_optimizer_2.load_state_dict(q_checkpoint_2[
            "optimizer_state_dict"])

        q_target_checkpoint_1 = torch.load(folder_name + "q_target_1",
                                           map_location=self.device)
        self.q_target_1.load_state_dict(
            q_target_checkpoint_1["model_state_dict"])

        q_target_checkpoint_2 = torch.load(folder_name + "q_target_2",
                                           map_location=self.device)
        self.q_target_2.load_state_dict(
            q_target_checkpoint_2["model_state_dict"])
