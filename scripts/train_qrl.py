import torch
from torch.utils.tensorboard import SummaryWriter
from actin import CDQL, ReplayBuffer, ActinUDInterface, QModel
import argparse

# device=torch.device("cuda:0")
device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--decision_time", type=int, default=5)
parser.add_argument("--num_decisions_per_episode", type=int, default=50)
parser.add_argument("--target_observable", type=float, default=2.0)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--num_explore_episodes", type=int, default=10)
parser.add_argument("--youngs_threshold", type=int, default=50)
parser.add_argument("--target_observable_name", type=str, default="upper_density", choices=["upper_density", "youngs_modulus"])


config = parser.parse_args()
decision_time = config.decision_time
num_decisions_per_episode = config.num_decisions_per_episode
target_observable = config.target_observable
tau = config.tau
num_explore_epsiodes = config.num_explore_episodes
youngs_threshold = config.youngs_threshold
target_observable_name = config.target_observable_name


tag = f"{target_observable_name}_target_observable_name_{youngs_threshold}_youngs_threshold_{decision_time}_decision_time_{num_decisions_per_episode}_num_decisions_per_episode_{target_observable}_target_observable_{tau}_tau_{num_explore_epsiodes}_num_explore_epsiodes"
compute_youngs_modulus = False
if target_observable_name == "upper_density":
    out_scaling = 40
elif target_observable_name == "youngs_modulus":
    out_scaling = 80
    compute_youngs_modulus = True
else:
    raise ValueError("Invalid target_observable_name")


all_actions = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
env_int = ActinUDInterface(decision_time=decision_time,
                           target_observable=target_observable,
                           all_burn_in_forces=all_actions, 
                           youngs_threshold=youngs_threshold,
                           target_observable_name=target_observable_name, 
                           compute_youngs_modulus=compute_youngs_modulus)

rl_model = QModel(state_dim=env_int.num_states, action_dim=len(all_actions),
                  tau=tau,
                  out_scaling=out_scaling,
                  device=device)
rbuffer = ReplayBuffer(capacity=int(1E6))
gamma = 0.9
num_decisions_per_episode = num_decisions_per_episode

folder_name = tag + str("/")
writer = SummaryWriter(log_dir=folder_name)


cdql = CDQL(env_int=env_int, rl_model=rl_model, rbuffer=rbuffer,
            writer=writer, gamma=gamma, num_decisions_per_episode=num_decisions_per_episode,
            all_actions=all_actions, num_explore_episodes=num_explore_epsiodes, device=device,
            batch_size=32,
            folder_name=folder_name)
cdql.train()
