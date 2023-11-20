"""
Grow networks for fixed time
"""
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from actin import CDQL, ReplayBuffer, ActinUDInterface, QModel
import argparse

# device=torch.device("cuda:0")
device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--decision_time", type=int, default=1)
parser.add_argument("--num_decisions_per_episode", type=int, default=50)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--num_explore_episodes", type=int, default=20)
parser.add_argument('--target_observables', action='store',
                    type=float, nargs="*", help='Residues to be included in the PDB file')
parser.add_argument("--youngs_threshold", type=int, default=50)
parser.add_argument("--target_observable_name", type=str,
                    default="upper_density", choices=["upper_density", "youngs_modulus"])

parser.add_argument("--init_run_num", type=int, default=0)

config = parser.parse_args()
decision_time = config.decision_time
num_decisions_per_episode = config.num_decisions_per_episode
target_observables = config.target_observables
tau = config.tau
num_explore_epsiodes = config.num_explore_episodes
youngs_threshold = config.youngs_threshold
target_observable_name = config.target_observable_name
init_run_num = config.init_run_num

assert len(target_observables) == 2

layer_0 = {1.0: 53, 2.0: 97, 3.0: 129, 4.0: 238, 5.0: 358}
layer_1 = {1.0: 75, 2.0: 119, 3.0: 140, 4.0: 232, 5.0: 350}
num_decisions_per_episode_by_layer = {}
num_decisions_per_episode_by_layer[0] = layer_0[target_observables[0]]
num_decisions_per_episode_by_layer[1] = layer_1[target_observables[1]]


root_train_folder_name = "/scratch/groups/rotskoff/actin/092823RLTraining/"


all_actions = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]


compute_youngs_modulus = False
if target_observable_name == "upper_density":
    out_scaling = 40
elif target_observable_name == "youngs_modulus":
    out_scaling = 80
    compute_youngs_modulus = True
else:
    raise ValueError("Invalid target_observable_name")

for run_num in range(init_run_num*20, (init_run_num+1)*20):
    env_int = ActinUDInterface(decision_time=decision_time,
                               target_observable=target_observables[0],
                               all_burn_in_forces=all_actions,
                               target_observable_name=target_observable_name,
                               compute_youngs_modulus=compute_youngs_modulus)

    all_rl_models = {}
    for target_observable in np.unique(target_observables):
        tag = f"{target_observable_name}_target_observable_name_{youngs_threshold}_youngs_threshold_{decision_time}_decision_time_{num_decisions_per_episode}_num_decisions_per_episode_{target_observable}_target_observable_{tau}_tau_{num_explore_epsiodes}_num_explore_epsiodes"
        train_folder_name = root_train_folder_name + tag + str("/")
        # Get all folder_names that are ints in train_folder_name
        all_folder_names = np.sort([x for x in os.listdir(train_folder_name)
                                    if x.isdigit()])
        epoch_to_load = all_folder_names[-2]
        epoch_folder_name = train_folder_name + epoch_to_load + "/"

        rl_model = QModel(state_dim=env_int.num_states, action_dim=len(all_actions),
                          tau=tau,
                          out_scaling=out_scaling,
                          device=device)

        print("Loading model from: ", epoch_folder_name)
        rl_model.load_networks(folder_name=epoch_folder_name)
        print("Loaded model from: ", epoch_folder_name)
        all_rl_models[target_observable] = rl_model

    rbuffer = None
    gamma = None
    num_decisions_per_episode = num_decisions_per_episode

    layered_tag = tag = f"{decision_time}_decision_time_{num_decisions_per_episode}_num_decisions_per_episode_{tau}_tau_{num_explore_epsiodes}_num_explore_epsiodes_"
    layered_tag += "_".join(map(str, target_observables)
                            ) + "_target_observables"
    layered_tag = layered_tag + str("/")
    try:
        os.mkdir(layered_tag)
    except FileExistsError:
        pass

    layered_tag += f"{run_num}" + str("/")

    try:
        os.mkdir(layered_tag)
    except FileExistsError:
        pass

    folder_name = layered_tag

    cdql = CDQL(env_int=env_int, rl_model=rl_model, rbuffer=rbuffer,
                writer=None, gamma=gamma, num_decisions_per_episode=num_decisions_per_episode,
                all_actions=all_actions, num_explore_episodes=num_explore_epsiodes, device=device,
                batch_size=32,
                folder_name=folder_name)
    curr_state = None
    for (layer_num, target_observable) in enumerate(target_observables):
        time_per_episode = num_decisions_per_episode_by_layer[layer_num]
        curr_state = cdql.make_layered(layer_num=layer_num,
                                       target_observable=target_observable,
                                       rl_model=all_rl_models[target_observable],
                                       num_decisions_per_episode=time_per_episode,
                                       curr_state=curr_state)
