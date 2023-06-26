import os
import torch
import torch.nn as nn
import numpy as np
import random
import copy


class CDQL:
    def __init__(self, env_int, rl_model, rbuffer, writer,
                 gamma, num_decisions_per_episode, all_actions, num_explore_episodes,
                 batch_size=16,
                 device=torch.device("cuda:0"), folder_name="./"):
        """Initialize CDQL
        Arguments:
            env_int {EnvInt} -- environment interface
            rl_model {RLModel} -- rl model
            rbuffer {ReplayBuffer} -- replay buffer
            writer {SummaryWriter} -- writer
            gamma {float} -- discount factor
            num_decisions_per_episode {int} -- number of decisions per episode
            all_actions {list} -- list of all actions
            num_explore_episodes {int} -- number of episodes to explore for
            batch_size {int} -- batch size
            device {torch.device} -- device
            folder_name {str} -- folder name
        """

        self.env_int = env_int
        self.rl_model = rl_model
        self.gamma = gamma
        self.all_actions = all_actions
        self.num_decisions_per_episode = num_decisions_per_episode
        self.num_explore_episodes = num_explore_episodes
        self.batch_size = batch_size

        self.rbuffer = rbuffer

        self.writer = writer
        self.folder_name = folder_name
        self.device = device

        self.all_actions = all_actions
        self.num_actions = len(self.all_actions)

    def _update_model(self, epoch):
        """Update model
        Arguments:
            epoch {int} -- epoch
        """
        if (len(self.rbuffer) < self.batch_size):
            return
        epoch -= self.batch_size
        assert not self.rl_model.q_target_1.training
        assert not self.rl_model.q_target_2.training

        assert self.rl_model.q_1.training
        assert self.rl_model.q_2.training

        transitions = self.rbuffer.sample(self.batch_size)
        batch = self.rbuffer.transition(*zip(*transitions))

        state_batch = torch.tensor(batch.state, device=self.device).float()
        action_batch = torch.tensor(batch.action,
                                    device=self.device).unsqueeze(-1).long()
        reward_batch = torch.tensor(batch.reward,
                                    device=self.device).unsqueeze(-1).float()
        next_state_batch = torch.tensor(batch.next_state,
                                        device=self.device).float()
        is_done_batch = torch.tensor(batch.done,
                                     device=self.device).unsqueeze(-1).bool()

        with torch.no_grad():
            Q_next_1 = ((~is_done_batch)
                        * (self.rl_model.q_target_1(next_state_batch).min(dim=-1)[0].unsqueeze(-1)))
            Q_next_2 = ((~is_done_batch)
                        * (self.rl_model.q_target_2(next_state_batch).min(dim=-1)[0].unsqueeze(-1)))

            # Use max want to avoid underestimation bias
            Q_next = torch.max(Q_next_1, Q_next_2)
            Q_expected = reward_batch + self.gamma * Q_next

        Q_1 = self.rl_model.q_1(state_batch).gather(-1, action_batch)
        Q_2 = self.rl_model.q_2(state_batch).gather(-1, action_batch)
        l_1 = nn.MSELoss(reduction="mean")(Q_1, Q_expected)
        l_2 = nn.MSELoss(reduction="mean")(Q_2, Q_expected)

        self.rl_model.q_optimizer_1.zero_grad()
        self.rl_model.q_optimizer_2.zero_grad()
        l_1.backward()
        l_2.backward()
        self.rl_model.q_optimizer_1.step()
        self.rl_model.q_optimizer_2.step()

        # Get estimate for optimal q value

        last_experience = self.rbuffer.get_element_from_buffer(
            self.rbuffer.position - 1)
        last_state = last_experience.state

        self.rl_model.q_1.eval()
        self.rl_model.q_2.eval()
        with torch.no_grad():
            last_state = torch.tensor(last_state,
                                      device=self.device).float().unsqueeze(0)
            q1_value = torch.min(self.rl_model.q_1(last_state), dim=-1)[0]
            q2_value = torch.min(self.rl_model.q_2(last_state), dim=-1)[0]
        self.rl_model.q_1.train()
        self.rl_model.q_2.train()

        self.writer.add_scalar("Q1MSELoss", l_1.item(), epoch)
        self.writer.add_scalar("Q2MSELoss", l_2.item(), epoch)
        self.writer.add_scalar("Q1", q1_value.item(), epoch)
        self.writer.add_scalar("Q2", q2_value.item(), epoch)
        self.rl_model.update_target_nn()

    def _get_action_index(self, state, episode):
        """Get action index
        Arguments:
            state {list} -- state
            episode {int} -- episode
        Returns:
            int -- action index
        """
        epsilon = 0.05
        if episode >= 0:
            # train
            if episode < self.num_explore_episodes:
                # return self.all_actions[episode % self.num_actions]
                action_index = episode % self.num_actions
                return action_index
            if (np.random.rand() < epsilon):
                action_index = random.choice(range(len(self.all_actions)))
                return action_index
        # Follow policy if not training or if training and (episode > 5 or if random number is greater than epsilon)
        self.rl_model.q_1.eval()
        with torch.no_grad():
            state = torch.tensor(state,
                                 device=self.device).float().unsqueeze(0)
            action_index = torch.argmin(self.rl_model.q_1(state),
                                        dim=-1).item()
        self.rl_model.q_1.train()
        return action_index

    def _save_info(self, episode_rewards, episode_actions, episode_states, episode):
        """Save info
        Arguments:
            episode_rewards {list} -- episode rewards
            episode_actions {list} -- episode actions
            episode_states {list} -- episode states
            episode {int} -- episode
        """
        save_folder_name = self.folder_name + str(episode) + "/"
        try:
            os.mkdir(save_folder_name)
        except FileExistsError:
            pass

        if self.rbuffer is not None:
            filename = save_folder_name + "replaybuffer.npy"
            np.save(filename, np.array(self.rbuffer.buffer, dtype=object))

        reward_filename = save_folder_name + "episode_rewards.npy"
        np.save(reward_filename, np.array(episode_rewards))
        action_filename = save_folder_name + "episode_actions.npy"
        np.save(action_filename, np.array(episode_actions))
        state_filename = save_folder_name + "episode_states.npy"
        np.save(state_filename, np.array(episode_states))

        self.rl_model.save_networks(save_folder_name)

    def train(self):
        """Train CDQL
        """
        for episode in range(5000):
            try:
                curr_state = self.env_int.init_system()
            except ValueError:
                break
            episode_actions = []
            episode_rewards = []
            episode_states = []
            for decision in range(self.num_decisions_per_episode):
                episode_states.append(curr_state)
                action_index = self._get_action_index(state=curr_state,
                                                      episode=episode)
                action = self.all_actions[action_index]
                try:
                    next_state, reward = self.env_int.take_action(action)
                except ValueError:
                    break
                episode_actions.append(action)
                episode_rewards.append(reward)
                transition_to_add = [curr_state,
                                     action_index, reward, next_state, False]
                self.rbuffer.push(*transition_to_add)
                self._update_model(epoch=((episode * self.num_decisions_per_episode)
                                          + decision))
                curr_state = copy.deepcopy(next_state)

            if episode % 1 == 0:
                self._save_info(episode_actions=episode_actions,
                                episode_rewards=episode_rewards,
                                episode_states=episode_states,
                                episode=episode)
                self.env_int.save_episode(self.folder_name
                                          + "/" + str(episode) + "/")

    def test(self, num_explore_episodes=10000, num_decisions_per_epsiode=200):
        """Test CDQL
        Arguments:
            num_explore_episodes {int} -- number of episodes to explore for
            num_decisions_per_epsiode {int} -- number of decisions per episode
        """
        
        for episode in range(num_explore_episodes):
            curr_state = self.env_int.init_system()
            episode_actions = []
            episode_rewards = []
            episode_states = []
            for decision in range(num_decisions_per_epsiode):
                episode_states.append(curr_state)
                action_index = self._get_action_index(state=curr_state,
                                                      episode=-1)
                action = self.all_actions[action_index]
                try:
                    next_state, reward = self.env_int.take_action(action)
                except ValueError:
                    break
                episode_actions.append(action)
                episode_rewards.append(reward)
                curr_state = copy.deepcopy(next_state)
            self._save_info(episode_actions=episode_actions,
                            episode_rewards=episode_rewards,
                            episode_states=episode_states,
                            episode=episode)
            self.env_int.save_episode(self.folder_name
                                      + "/" + str(episode) + "/")

    def _update_rl_model(self, new_rl_model):
        """Update rl model
        Arguments:
            new_rl_model {RLModel} -- new rl model
        """
        self.rl_model = new_rl_model

    def make_layered_by_time(self, all_action_indices):
        layer_num = 0
        curr_state = self.env_int.init_system_to_height(height_to_grow=0.5)
        episode_actions = []
        episode_rewards = []
        episode_states = []
        for action_index in all_action_indices:
            episode_states.append(curr_state)
            action = self.all_actions[action_index]
            print(action, action_index)
            try:
                next_state, reward = self.env_int.take_action(action)
            except ValueError:
                break
            episode_actions.append(action)
            episode_rewards.append(reward)
            curr_state = copy.deepcopy(next_state)
        self._save_info(episode_actions=episode_actions,
                        episode_rewards=episode_rewards,
                        episode_states=episode_states,
                        episode=layer_num)
        self.env_int.append_stiffness_matrix_to_traj()
        self.env_int.save_episode(self.folder_name
                                  + "/" + str(layer_num) + "/")






    def make_layered(self, layer_num,
                     target_observable, 
                     rl_model, 
                     num_decisions_per_episode, curr_state=None):
        """Make layered
        Arguments:
            layer_num {int} -- layer number
            target_observable {str} -- target observable
            rl_model {RLModel} -- rl model
            num_decisions_per_episode {int} -- number of decisions per episode
            curr_state {list} -- current state
        Returns:
            list -- current state
        """
        self.env_int.update_target_observable(
            new_target_observable=target_observable)
        self._update_rl_model(new_rl_model=rl_model)
        if curr_state is None:
            curr_state = self.env_int.init_system_to_height(height_to_grow=0.5)
        episode_actions = []
        episode_rewards = []
        episode_states = []
        for decision in range(num_decisions_per_episode):
            episode_states.append(curr_state)
            action_index = self._get_action_index(state=curr_state,
                                                  episode=-1)
            action = self.all_actions[action_index]
            try:
                next_state, reward = self.env_int.take_action(action)
            except ValueError:
                break
            episode_actions.append(action)
            episode_rewards.append(reward)
            curr_state = copy.deepcopy(next_state)
        self._save_info(episode_actions=episode_actions,
                        episode_rewards=episode_rewards,
                        episode_states=episode_states,
                        episode=layer_num)
        self.env_int.append_stiffness_matrix_to_traj()
        self.env_int.save_episode(self.folder_name
                                  + "/" + str(layer_num) + "/")
        return curr_state
    
    def make_layered_by_height(self, layer_num,
                               target_observable, 
                               rl_model, 
                               height_to_grow, curr_state=None):
        """Make layered networks by height
        Arguments:
            layer_num {int} -- layer number
            target_observable {str} -- target observable
            rl_model {RLModel} -- rl model
            height_to_grow {float} -- height to grow
            curr_state {list} -- current state
        Returns:
            list -- current state
        """
        self.env_int.update_target_observable(
            new_target_observable=target_observable)
        self._update_rl_model(new_rl_model=rl_model)
        if curr_state is None:
            curr_state = self.env_int.init_system_to_height(height_to_grow=0.5)
        episode_actions = []
        episode_rewards = []
        episode_states = []

        curr_network_height = self.env_int.get_curr_network_height()
        target_network_height = curr_network_height + height_to_grow
        while curr_network_height < target_network_height:
            episode_states.append(curr_state)
            action_index = self._get_action_index(state=curr_state,
                                                  episode=-1)
            action = self.all_actions[action_index]
            try:
                next_state, reward = self.env_int.take_action(action)
            except ValueError:
                break
            episode_actions.append(action)
            episode_rewards.append(reward)
            curr_state = copy.deepcopy(next_state)
            curr_network_height = self.env_int.get_curr_network_height()
        

        self.env_int.append_stiffness_matrix_to_traj()
        self._save_info(episode_actions=episode_actions,
                        episode_rewards=episode_rewards,
                        episode_states=episode_states,
                        episode=layer_num)
        self.env_int.save_episode(self.folder_name
                                  + "/" + str(layer_num) + "/")
        return curr_state