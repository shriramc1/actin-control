from actin.solver import LiteratureConstants, Solver
import numpy as np


class EnvironmentInterface:
    def __init__(self):
        pass

    def _get_reward(self, info):
        raise NotImplementedError()

    def take_action(self, action):
        raise NotImplementedError()


class ActinUDInterface(EnvironmentInterface):
    def __init__(self, decision_time=10,
                 target_observable=5, all_burn_in_forces=[0.5], youngs_threshold=50,
                 target_observable_name="upper_density", compute_youngs_modulus=False):
        """Interface for actin polymerization environment
        Arguments:
            decision_time {float} -- time to run simulation for
            target_observable {float} -- target observable to reach
            all_burn_in_forces {list} -- list of burn in forces to use
            youngs_threshold {float} -- threshold for youngs modulus
            target_observable_name {str} -- name of target observable
            compute_youngs_modulus {bool} -- whether to compute youngs modulus
        """

        self.decision_time = decision_time
        self.target_observable = target_observable
        self.num_states = 3

        constants = LiteratureConstants()
        num_nucleation_sites = 100
        box_length_scale = 1000
        num_monomers = 500000
        num_cps = 100000
        num_arp23 = 100000
        interface_height = 0.0
        arp23_binding_distance_scaled = 10
        num_npfs = 30
        self.target_observable_name = target_observable_name

        self.init_solver = lambda _: Solver(constants=constants,
                                            num_nucleation_sites=num_nucleation_sites,
                                            box_length_scale=box_length_scale, num_monomers=num_monomers,
                                            num_arp23=num_arp23, arp23_binding_distance_scaled=arp23_binding_distance_scaled,
                                            num_cps=num_cps, num_npfs=num_npfs,
                                            interface_height=interface_height,
                                            youngs_threshold=youngs_threshold,
                                            compute_youngs_modulus=compute_youngs_modulus)

        self.all_traj = None
        self.all_plotting_info = None

        self.all_burn_in_forces = all_burn_in_forces

    def _get_reward(self, info):
        """Returns reward based on target observable
        Arguments:
            info {float} -- current observable
        Returns:
            float -- reward
        """
        curr_observable = info
        if self.target_observable_name == "youngs_modulus":
            reward = np.abs(self.target_observable - curr_observable)/5 - 1
            reward = np.clip(reward, 0, 8)
        else:
            reward = np.abs(self.target_observable - curr_observable)

        return reward

    def _get_state_from_traj(self, traj):
        """Returns state from trajectory
        Arguments:
            traj {dict} -- trajectory
        Returns:
            list -- state
        """
        state = [traj["num_bound_monomers_tirf"]/1000, traj["num_bound_arp23_tirf"]/10,
                 traj["num_bound_cps_tirf"]/10]  # , traj["network_max_height"]]
        return state

    def update_target_observable(self, new_target_observable):
        """Updates target observable
        Arguments:
            new_target_observable {float} -- new target observable
        """
        self.target_observable = new_target_observable

    def init_system(self, return_network_height=False):
        """Initializes system
        Arguments:
            return_network_height {bool} -- whether to return network height
        Returns:
            list -- state
        """
        self.s = self.init_solver(None)
        self.all_traj = []
        self.all_plotting_info = []
        burn_in_force = np.random.choice(self.all_burn_in_forces)
        self.s.update_interface_drift(new_interface_drift=burn_in_force)
        traj, _ = self.s.solve(sim_time=20,
                               max_steps=1000000,
                               save_freq=None)
        curr_state = self._get_state_from_traj(traj[-1])

        self.all_traj.append(traj)
        self.all_plotting_info.append(self.s.get_plotting_info())
        return curr_state

    def init_system_to_height(self, height_to_grow = 0.1, return_network_height=False):
        """Initializes system to a height
        Arguments:
            height_to_grow {float} -- height to grow to
            return_network_height {bool} -- whether to return network height
        Returns:
            list -- state
        """

        self.s = self.init_solver(None)
        self.all_traj = []
        self.all_plotting_info = []
        burn_in_force = np.max(self.all_burn_in_forces)
        self.s.update_interface_drift(new_interface_drift=burn_in_force)
        height = 0
        while height < height_to_grow:
            traj, _ = self.s.solve(sim_time=1,
                                   max_steps=1000000,
                                   save_freq=100)
            height = traj[-1]["network_max_height"]
        
        curr_state = self._get_state_from_traj(traj[-1])
        self.all_traj.append(traj)
        self.all_plotting_info.append(self.s.get_plotting_info())
        return curr_state

    def take_action(self, action, return_network_height=False):
        """Takes action and returns current state
        Arguments:
            action {float} -- action to take
            return_network_height {bool} -- whether to return network height
        Returns:
            list -- state
            float -- reward
            float -- network height (if return_network_height is True)
        """
        self.s.update_interface_drift(new_interface_drift=action)
        traj, _ = self.s.solve(sim_time=self.decision_time,
                               max_steps=1000000,
                               save_freq=100)

        curr_observable = traj[-1][self.target_observable_name]
        reward = self._get_reward(info=curr_observable)
        curr_state = self._get_state_from_traj(traj[-1])
        self.all_traj.append(traj)
        self.all_plotting_info.append(self.s.get_plotting_info())
        if not return_network_height:
            return curr_state, reward
        else:
            network_height = traj[-1]["network_max_height"]
            return curr_state, reward, network_height

    def append_stiffness_matrix_to_traj(self):
        """Appends stiffness matrix to trajectory
        """
        stiffness_matrix, all_node_pos, all_edges, all_distances, all_unit_vectors = self.s.get_stiffness_matrix()
        traj_to_append = {"stiffness_matrix": stiffness_matrix,
                          "all_node_pos": all_node_pos,
                          "all_edges": all_edges,
                          "all_distances": all_distances,
                          "all_unit_vectors": all_unit_vectors}
        self.all_traj[-1][-1] = (self.all_traj[-1][-1] | traj_to_append)

    def get_curr_network_height(self):
        """Returns current network height
        Returns:
            float -- network height
        """
        return self.all_traj[-1][-1]["network_max_height"]

    def save_episode(self, folder_name):
        """Saves episode
        Arguments:
            folder_name {str} -- folder name
        """
        np.save(folder_name + "all_traj.npy",
                np.array(self.all_traj, dtype=object))
        np.save(folder_name + "all_plotting_info.npy",
                np.array(self.all_plotting_info, dtype=object))
