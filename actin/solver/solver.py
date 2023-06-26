import numpy as np
import random
from .actin import ActinNetwork, Interface
import warnings


class Solver:
    def __init__(self, constants, num_nucleation_sites=100,
                 box_length_scale=4,
                 num_monomers=5000, interface_height=10.,
                 arp23_binding_distance_scaled=5, num_npfs=25,
                 num_arp23=100, num_cps=10000, youngs_threshold=50, compute_youngs_modulus=False):
        """
        Arguments:
            constants: Constants object
            num_nucleation_sites: Number of nucleation sites
            box_length_scale: Box length scale (box length = box_length_scale * monomer_diameter)
            num_monomers: Number of monomers
            interface_height: Interface height
            arp23_binding_distance_scaled: Arp2/3 binding distance scaled (distance = arp23_binding_distance_scaled * monomer_diameter)
            num_npfs: Number of nucleation promoting factors
            num_arp23: Number of Arp2/3
            num_cps: Number of capping proteins
            youngs_threshold: Young's threshold
            compute_youngs_modulus: Compute Young's modulus
        """
        self.constants = constants
        box_length = np.array(box_length_scale
                              * self.constants.monomer_diameter)
        arp23_binding_distance = (arp23_binding_distance_scaled
                                  * self.constants.monomer_diameter)
        self.actin_network = ActinNetwork(num_nucleation_sites=num_nucleation_sites,
                                          monomer_diameter=self.constants.monomer_diameter,
                                          cp_diameter=self.constants.cp_diameter,
                                          box_length=box_length, num_monomers=num_monomers,
                                          num_cps=num_cps, num_npfs=num_npfs,
                                          num_arp23=num_arp23, arp23_binding_distance=arp23_binding_distance,
                                          tirf_distance_height=self.constants.tirf_distance_height)
        self.interface = Interface(height=interface_height,
                                   interface_diff=self.constants.interface_diff,
                                   interface_drift_scale=self.constants.interface_drift_scale)
        self.kp_on = np.array([self.constants.kp_on])
        self.kp_off = np.array([self.constants.kp_off])
        self.arp23_on = np.array([self.constants.arp23_on])
        self.cp_on = np.array([self.constants.cp_on])
        self.youngs_threshold = youngs_threshold
        self.compute_youngs_modulus = compute_youngs_modulus

    def update_interface_drift(self, new_interface_drift):
        """Update interface drift
        Arguments:
            new_interface_drift: New interface drift
        """
        self.interface.set_interface_drift(
            new_interface_drift=new_interface_drift)

    def get_plotting_info(self):
        """Get plotting information
        Returns:
            tuple -- all_monomer_positions, bound_arp23_pos, bound_capping_pos, interface_height
        """
        all_monomer_positions, bound_arp23_pos, bound_capping_pos = self.actin_network.get_detailed_traj_information()
        return all_monomer_positions, bound_arp23_pos, bound_capping_pos, self.interface.height

    def get_upper_density(self):
        """Gets growth front density (upper density)
        Returns:
            float -- upper density
        """
        return self.actin_network.get_upper_monomer_density(interface_height=self.interface.height)

    def get_tirf_measurements(self):
        """Gets TIRF measurements
        Returns:
            tuple -- num_bound_monomers_tirf, num_bound_arp23_tirf, num_bound_cps_tirf
        """
        num_bound_monomers_tirf, num_bound_arp23_tirf, num_bound_cps_tirf = self.actin_network.get_tirf_measurements(
            interface_height=self.interface.height)
        return num_bound_monomers_tirf, num_bound_arp23_tirf, num_bound_cps_tirf

    def compute_youngs_modulus(self, network_max_height, return_stress_height=False):
        """Computes Young's modulus
        Arguments:
            network_max_height: Network max height
            return_stress_height: Return stress height
        Returns:
            float -- Young's modulus
        """
        fixed_threshold = (network_max_height
                           - self.youngs_threshold * self.constants.monomer_diameter)
        youngs_modulus, all_all_node_pos_j, all_edges, all_distances, all_unit_vectors, stress, height = self.actin_network.get_youngs_modulus(y_threshold=0.1,
                                                                                                                                               force_threshold=0.01,
                                                                                                                                               fixed_threshold=fixed_threshold)
        if return_stress_height:
            return youngs_modulus, stress, height
        else:
            return youngs_modulus

    def get_stiffness_matrix(self):
        """Gets stiffness matrix
        Returns:
            array -- stiffness matrix
        """
        return self.actin_network.compute_stiffnes_matrix()

    def solve(self, sim_time=10, max_steps=10000000, save_freq=10):
        """Solves the system
        Arguments:
            sim_time {float} -- Simulation time (default: {10})
            max_steps {int} -- Maximum number of steps (default: {10000000})
            save_freq {int} -- Save frequency (default: {10})
        Returns:
            tuple -- traj, all_t
        Raises:
            ValueError: Maximum time exceeded number of allotted steps
        """

        u1 = np.random.rand(max_steps)  # Randomly pick time step
        u2 = np.random.rand(max_steps)  # Randomly pick time reaction
        monomer_binding_heat = np.log(((self.kp_on * self.constants.actin_concentration)
                                       / self.kp_off))
        old_num_bound_monomers = self.actin_network.get_num_bound_monomers()
        all_t = []
        all_heat = []
        all_work = []
        traj = []
        for step in range(max_steps):
            if ((step % 1000) == 0):
                print(step)
            self.actin_network.update_arp23able_monomers(
                interface_height=self.interface.height)
            num_free_barbed_ends = self.actin_network.get_num_free_barbed_ends(
                interface_height=self.interface.height)

            effective_arp23_num = self.actin_network.num_npfs * max(0,
                                                                    (self.actin_network.num_npfs - num_free_barbed_ends)/self.actin_network.num_npfs)

            # Get num filaments that can bind monomer (Rxn 0)
            num_bindable_filaments = len(self.actin_network.bindable_filaments)

            # Get num filaments that can unbind monomer (Rxn 1)
            num_unbindable_filaments = len(
                self.actin_network.unbindable_filaments)

            # Get all monomers that can bind arp2/3 (Rxn 2)
            arp23_binding_sites = len(
                self.actin_network.all_arp23able_monomers)

            # Get all filaments that can bind capping protein (Rxn 3)
            cp_binding_sites = len(self.actin_network.cappable_filaments)

            rates = np.concatenate([self.kp_on * num_bindable_filaments * self.constants.actin_concentration,
                                    self.kp_off * num_unbindable_filaments,
                                    (self.arp23_on * arp23_binding_sites *
                                     effective_arp23_num * self.constants.arp23_concentration),
                                    self.cp_on * cp_binding_sites * self.constants.cp_concentration])
            all_t.append((np.log(1 / u1[step]) / np.sum(rates)))
            rxn_idx = (((np.cumsum(rates, axis=-1) / rates.sum())
                        - u2[step]) <= 0).sum()
            if rates.sum() == 0:
                """No possible reactions (shouldn't happen)
                """
                print(num_bindable_filaments)
                print(num_unbindable_filaments)
                print(arp23_binding_sites)
                print(cp_binding_sites)
                network_max_height = self.actin_network.all_high_filaments[0].p_height
                traj.append([self.actin_network.get_traj_information(),
                             self.interface.height.copy(),
                             network_max_height])
                raise ValueError("No reaction selected (Shouldn't Happen)")


            if (rxn_idx == 0):
                """Add monomer to filament
                """
                filament = random.choice(self.actin_network.bindable_filaments)
                try:
                    free_monomer = self.actin_network.all_free_monomers.pop(0)
                except IndexError:
                    raise ValueError("Increase number of monomers initialized")

                filament.add_monomer(free_monomer)
   
                self.actin_network.all_bound_monomers.append(free_monomer)
                self.actin_network.all_monomers_near_interface.append(free_monomer)
            elif (rxn_idx == 1):
                """Unbind monomer from filament
                """
                filament = random.choice(
                    self.actin_network.unbindable_filaments)
                unbound_monomer = filament.unbind_monomer()

                self.actin_network.all_bound_monomers.remove(unbound_monomer)
                self.actin_network.all_monomers_near_interface.remove(unbound_monomer)
                self.actin_network.all_free_monomers.append(unbound_monomer)
            elif (rxn_idx == 2):
                """Add arp2/3
                """
                monomer = random.choice(
                    self.actin_network.all_arp23able_monomers)
                arp23 = self.actin_network.all_free_arp23.pop(0)
                try:
                    free_monomer = self.actin_network.all_free_monomers.pop(0)
                except IndexError:
                    raise ValueError("Increase number of Arp2/3 initialized")

                added_filament = self.actin_network.add_branched_filament(
                    branching_monomer=monomer)
                root_filament = self.actin_network.all_filaments[monomer.filament_id]
                root_filament.add_connected_filament(
                    added_filament.filament_id)
                added_filament.add_connected_filament(
                    root_filament.filament_id)

                monomer.bind_arp23(arp23)
                arp23.bind(monomer,
                           branch_filament_id=added_filament.filament_id)
                self.actin_network.all_bound_arp23.append(arp23)

            elif (rxn_idx == 3):
                """Add a capping protein
                """
                filament = random.choice(self.actin_network.cappable_filaments)
                try:
                    free_cp = self.actin_network.all_free_cps.pop(0)
                except IndexError:
                    raise ValueError("Increase number of capping protein initialized")

                filament.add_capping_protein(cp_to_add=free_cp)
                self.actin_network.all_bound_cps.append(free_cp)
                
                self.actin_network.all_uncapped_filaments.remove(filament)
            else:
                raise ValueError("Reaction not defined")
            
            if step % 1000 == 0:
                self.actin_network.prune_all_monomers_near_interface(interface_height=self.interface.height)
                self.actin_network.prune_all_high_filaments(interface_height=self.interface.height)
            


            
            self.actin_network.all_high_filaments.sort(key=lambda x: x.p_height,
                                                       reverse=True)
            network_max_height = self.actin_network.all_high_filaments[0].p_height

            old_interface_height = self.interface.height
            self.interface.move_boundary(all_t[-1], network_max_height)
            new_interface_height = self.interface.height
            work = self.interface.get_interface_drift() * (new_interface_height -
                                                           old_interface_height)
            all_work.append(work)

            num_bound_monomers = self.actin_network.get_num_bound_monomers()
            heat = ((num_bound_monomers - old_num_bound_monomers)
                    * monomer_binding_heat)
            all_heat.append(heat)
            old_num_bound_monomers = num_bound_monomers

            self.actin_network.update_binding_behavior(max_height=self.interface.height,
                                                       min_height=0, monomer_diameter=self.constants.monomer_diameter,
                                                       cp_diameter=self.constants.cp_diameter)


            self.actin_network.update_arp23able_monomers(interface_height=self.interface.height)



            if (np.sum(all_t) >= sim_time):
                num_bound_arp23, num_bound_capping, num_bound_monomers, num_monomers_by_filament, all_filament_polarities = self.actin_network.get_traj_information()
                num_bound_monomers_tirf, num_bound_arp23_tirf, num_bound_cps_tirf = self.get_tirf_measurements()
                traj_i = {"num_bound_arp23": num_bound_arp23,
                          "upper_density": self.get_upper_density(),
                          "num_bound_capping": num_bound_capping,
                          "num_bound_monomers": num_bound_monomers,
                          "num_monomers_by_filament": num_monomers_by_filament,
                          "all_filament_polarities": all_filament_polarities,
                          "num_free_barbed_ends": num_free_barbed_ends,
                          "interface_height": self.interface.height,
                          "network_max_height": network_max_height,
                          "num_bound_monomers_tirf": num_bound_monomers_tirf,
                          "num_bound_arp23_tirf": num_bound_arp23_tirf,
                          "num_bound_cps_tirf": num_bound_cps_tirf,
                          "rxn_propensities": rates,
                          "step": step,
                          "curr_t": np.sum(all_t),
                          "all_heat_rate": all_heat,
                          "all_work_rate": all_work,
                          "all_t": all_t}
                traj.append(traj_i)
            if np.sum(all_t) > sim_time:
                return traj, all_t
        raise ValueError("Maximum time exceeded number of allotted steps")
