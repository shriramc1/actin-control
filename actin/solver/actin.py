import numpy as np
import random
from scipy import special
import scipy.optimize as opt
from .utils import get_intersection, get_all_lines, get_all_intersections


class Interface:
    def __init__(self, height,
                 interface_diff,
                 interface_drift_scale):
        self.height = height
        self.interface_diff = interface_diff
        self.interface_drift_scale = interface_drift_scale

    def set_interface_drift(self, new_interface_drift):
        """Sets interface drift
        Arguments:
            new_interface_drift {float} -- new interface drift
        """
        self.interface_drift = self.interface_drift_scale * new_interface_drift

    def get_interface_drift(self):
        """Returns interface drift
        Returns:
            float -- interface drift
        """
        return self.interface_drift

    def move_boundary(self, dt, network_max_height):
        """Moves the boundary
        Arguments:
            dt {float} -- time step
            network_max_height {float} -- maximum height of network
        """
        dh = (self.interface_drift * dt
              + np.random.randn() * np.sqrt(2 * self.interface_diff * dt))
        x0 = self.height - network_max_height
        new_height = self.height + dh

        if new_height < network_max_height:
            """Reflect
            """
            def nfun(q):
                r = np.random.uniform(0,
                                      0.5 * special.erfc((x0 + self.interface_drift * dt)
                                                         / (np.sqrt(4 * self.interface_diff * dt))),
                                      1)

                fun = (0.5 * special.erfc((x0 + self.interface_drift * dt)/(np.sqrt(4 * self.interface_diff * dt)))
                       - (0.5
                           * np.exp(q * self.interface_drift/self.interface_diff)
                           * special.erfc((q + x0 + (self.interface_drift*dt))
                                          / (np.sqrt(4 * self.interface_diff * dt))))
                       - r)
                return fun
            r1 = opt.brentq(lambda x: nfun(x), 0, 1)
            self.height = r1 + network_max_height
        else:
            self.height = new_height

    def move_boundary_absorbing(self, dt, network_max_height):
        """Moves the boundary with absorbing boundary condition
        Arguments:
            dt {float} -- time step
            network_max_height {float} -- maximum height of network
        """
        F = self.interface_drift * dt  # assume a constant drift force
        dh = np.random.randn() * (2 * self.interface_diff * np.sqrt(dt)) + F
        self.height += dh

        if self.height < network_max_height:
            """
            Reflect
            """
            self.height = network_max_height
            # self.height = (2 * network_max_height - self.height)


class CappingProtein:
    def __init__(self, cp_id, diameter=0.1):
        """
        Arguments:
            cp_id {int} -- capping protein id
            diameter {float} -- diameter of capping protein (default: {0.1})
        """
        self.cp_id = cp_id
        self.diameter = diameter
        self.position = None
        self.filament_id = None

    def bind_filament(self, filament_id, position):
        """Adds capping protein to filament
        Arguments:
            filament_id {int} -- filament id
            position {np.array} -- position of capping protein
        """
        assert self.filament_id is None
        assert self.position is None

        self.filament_id = filament_id
        self.position = position


class ActinMonomer:
    def __init__(self, monomer_id, diameter=0.1):
        """
        Arguments:
            monomer_id {int} -- monomer id
            diameter {float} -- diameter of monomer (default: {0.1})
        """
        self.monomer_id = monomer_id
        self.diameter = diameter
        self.position = None
        self.filament_id = None
        self.arp23 = None

    def bind_filament(self, filament_id, position):
        """Adds monomer to filament
        Arguments:
            filament_id {int} -- filament id
            position {np.array} -- position of monomer
        """
        assert self.filament_id is None
        assert self.position is None
        assert self.arp23 is None

        self.filament_id = filament_id
        self.position = position

    def unbind_filament(self):
        """Removes monomer from filament
        """
        assert self.filament_id is not None
        assert self.position is not None
        assert self.arp23 is None

        self.filament_id = None
        self.position = None

    def bind_arp23(self, arp23):
        """Binds arp23 to monomer
        """
        assert self.arp23 is None
        assert self.filament_id is not None
        assert self.position is not None
        self.arp23 = arp23

    def unbind_arp23(self):
        """Unbinds arp23 from monomer
        """
        assert self.arp23 is not None
        self.arp23 = None


class ARP23:
    def __init__(self, arp23_id):
        """
        Arguments:
            arp23_id {int} -- arp23 id
        """
        self.arp23_id = arp23_id
        self.branch_filament_id = None
        self.bound_monomer = None

    def bind(self, monomer, branch_filament_id):
        """Binds arp23 to monomer
        Arguments:
            monomer {ActinMonomer} -- monomer to bind to
            branch_filament_id {int} -- filament id of branch
        """
        assert self.bound_monomer is None
        assert self.branch_filament_id is None

        self.bound_monomer = monomer
        self.branch_filament_id = branch_filament_id

    def unbind(self):
        """Unbinds arp23 from monomer
        """
        assert self.bound_monomer is not None
        assert self.branch_filament_id is not None

        self.bound_monomer = None

    def get_position(self):
        """Returns position of bound monomer
        Returns:
            np.array -- position of bound monomer
        """
        if self.bound_monomer is None:
            return None
        else:
            return self.bound_monomer.position


class ActinFilament:
    def __init__(self, m_end, filament_id, init_angle,
                 box_length,
                 is_surface=True):
        """
        Arguments:
            m_end {np.array} -- position of m_end
            filament_id {int} -- filament id
            init_angle {float} -- initial angle of filament
            box_length {float} -- box length
            is_surface {bool} -- whether filament is surface filament (default: {True})
        """
        self.all_monomers = []
        self.cp = None

        self.m_end = m_end
        self.p_end = self.m_end.copy()
        self.p_height = self.p_end[1]
        self.box_length = box_length
        self.filament_id = filament_id
        self.unit_vector = np.stack([np.sin(init_angle),
                                     np.cos(init_angle)])
        self.is_surface = is_surface
        self.can_unbind_monomer = False
        self.can_bind_monomer = True
        self.can_bind_cp = False

        self.angle = init_angle

        self.all_connected_filaments = []

    def add_monomer(self, monomer_to_add):
        """Adds monomer to filament
        Arguments:
            monomer_to_add {ActinMonomer} -- monomer to add
        """
        new_pos = self.p_end + self.unit_vector * monomer_to_add.diameter

        new_pos[0] -= (new_pos[0] >= self.box_length) * self.box_length
        new_pos[0] += (new_pos[0] < 0) * self.box_length

        self.p_end = new_pos
        monomer_position = self.p_end.copy()
        self.p_height = self.p_end[1]
        monomer_to_add.bind_filament(self.filament_id, monomer_position)
        self.all_monomers.append(monomer_to_add)

    def unbind_monomer(self):
        """Unbinds monomer from filament
        """
        assert self.cp is None

        monomer_to_unbind = self.all_monomers[-1]
        new_pos = self.p_end - self.unit_vector * monomer_to_unbind.diameter

        new_pos[0] -= (new_pos[0] > self.box_length) * self.box_length
        new_pos[0] += (new_pos[0] < 0) * self.box_length

        self.p_end = new_pos
        self.p_height = self.p_end[1]
        monomer_to_unbind.unbind_filament()
        self.all_monomers = self.all_monomers[:-1]
        return monomer_to_unbind

    def add_capping_protein(self, cp_to_add):
        """Adds capping protein to filament
        Arguments:
            cp_to_add {CappingProtein} -- capping protein to add
        """
        assert self.cp is None

        new_pos = self.p_end + self.unit_vector * cp_to_add.diameter

        new_pos[0] -= (new_pos[0] > self.box_length) * self.box_length
        new_pos[0] += (new_pos[0] < 0) * self.box_length

        self.p_end = new_pos
        cp_position = self.p_end.copy()
        self.p_height = self.p_end[1]
        cp_to_add.bind_filament(self.filament_id, cp_position)

        self.cp = cp_to_add

    def update_binding_behavior(self, max_height, min_height=0, monomer_diameter=0.1,
                                cp_diameter=0.1):
        """Updates binding behavior of filament for monomers and capping proteins
        Arguments:
            max_height {float} -- maximum height of network
            min_height {float} -- minimum height of network (default: {0})
            monomer_diameter {float} -- diameter of monomer (default: {0.1})
            cp_diameter {float} -- diameter of capping protein (default: {0.1})
        """

        if (len(self.all_monomers) > 0
                and self.cp is None
                and self.all_monomers[-1].arp23 is None):
            self.can_unbind_monomer = True
        else:
            self.can_unbind_monomer = False
        self.can_bind_monomer = ((min_height < (self.p_height + self.unit_vector[1] * monomer_diameter) < max_height)
                                 and (self.cp is None))

        self.can_bind_cp = ((min_height < (self.p_height + self.unit_vector[1] * cp_diameter) < max_height)
                            and (len(self.all_monomers) > 0)
                            and self.cp is None)

    def add_connected_filament(self, crosslinked_filament_id):
        """Adds crosslinked filament to list of connected filaments
        Arguments:
            crosslinked_filament_id {int} -- filament id of crosslinked filament
        """
        assert crosslinked_filament_id not in self.all_connected_filaments
        self.all_connected_filaments.append(crosslinked_filament_id)


class ActinNetwork:
    def __init__(self, num_nucleation_sites=100,
                 monomer_diameter=0.1, cp_diameter=0.1,
                 box_length=4,
                 num_arp23=100, arp23_binding_distance=0.5,
                 num_cps=5000, num_npfs=25,
                 num_monomers=5000,
                 tirf_distance_height=0.1):
        """
        Arguments:
            num_nucleation_sites {int} -- number of nucleation sites (default: {100})
            monomer_diameter {float} -- diameter of monomer (default: {0.1})
            cp_diameter {float} -- diameter of capping protein (default: {0.1})
            box_length {float} -- box length (default: {4})
            num_arp23 {int} -- number of arp23 (default: {100})
            arp23_binding_distance {float} -- distance from interface to bind arp23 (default: {0.5})
            num_cps {int} -- number of capping proteins (default: {5000})
            num_npfs {int} -- number of nucleation promoting factors (default: {25})
            num_monomers {int} -- number of monomers (default: {5000})
            tirf_distance_height {float} -- distance from interface to measure "TIRF density"  (default: {0.1})
        """
        self.box_length = box_length
        self.monomer_diameter = monomer_diameter
        self.cp_diameter = cp_diameter
        self.all_monomers = [ActinMonomer(id, monomer_diameter)
                             for id in range(num_monomers)]

        all_init_angles = ((np.random.rand(num_nucleation_sites) - 0.5)
                           * np.pi/2)
        all_init_positions = np.stack((np.random.rand(num_nucleation_sites)*self.box_length,
                                       np.zeros(num_nucleation_sites)), axis=-1)
        self.all_filaments = [ActinFilament(m_end=init_position,
                                            filament_id=index, init_angle=init_angle,
                                            box_length=box_length,
                                            is_surface=True)
                              for (index, (init_angle, init_position))
                              in enumerate(zip(all_init_angles, all_init_positions))]
        self.arp23_binding_distance = arp23_binding_distance
        self.tirf_distance_height = tirf_distance_height

        self.all_arp23 = [ARP23(i) for i in range(num_arp23)]
        self.all_cps = [CappingProtein(i, cp_diameter)
                        for i in range(num_cps)]
        self.num_npfs = num_npfs

        self.all_free_monomers = self.all_monomers.copy()
        self.all_free_arp23 = self.all_arp23.copy()
        self.all_free_cps = self.all_cps.copy()

        self.all_bound_monomers = []
        self.all_bound_cps = []
        self.all_bound_arp23 = []

        self.all_monomers_near_interface = []
        self.all_arp23able_monomers = []

        # Copy not a deepcopy
        self.all_uncapped_filaments = self.all_filaments.copy()
        self.bindable_filaments = self.all_filaments.copy()  # monomer_bindable filaments
        self.unbindable_filaments = []
        self.cappable_filaments = []
        self.all_high_filaments = self.all_filaments.copy()

    def add_branched_filament(self, branching_monomer):
        """Adds branched filament to network
        Arguments:
            branching_monomer {ActinMonomer} -- monomer to branch from
        Returns:
            ActinFilament -- branched filament
        """
        new_filament_id = len(self.all_filaments)
        angle = self.all_filaments[branching_monomer.filament_id].angle
        clockwise_rotation = (np.random.random() < 0.5)
        angle_to_rotate = np.random.normal(loc=70*0.0174533,
                                           scale=10*0.0174533)
        if clockwise_rotation:
            """Clockwise and counter clockwise might be switched 
            but doesnt matter if 50/50
            """
            init_angle = angle + angle_to_rotate
        else:
            init_angle = angle - angle_to_rotate

        m_end = np.array([branching_monomer.position[0],
                         branching_monomer.position[1]])
        added_filament = ActinFilament(m_end=m_end,
                                       filament_id=new_filament_id,
                                       box_length=self.box_length,
                                       is_surface=False,
                                       init_angle=init_angle)
        self.all_filaments.append(added_filament)
        self.all_high_filaments.append(added_filament)
        self.all_uncapped_filaments.append(added_filament)

        self.cappable_filaments.append(added_filament)
        self.bindable_filaments.append(added_filament)
        self.unbindable_filaments.append(added_filament)
        return added_filament

    def get_upper_monomer_density(self, interface_height):
        """Gets growth front density ("upper monomer density")
        Arguments:
            interface_height {float} -- height of interface
        Returns:
            float -- upper monomer density
        """
        num_upper_monomers = [monomer for monomer in self.all_bound_monomers
                              if (monomer.position[1] > (interface_height - self.arp23_binding_distance))]
        upper_monomer_density = (len(num_upper_monomers)
                                 /(self.arp23_binding_distance * self.box_length))
        upper_monomer_density = upper_monomer_density/1000
        return upper_monomer_density

    def update_binding_behavior(self, max_height, min_height=0, monomer_diameter=0.1,
                                cp_diameter=0.1):
        """Updates binding behavior of each filament in network for monomers and capping proteins
        Arguments:
            max_height {float} -- maximum height of network
            min_height {float} -- minimum height of network (default: {0})
            monomer_diameter {float} -- diameter of monomer (default: {0.1})
            cp_diameter {float} -- diameter of capping protein (default: {0.1})
        """
        [filament.update_binding_behavior(max_height, min_height, monomer_diameter, cp_diameter)
         for filament in self.all_uncapped_filaments]
        self.bindable_filaments = [
            filament for filament in self.all_uncapped_filaments if filament.can_bind_monomer]
        self.unbindable_filaments = [
            filament for filament in self.all_uncapped_filaments if filament.can_unbind_monomer]
        self.cappable_filaments = [
            filament for filament in self.all_uncapped_filaments if filament.can_bind_cp]

    def get_tirf_measurements(self, interface_height):
        """Gets number of monomers, arp23, and capping proteins bound near interface (within TIRF distance)
        Arguments:
            interface_height {float} -- height of interface
        Returns:
            tuple -- number of monomers, arp23, and capping proteins bound near interface
        """
        num_bound_monomers = len([monomer for monomer in self.all_bound_monomers
                                  if (monomer.position[1] > (interface_height - self.tirf_distance_height))])
        num_bound_arp23 = len([arp23 for arp23 in self.all_bound_arp23
                               if (arp23.bound_monomer.position[1] > (interface_height - self.tirf_distance_height))])
        num_bound_cps = len([cp for cp in self.all_bound_cps
                             if (cp.position[1] > (interface_height - self.tirf_distance_height))])
        return num_bound_monomers, num_bound_arp23, num_bound_cps

    def prune_all_monomers_near_interface(self, interface_height):
        """Clear up list of monomers near interface
        """
        self.all_monomers_near_interface = [monomer for monomer in self.all_monomers_near_interface
                                            if (monomer.filament_id is not None
                                                and ((monomer.position[1] > (interface_height - 2 * self.arp23_binding_distance))
                                                     or self.all_filaments[monomer.filament_id].cp is None))]

    def prune_all_high_filaments(self, interface_height):
        """Clear up list of tallest filaments
        """
        self.all_high_filaments = [filament for filament in self.all_high_filaments
                                   if (filament.p_height > (interface_height - 2 * self.arp23_binding_distance) or filament.cp is None)]

    def update_arp23able_monomers(self, interface_height):
        """Updates list of monomers that can bind arp23
        """
        self.all_arp23able_monomers = [monomer for monomer in self.all_monomers_near_interface
                                       if (monomer.filament_id is not None
                                           and (monomer.position[1] > (interface_height - self.arp23_binding_distance))
                                           and monomer.arp23 is None)]

    def get_num_free_barbed_ends(self, interface_height):
        """Gets number of free barbed ends
        Arguments:
            interface_height {float} -- height of interface
        Returns:
            int -- number of free barbed ends
        """
        num_free_barbed_ends = sum(1 for filament in self.all_uncapped_filaments
                                   if (len(filament.all_monomers) > 0
                                       and (filament.cp is None)
                                       and (filament.p_end[1] > (interface_height - self.arp23_binding_distance))))
        return num_free_barbed_ends

    def get_num_bound_monomers(self):
        """Gets number of bound monomers
        """
        return len(self.all_bound_monomers)

    def get_traj_information(self):
        """Gets trajectory information
        Returns:
            tuple -- number of bound arp23, number of bound capping proteins, number of bound monomers, number of monomers by filament, filament polarities
        """

        num_bound_arp23 = len(self.all_bound_arp23)
        num_bound_capping = len(self.all_bound_cps)
        num_bound_monomers = len(self.all_bound_monomers)
        all_unit_vectors = np.stack([filament.unit_vector
                                     for filament in self.all_filaments])
        num_monomers_by_filament = [len(filament.all_monomers)
                                    for filament in self.all_filaments]
        all_filament_polarities = np.arctan2(all_unit_vectors[:, 0],
                                             all_unit_vectors[:, 1])

        return num_bound_arp23, num_bound_capping, num_bound_monomers, num_monomers_by_filament, all_filament_polarities

    def get_detailed_traj_information(self):
        """Gets detailed trajectory information
        Returns:
            tuple -- all monomer positions, all bound arp23 positions, all bound capping protein positions
        """
        bound_arp23_pos = [[arp23.bound_monomer.filament_id,
                            arp23.bound_monomer.position]
                           for arp23 in self.all_bound_arp23]

        bound_capping_pos = [cp.position for cp in self.all_bound_cps]

        all_monomer_positions = []
        for (index, f) in enumerate(self.all_filaments):
            if len(f.all_monomers) == 0:
                continue
            monomer_positions = np.array([monomer.position
                                          for monomer in f.all_monomers])
            m_end_f = f.m_end
            monomer_positions = np.append(np.expand_dims(m_end_f, 0),
                                          monomer_positions, axis=0)
            all_monomer_positions.append(monomer_positions)
        return all_monomer_positions, bound_arp23_pos, bound_capping_pos

    def get_graph_information(self):
        """Gets node and edge information
        Returns:
            tuple -- all node positions, all edges, all distances, all unit vectors
        """
        all_node_pos = []
        all_edges = []
        all_distances = []
        all_unit_vectors = []
        all_arp23_ids = []
        for filament in self.all_filaments:
            if (len(filament.all_monomers) == 0):
                continue
            if (filament.is_surface):
                """Filament starting from surface
                """
                all_node_pos.append(filament.m_end)
                start_node = len(all_node_pos) - 1
            else:
                """Daughter filament from ARP23
                """
                start_node = np.where(filament.m_end
                                      == np.stack(all_node_pos))[0][0]
            total_distance = 0
            for (index, monomer) in enumerate(filament.all_monomers):
                total_distance += monomer.diameter
                if (monomer.arp23 is not None
                        and len(self.all_filaments[monomer.arp23.branch_filament_id].all_monomers) > 0):
                    """Daughter filament from ARP23
                    """
                    a_id = monomer.arp23.arp23_id
                    a_pos = monomer.arp23.get_position()
                    if a_id not in all_arp23_ids:
                        all_arp23_ids.append(a_id)
                        all_node_pos.append(a_pos)
                        end_node = len(all_node_pos) - 1
                    else:
                        raise ValueError("ARP23 ID already exists")
                        end_node = np.where(a_pos
                                            == np.stack(all_node_pos))[0][0]
                    all_edges.append([start_node, end_node])
                    all_unit_vectors.append(filament.unit_vector)
                    start_node = end_node
                    all_distances.append(total_distance)
                    total_distance = 0
            if (filament.all_monomers[-1].arp23 is None):
                """Accounts for capping protein (via p_end)
                """
                all_node_pos.append(filament.p_end)
                end_node = len(all_node_pos) - 1
                all_edges.append([start_node, end_node])
                all_unit_vectors.append(filament.unit_vector)
                all_distances.append(total_distance)
                total_distance = 0
        all_node_pos = np.stack(all_node_pos)
        all_edges = np.array(all_edges)
        all_distances = np.array(all_distances)
        all_unit_vectors = np.stack(all_unit_vectors)
        return all_node_pos, all_edges, all_distances, all_unit_vectors

    def get_graph_information_w_crosslinks(self):
        """Gets node and edge information with crosslinks
        Returns:
            tuple -- all node positions, all edges, all distances, all unit vectors
        """
        all_positions, all_edges, all_distances, all_unit_vectors = self.get_graph_information()

        all_lines, all_line_ids = get_all_lines(all_distances, all_edges,
                                                all_unit_vectors, all_positions, self.box_length)

        all_intersections_to_add, all_line_ids_to_change = get_all_intersections(all_lines,
                                                                                 all_line_ids)

        all_new_edges = []
        all_new_unit_vectors = []
        for line_id in np.unique(all_line_ids):
            if line_id in all_line_ids_to_change:
                intersections_to_add = all_intersections_to_add[all_line_ids_to_change == line_id]
                line_ids_to_change = all_line_ids_to_change[all_line_ids_to_change == line_id]
                curr_unit_vector = all_unit_vectors[line_id]
                if intersections_to_add[:, 1].var() == 0:
                    """Fully horizontal line
                    """
                    resort_indices = intersections_to_add[:, 0].argsort()
                    if curr_unit_vector[0] < 0:
                        """Line points left
                        """
                        resort_indices = np.flip(resort_indices)
                else:
                    resort_indices = intersections_to_add[:, 1].argsort()
                    if curr_unit_vector[1] < 0:
                        """y-component of line points down"""
                        resort_indices = np.flip(resort_indices)
                intersections_to_add = intersections_to_add[resort_indices]
                line_ids_to_change = line_ids_to_change[resort_indices]
                curr_edge_to_split = all_edges[line_id]
                new_edges = []
                new_positions = []
                new_unit_vectors = []
                curr_edge_to_split_start_node = curr_edge_to_split[0]
                curr_edge_to_split_end_node = curr_edge_to_split[1]
                for intersection_point in intersections_to_add:
                    intersection_point_index = np.where(np.linalg.norm(
                        all_positions - intersection_point, axis=-1) < 1E-8)[0]
                    if len(intersection_point_index) == 0:
                        intersection_point_index = None
                    if intersection_point_index is not None and curr_edge_to_split_start_node in intersection_point_index:
                        """Intersection point is already a node on this edge (happens if intersection 
                        happens on first node)
                        """
                        continue
                    if intersection_point_index is not None:
                        """Intermediate node should only be present once
                        """
                        try:
                            assert len(intersection_point_index) == 1
                        except AssertionError:
                            # generate random number to save file
                            random_number = random.randint(0, 100000000)
                            all_positions, all_edges, all_distances, all_unit_vectors = self.get_graph_information()
                            np.save(str(random_number) +
                                    "_all_positions.npy", all_positions)
                            np.save(str(random_number) +
                                    "_all_edges.npy", all_edges)
                            np.save(str(random_number) +
                                    "_all_distances.npy", all_distances)
                            np.save(str(random_number) +
                                    "_all_unit_vectors.npy", all_unit_vectors)
                            np.save(
                                str(random_number) + "_all_intersections_to_add.npy", all_intersections_to_add)
                            raise
                        intersection_point_index = intersection_point_index[0]
                    if intersection_point_index is not None:
                        new_end_node = intersection_point_index
                    else:
                        new_end_node = len(all_positions) + len(new_positions)
                        new_positions.append(intersection_point)
                    new_edge = [curr_edge_to_split_start_node, new_end_node]
                    new_edges.append(new_edge)
                    new_unit_vectors.append(curr_unit_vector)
                    curr_edge_to_split_start_node = new_end_node
                if curr_edge_to_split_start_node != curr_edge_to_split_end_node:
                    new_edges.append([curr_edge_to_split_start_node,
                                      curr_edge_to_split_end_node])
                    new_unit_vectors.append(curr_unit_vector)
                if len(new_positions) > 0:
                    all_positions = np.concatenate(
                        [all_positions, new_positions])
                all_new_edges.append(new_edges)
                all_new_unit_vectors.append(new_unit_vectors)
            else:
                all_new_edges.append([all_edges[line_id].tolist()])
                all_new_unit_vectors.append(
                    [all_unit_vectors[line_id].tolist()])
        all_new_edges = np.concatenate(all_new_edges, axis=0)
        all_new_unit_vectors = np.concatenate(all_new_unit_vectors, axis=0)
        all_new_distances = []
        for (unit_vector, edge) in zip(all_new_unit_vectors, all_new_edges):
            if unit_vector[-1] == 0:
                distance = (all_positions[edge[1], 0] -
                            all_positions[edge[0], 0])/unit_vector[0]
            else:
                distance = (all_positions[edge[1], 1] -
                            all_positions[edge[0], 1])/unit_vector[1]
            all_new_distances.append(distance)
        all_new_distances = np.array(all_new_distances)
        return all_positions, all_new_edges, all_new_distances, all_new_unit_vectors

    def compute_stiffnes_matrix(self):
        """Computes stiffness matrix
        Returns:
            tuple -- stiffness matrix, all node positions, all edges, all distances, all unit vectors
        """
        all_node_pos, all_edges, all_distances, all_unit_vectors = self.get_graph_information_w_crosslinks()

        C = np.zeros((all_edges.shape[0],
                      all_node_pos.shape[0], 2))
        indices = np.arange(0, all_edges.shape[0])
        C[indices, all_edges[:, 0]] = all_unit_vectors  # Source
        C[indices, all_edges[:, 1]] = -all_unit_vectors  # Target

        # Q is compatbility matrix
        Q_t = np.concatenate((C[:, :, 0], C[:, :, 1]), axis=-1)  # Q_transpose
        # k is bond stiffness
        k = 1/all_distances
        stiffness_matrix = Q_t.T@np.diag(k)@Q_t
        return stiffness_matrix, all_node_pos, all_edges, all_distances, all_unit_vectors

    def get_youngs_modulus(self,  y_threshold, force_threshold, fixed_threshold):
        """Gets young's modulus
        Arguments:
            y_threshold {float} -- height threshold to apply force
            force_threshold {float} -- force to apply
            fixed_threshold {float} -- height threshold to fix
        Returns:
            tuple -- young's modulus, all node positions, all edges, all distances, all unit vectors, stress, height
        """
        stiffness_matrix, all_node_pos, all_edges, all_distances, all_unit_vectors = self.compute_stiffnes_matrix()
        max_h = np.max(all_node_pos[:, 1])

        indices = np.arange(all_node_pos.shape[0])

        # find the nucleation sites, which will not move under applied force
        # fixed_nodes = np.where(all_node_pos[:, 1] == 0)[0]
        fixed_nodes = np.where((all_node_pos[:, 1] <= fixed_threshold)
                               | ((all_node_pos[:, 1] == 0)))[0]

        # find all the nodes which will change value under applied force
        free_nodes = np.where(all_node_pos[:, 1] != 0)[0]

        # Only the y degrees of freedom change under a vertical load
        # 1. Get bottom diagonal of stiffness matrix
        stiffness_matrix_free_nodes_y = stiffness_matrix[all_node_pos.shape[0]:,
                                                         all_node_pos.shape[0]:]

        # 2. Resort based on free nodes
        reindex_order = np.concatenate((fixed_nodes, free_nodes))
        stiffness_matrix_free_nodes_y = stiffness_matrix_free_nodes_y[reindex_order, :]
        stiffness_matrix_free_nodes_y = stiffness_matrix_free_nodes_y[:, reindex_order]

        # 3. Remove the fixed nodes
        stiffness_matrix_free_nodes_y = stiffness_matrix_free_nodes_y[fixed_nodes.shape[0]:,
                                                                      fixed_nodes.shape[0]:]

        # Define the force array to act on all nodes above a certain height (y_threshold)
        free_nodes_above_threshold = np.where(
            all_node_pos[free_nodes, 1] > (max_h-y_threshold))[0]

        stress = []
        height = []
        all_all_node_pos_j = []
        for j in range(50):
            all_node_pos_free = all_node_pos[free_nodes]
            force_on_free_nodes = np.zeros(free_nodes.shape[0])
            force_on_free_nodes[free_nodes_above_threshold] = - \
                force_threshold*j
            d = np.linalg.solve(
                stiffness_matrix_free_nodes_y, force_on_free_nodes)
            all_node_pos_free[:, 1] = np.clip(
                all_node_pos_free[:, 1] + d, a_min=0, a_max=None)
            height.append(1-np.max(all_node_pos_free[:, 1])/max_h)
            stress.append(j*force_threshold*len(free_nodes_above_threshold))

            all_node_pos_j = np.zeros_like(all_node_pos)
            all_node_pos_j[free_nodes] = all_node_pos_free
            all_node_pos_j[fixed_nodes] = all_node_pos[fixed_nodes]
            all_all_node_pos_j.append(all_node_pos_j)
        youngs_modulus = (stress[1]-stress[0])/(height[1]-height[0])
        return youngs_modulus, all_all_node_pos_j, all_edges, all_distances, all_unit_vectors, stress, height
