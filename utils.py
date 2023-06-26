import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Helper functions for computing structural mechanics of an actin network


def compute_local_stiffness_matrix(i, all_edges, all_distances, all_unit_vectors):
    """Computes the local stiffness matrix for a given edge
    Arguments:
        i {int} -- edge index
        all_edges {np.array} -- array of edges
        all_distances {np.array} -- array of distances
        all_unit_vectors {np.array} -- array of unit vectors
    Returns:
        np.array -- local stiffness matrix
    """

    """Here EI is the bending stiffness and AE is the stretching stiffness"""
    EI = np.ones(len(all_edges))*1e-2
    AE = np.ones(len(all_edges))*1

    k = AE/all_distances

    c = all_unit_vectors[:, 0]
    s = all_unit_vectors[:, 1]
    K = np.array([[k[i], 0, 0, -k[i], 0, 0],
                  [0, EI[i]*12/all_distances[i]**3, EI[i]*6/all_distances[i]**2,
                   0, -EI[i]*12/all_distances[i]**3, EI[i]*6/all_distances[i]**2],
                  [0, EI[i]*6/all_distances[i]**2, EI[i]*4/all_distances[i],
                   0, -EI[i]*6/all_distances[i]**2, EI[i]*2/all_distances[i]],
                  [-k[i], 0, 0, k[i], 0, 0],
                  [0, -EI[i]*12/all_distances[i]**3, -EI[i]*6/all_distances[i]**2,
                   0, EI[i]*12/all_distances[i]**3, -EI[i]*6/all_distances[i]**2],
                  [0, EI[i]*6/all_distances[i]**2, EI[i]*2/all_distances[i], 0, -EI[i]*6/all_distances[i]**2, EI[i]*4/all_distances[i]]])
    T = np.array([[c[i], s[i], 0, 0, 0, 0], [-s[i], c[i], 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, c[i], s[i], 0], [0, 0, 0, -s[i], c[i], 0], [0, 0, 0, 0, 0, 1]])
    l_s_m = np.transpose(T)@K@T
    return l_s_m


def compute_global_stiffness_matrix(all_node_pos, all_edges, all_distances, all_unit_vectors):
    """Computes the global stiffness matrix for the network by summing up the local stiffness matrices for each edge
    Arguments:
        all_node_pos {np.array} -- array of node positions
        all_edges {np.array} -- array of edges
        all_distances {np.array} -- array of distances
        all_unit_vectors {np.array} -- array of unit vectors
    Returns:
        np.array -- stiffness matrix
    """
    vlsm = np.vectorize(compute_local_stiffness_matrix,
                        signature='()->(n,n)', excluded=[1, 2, 3])
    vl = vlsm(np.arange(len(all_edges)), all_edges,
              all_distances, all_unit_vectors)
    g_s_m = np.zeros((3*all_node_pos.shape[0], 3*all_node_pos.shape[0]))
    for i in range(all_edges.shape[0]):
        g_s_m[3*all_edges[i, 0]:3*all_edges[i, 0]+3, 3*all_edges[i, 0]:3*all_edges[i, 0]+3] = g_s_m[3*all_edges[i, 0]:3*all_edges[i, 0]+3,
                                                                                                    3*all_edges[i, 0]:3*all_edges[i, 0]+3] + vl[i, 0:3, 0:3]
        g_s_m[3*all_edges[i, 0]:3*all_edges[i, 0]+3, 3*all_edges[i, 1]:3*all_edges[i, 1]+3] = g_s_m[3*all_edges[i, 0]:3*all_edges[i, 0]+3,
                                                                                                    3*all_edges[i, 1]:3*all_edges[i, 1]+3] + vl[i, 0:3, 3:6]
        g_s_m[3*all_edges[i, 1]:3*all_edges[i, 1]+3, 3*all_edges[i, 0]:3*all_edges[i, 0]+3] = g_s_m[3*all_edges[i, 1]:3*all_edges[i, 1]+3,
                                                                                                    3*all_edges[i, 0]:3*all_edges[i, 0]+3] + vl[i, 3:6, 0:3]
        g_s_m[3*all_edges[i, 1]:3*all_edges[i, 1]+3, 3*all_edges[i, 1]:3*all_edges[i, 1]+3] = g_s_m[3*all_edges[i, 1]:3*all_edges[i, 1]+3,
                                                                                                    3*all_edges[i, 1]:3*all_edges[i, 1]+3] + vl[i, 3:6, 3:6]
    return g_s_m


def mass_matrix(i, all_distances, all_unit_vectors):
    """Computes the mass matrix for a given edge
    Arguments:
        i {int} -- edge index
        all_edges {np.array} -- array of edges
        all_distances {np.array} -- array of distances
        all_unit_vectors {np.array} -- array of unit vectors
    Returns:
        np.array -- local stiffness matrix
    """
    #Here rho is the mass per unit length of the beam
    rho = 1
    c = all_unit_vectors[:, 0]
    s = all_unit_vectors[:, 1]
    M = rho*all_distances[i]/420*np.array([[140, 0, 0, 70, 0, 0],
                                          [0, 156, 22*all_distances[i], 0,
                                              54, -13*all_distances[i]],
                                          [0, 22*all_distances[i], 4*all_distances[i]**2,
                                              0, 13*all_distances[i], -3*all_distances[i]**2],
                                          [70, 0, 0, 140, 0, 0],
                                          [0, 54, 13*all_distances[i], 0,
                                              156, -22*all_distances[i]],
                                          [0, -13*all_distances[i], -3*all_distances[i]**2, 0, -22*all_distances[i], 4*all_distances[i]**2]])
    T = np.array([[c[i], s[i], 0, 0, 0, 0],
                  [-s[i], c[i], 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, c[i], s[i], 0],
                  [0, 0, 0, -s[i], c[i], 0],
                  [0, 0, 0, 0, 0, 1]])
    mass_m = np.transpose(T)@M@T
    return mass_m


def compute_mass_matrix(all_node_pos, all_edges, all_distances, all_unit_vectors):
    """Computes the global mass matrix for the network by summing up the mass matrices for each edge
    Arguments:
        all_node_pos {np.array} -- array of node positions
        all_edges {np.array} -- array of edges
        all_distances {np.array} -- array of distances
        all_unit_vectors {np.array} -- array of unit vectors
    Returns:
        np.array -- stiffness matrix
    """
    vlsm = np.vectorize(mass_matrix, signature='()->(n,n)', excluded=[1, 2, 3])
    vl = vlsm(np.arange(len(all_edges)), all_distances, all_unit_vectors)
    g_s_m = np.zeros((3*all_node_pos.shape[0], 3*all_node_pos.shape[0]))
    for i in range(all_edges.shape[0]):
        g_s_m[3*all_edges[i, 0]:3*all_edges[i, 0]+3, 3*all_edges[i, 0]:3*all_edges[i, 0]+3] = g_s_m[3 *
                                                                                                    all_edges[i, 0]:3*all_edges[i, 0]+3, 3*all_edges[i, 0]:3*all_edges[i, 0]+3] + vl[i, 0:3, 0:3]
        g_s_m[3*all_edges[i, 0]:3*all_edges[i, 0]+3, 3*all_edges[i, 1]:3*all_edges[i, 1]+3] = g_s_m[3 *
                                                                                                    all_edges[i, 0]:3*all_edges[i, 0]+3, 3*all_edges[i, 1]:3*all_edges[i, 1]+3] + vl[i, 0:3, 3:6]
        g_s_m[3*all_edges[i, 1]:3*all_edges[i, 1]+3, 3*all_edges[i, 0]:3*all_edges[i, 0]+3] = g_s_m[3 *
                                                                                                    all_edges[i, 1]:3*all_edges[i, 1]+3, 3*all_edges[i, 0]:3*all_edges[i, 0]+3] + vl[i, 3:6, 0:3]
        g_s_m[3*all_edges[i, 1]:3*all_edges[i, 1]+3, 3*all_edges[i, 1]:3*all_edges[i, 1]+3] = g_s_m[3 *
                                                                                                    all_edges[i, 1]:3*all_edges[i, 1]+3, 3*all_edges[i, 1]:3*all_edges[i, 1]+3] + vl[i, 3:6, 3:6]
    return g_s_m


def get_response_function_w(mass_matrix, gsf, w):
    """Computes the response function for a given frequency
    Arguments:
        mass_matrix {torch.tensor} -- mass matrix
        gsf {torch.tensor} -- global stiffness matrix
        w {float} -- frequency
    """
    # I = torch.eye(gsf.shape[0], device="cuda:0").float()
    G_inv = (-(w**2) * mass_matrix + gsf)
    G = torch.linalg.inv(G_inv)
    G_t = G.conj().T
    response_function_w = torch.trace(G_t @ G).item()
    return response_function_w


def get_response_function(all_w, all_node_pos, all_edges, all_distances, all_unit_vectors):
    """Computes the response function for a given network
    Arguments:
        all_w {np.array} -- array of frequencies
        all_node_pos {np.array} -- array of node positions
        all_edges {np.array} -- array of edges
        all_distances {np.array} -- array of distances
        all_unit_vectors {np.array} -- array of unit vectors
    Returns:
        [np.array] -- array of response function values
    """
    gsf = compute_global_stiffness_matrix(all_node_pos, all_edges,
                                          all_distances, all_unit_vectors)

    mass_matrix = compute_mass_matrix(
        all_node_pos, all_edges, all_distances, all_unit_vectors)
    """Apply boundary conditions"""
    fixed_nodes = np.where(all_node_pos[:, 1] <= 0.5)[0]
    fixed_node_indices = np.concatenate([3*fixed_nodes,
                                         3*fixed_nodes+1, 3*fixed_nodes+2], axis=0)

    """delete rows and columns in global stiffness matrix corresponding to nodes with y=0"""
    gsf = np.delete(gsf, fixed_node_indices, axis=0)
    gsf = np.delete(gsf, fixed_node_indices, axis=1)
    mass_matrix = np.delete(mass_matrix, fixed_node_indices, axis=0)
    mass_matrix = np.delete(mass_matrix, fixed_node_indices, axis=1)

    gsf = torch.tensor(gsf, device=device).float()
    mass_matrix = torch.tensor(mass_matrix, device=device).float()
    response_function = [get_response_function_w(
        mass_matrix, gsf, w) for w in all_w]
    return response_function


def get_response_function_w_diag(mass_matrix, gsf, w):
    """Computes the response function of each node for a given frequency (diagonal of the response function matrix)
    Arguments:
        mass_matrix {torch.tensor} -- mass matrix
        gsf {torch.tensor} -- global stiffness matrix
        w {float} -- frequency
    Returns:
        np.array -- array of response function values
    """
    # I = torch.eye(gsf.shape[0], device="cuda:0").float()
    G_inv = (-(w**2) * mass_matrix + gsf)
    G = torch.linalg.inv(G_inv)
    G_t = G.conj().T
    response_function_diag = torch.diag(G_t @ G)
    response_function_diag = response_function_diag.reshape(
        (-1, 3)).sum(-1).cpu().numpy()
    return response_function_diag


def get_response_function_diag(all_w, all_node_pos, all_edges, all_distances, all_unit_vectors):
    """Computes the response function of each node for a given network (diagonal of the response function matrix)
    Arguments:
        all_w {np.array} -- array of frequencies
        all_node_pos {np.array} -- array of node positions
        all_edges {np.array} -- array of edges
        all_distances {np.array} -- array of distances
        all_unit_vectors {np.array} -- array of unit vectors
    Returns:
        [np.array] -- array of response function values
    """
    gsf = compute_global_stiffness_matrix(all_node_pos, all_edges,
                                          all_distances, all_unit_vectors)

    mass_matrix = compute_mass_matrix(
        all_node_pos, all_edges, all_distances, all_unit_vectors)
    """Apply boundary conditions"""
    fixed_nodes = np.where(all_node_pos[:, 1] <= 0.5)[0]
    fixed_node_indices = np.concatenate([3*fixed_nodes,
                                         3*fixed_nodes+1, 3*fixed_nodes+2], axis=0)

    """delete rows and columns in global stiffness matrix corresponding to nodes with y=0"""
    gsf = np.delete(gsf, fixed_node_indices, axis=0)
    gsf = np.delete(gsf, fixed_node_indices, axis=1)
    mass_matrix = np.delete(mass_matrix, fixed_node_indices, axis=0)
    mass_matrix = np.delete(mass_matrix, fixed_node_indices, axis=1)

    gsf = torch.tensor(gsf, device=device).float()
    mass_matrix = torch.tensor(mass_matrix, device=device).float()
    response_function = [get_response_function_w_diag(
        mass_matrix, gsf, w) for w in all_w]
    return response_function


def get_eigenvalues(all_node_pos, all_edges, all_distances, all_unit_vectors):
    """Gets the eigenvalues of the network
    Arguments:
        all_node_pos {np.array} -- array of node positions
        all_edges {np.array} -- array of edges
        all_distances {np.array} -- array of distances
        all_unit_vectors {np.array} -- array of unit vectors
    Returns:
        [np.array] -- array of eigenvalues
    """
    gsf = compute_global_stiffness_matrix(all_node_pos, all_edges,
                                          all_distances, all_unit_vectors)

    mass_matrix = compute_mass_matrix(
        all_node_pos, all_edges, all_distances, all_unit_vectors)
    """Apply boundary conditions"""
    fixed_nodes = np.where(all_node_pos[:, 1] <= 0.5)[0]
    fixed_node_indices = np.concatenate([3*fixed_nodes,
                                         3*fixed_nodes+1, 3*fixed_nodes+2], axis=0)

    """delete rows and columns in global stiffness matrix corresponding to nodes with y=0"""
    gsf = np.delete(gsf, fixed_node_indices, axis=0)
    gsf = np.delete(gsf, fixed_node_indices, axis=1)
    mass_matrix = np.delete(mass_matrix, fixed_node_indices, axis=0)
    mass_matrix = np.delete(mass_matrix, fixed_node_indices, axis=1)

    gsf = torch.tensor(gsf, device=device).float()
    mass_matrix = torch.tensor(mass_matrix, device=device).float()
    m_gsf = torch.linalg.inv(mass_matrix) @ gsf
    eigenvalues, eigenvectors = torch.linalg.eig(m_gsf)
    return eigenvalues.cpu().numpy(), eigenvectors.cpu().numpy()


def compute_compression_step_stress_strain(all_node_pos, all_edges, all_distances, all_unit_vectors, force_threshold=0.1):
    """Computes the stress and strain curve for a given network by applying a force to the top of the network and then computing the resulting stress and strain
    Arguments:
        all_node_pos {np.array} -- array of node positions
        all_edges {np.array} -- array of edges
        all_distances {np.array} -- array of distances
        all_unit_vectors {np.array} -- array of unit vectors
        force_threshold {float} -- force threshold for the top of the network (default: {0.1})
    Returns:
        [np.array] -- array of stress values
        [np.array] -- array of strain values
    """
    all_all_node_pos = [all_node_pos]
    all_all_edges = [all_edges]
    all_all_distances = [all_distances]
    all_all_unit_vectors = [all_unit_vectors]
    stress = []
    strain = []
    max_h = np.max(all_node_pos[:, 1])
    new_max_height = max_h
    strain_i = 0
    i = 0
    while strain_i < 0.6:
        all_node_pos = all_all_node_pos[-1]
        all_edges = all_all_edges[-1]
        all_distances = all_all_distances[0]
        all_unit_vectors = all_all_unit_vectors[-1]

        global_stiffness_matrix = compute_global_stiffness_matrix(all_node_pos, all_edges, all_distances, all_unit_vectors)

        """Apply boundary conditions"""
        fixed_nodes = np.where(all_node_pos[:, 1] <= 0.5)[0]
        free_nodes = np.where(all_node_pos[:, 1] > 0.5)[0]

        fixed_node_indices = np.concatenate(
            [3*fixed_nodes, 3*fixed_nodes+1, 3*fixed_nodes+2], axis=0)

        """delete rows and columns in global stiffness matrix corresponding to nodes with y=0"""
        global_stiffness_matrix = np.delete(
            global_stiffness_matrix, fixed_node_indices, axis=0)
        global_stiffness_matrix = np.delete(
            global_stiffness_matrix, fixed_node_indices, axis=1)

        """Apply force to free nodes above y_threshold (default = 0.2), in the negative y direction"""
        force_vector = np.zeros((3*free_nodes.shape[0]))
        free_nodes_above_threshold = np.where(
            all_node_pos[free_nodes, 1] > np.max(all_node_pos[:, 1]-0.2))[0]

        force_vector[3*free_nodes_above_threshold+1] = -force_threshold*i

        box_length = 2.7

        """Solve for displacement vector"""
        d = np.linalg.inv(global_stiffness_matrix)@force_vector

        """Compute new node positions"""
        deformed_node_pos = np.zeros((all_node_pos.shape[0], 2))
        deformed_node_pos[free_nodes,
                          0] = all_node_pos[free_nodes, 0] + d[0::3]

        """Apply periodic boundary conditions in the x direction"""
        deformed_node_pos[free_nodes,
                          0] -= (deformed_node_pos[free_nodes, 0] >= box_length) * box_length
        deformed_node_pos[free_nodes,
                          0] += (deformed_node_pos[free_nodes, 0] < 0) * box_length

        """Apply boundary conditions in the y direction"""
        deformed_node_pos[free_nodes, 1] = np.clip(
            all_node_pos[free_nodes, 1] + d[1::3], a_min=0.0, a_max=None)
        deformed_node_pos[fixed_nodes, 0] = all_node_pos[fixed_nodes, 0]
        deformed_node_pos[fixed_nodes, 1] = all_node_pos[fixed_nodes, 1]

        """Compute new edge lengths and unit vectors"""
        deformed_displacements = (deformed_node_pos[all_edges[:, 1]]
                                  - deformed_node_pos[all_edges[:, 0]])
        #Add PBC in the x-direction for deformed_Displacements
        deformed_displacements[:, 0] = (deformed_displacements[:, 0] 
                                        - box_length * np.round(deformed_displacements[:, 0] / box_length))
        
        deformed_distances = np.linalg.norm(deformed_displacements, axis=1)
        deformed_unit_vectors = (deformed_displacements
                                 / deformed_distances[:, np.newaxis])

        # deformed_distances = np.linalg.norm(deformed_node_pos[all_edges[:,0]]-deformed_node_pos[all_edges[:,1]],axis=1)
        # deformed_unit_vectors = (deformed_node_pos[all_edges[:,1]]-deformed_node_pos[all_edges[:,0]])/deformed_distances[:,np.newaxis]

        """Compute stress and strain"""
        stress_i = force_threshold*len(free_nodes_above_threshold)*i
        new_max_height = np.max(deformed_node_pos[:, 1])
        strain_i = -(new_max_height-max_h)/(max_h)
        stress.append(stress_i)
        strain.append(strain_i)
        all_all_node_pos.append(deformed_node_pos)
        all_all_unit_vectors.append(deformed_unit_vectors)
        all_all_edges.append(all_edges)
        i += 1

    return stress, strain, all_all_node_pos, all_all_edges, all_all_distances, all_all_unit_vectors
