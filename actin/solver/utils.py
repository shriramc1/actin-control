import numpy as np
def get_intersection(p1, p2, p3, p4):
    """
    Return the intersection point of two lines.
    Each line is defined by two points.
    If the lines are parallel, return None.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # Compute the determinant
    det = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if det == 0:
        """Parallel
        """
        return None

    # Compute the intersection point
    x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4))/det
    y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4))/det
    if (x >= min(x1, x2) and x <= max(x1, x2) and
        x >= min(x3, x4) and x <= max(x3, x4) and
        y >= min(y1, y2) and y <= max(y1, y2) and
            y >= min(y3, y4) and y <= max(y3, y4)):
        return (x, y)
    else:
        return None


def get_all_lines(all_distances, all_edges, all_unit_vectors, all_positions, box_length):
    all_lines = []
    all_line_ids = []
    for (edge_num, (distance, edge, unit_vector)) in enumerate(zip(all_distances,
                                                                   all_edges,
                                                                   all_unit_vectors)):
        point_0, point_1 = all_positions[edge]
        if unit_vector[0] > 0:
            x_distance = point_0[0] + distance * unit_vector[0]
        elif unit_vector[0] < 0:
            x_distance = box_length - point_0[0] - distance * unit_vector[0]
        num_wraps = int(x_distance/box_length)
        for segment_num in range(num_wraps):
            if unit_vector[0] > 0:
                x_0_end = box_length
                y_0_end = ((unit_vector[1]/unit_vector[0])
                           * (x_0_end - point_0[0]) + point_0[1])

            elif unit_vector[0] < 0:
                x_0_end = 0
                y_0_end = ((unit_vector[1]/unit_vector[0])
                           * (x_0_end - point_0[0]) + point_0[1])
            else:
                raise ValueError("Shouldn't be here")
            line_1 = [point_0, [x_0_end, y_0_end]]
            all_lines.append(line_1)
            all_line_ids.append(edge_num)
            point_0 = [np.abs(x_0_end - box_length), y_0_end]

        all_lines.append([point_0, point_1])
        all_line_ids.append(edge_num)
    return all_lines, all_line_ids


def get_all_intersections(all_lines, all_line_ids):
    all_new_intersections = []
    for (i, (line_0, line_id)) in enumerate(zip(all_lines, all_line_ids)):
        for (line_1, line_id_1) in zip(all_lines[i+1:], all_line_ids[i+1:]):
            intersection = get_intersection(line_0[0], line_0[1],
                                            line_1[0], line_1[1])
            if intersection is not None:
                all_new_intersections.append(
                    [intersection, line_id, line_id_1])

    all_line_ids_to_change = []
    all_intersections_to_add = []
    for (index, intersection) in enumerate(all_new_intersections):
        intersection_point, line_id, line_id_1 = intersection
        all_line_ids_to_change.append(line_id)
        all_line_ids_to_change.append(line_id_1)
        all_intersections_to_add.append(list(intersection_point))
        all_intersections_to_add.append(list(intersection_point))

    all_intersections_to_add = np.array(all_intersections_to_add)
    all_line_ids_to_change = np.array(all_line_ids_to_change)

    return all_intersections_to_add, all_line_ids_to_change
