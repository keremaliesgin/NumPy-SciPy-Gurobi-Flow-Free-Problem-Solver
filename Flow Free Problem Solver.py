#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# load libraries
import numpy as np
import scipy.sparse as sp

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import gurobipy as gp
from gurobipy import GRB

GLOBAL_ENV = gp.Env(empty=True)
GLOBAL_ENV.setParam("OutputFlag", 0)
GLOBAL_ENV.start()


# In[ ]:


def mixed_integer_linear_programming(direction, A, senses, b, c, l, u, types, names, solver_params=None):
    """
    Optimized wrapper. Uses a global environment and allows parameter tuning.
    """
    # 1. Use the global environment to avoid license/startup overhead
    # We use the existing GLOBAL_ENV defined outside
    model = gp.Model("MILP_Solver", env=GLOBAL_ENV)

    # 2. Apply dynamic parameters (e.g., TimeLimit, MIPGap, Heuristics)
    if solver_params:
        for param, value in solver_params.items():
            model.setParam(param, value)

    # 3. Add Variables (Matrix API)
    x = model.addMVar(
        shape=len(c), 
        lb=l, 
        ub=u, 
        obj=c, 
        vtype=types, 
        name="x"  # Base name
    )

    # 4. Define Objective Sense
    if direction == "maximize":
        model.ModelSense = GRB.MAXIMIZE
    else:
        model.ModelSense = GRB.MINIMIZE

    # 5. Add Constraints
    sense_map = {'L': '<', 'G': '>', 'E': '='}
    gurobi_senses = np.array([sense_map[s] for s in senses])
    
    # Using addMConstr is correct and efficient for sparse matrices
    model.addMConstr(A, x, gurobi_senses, b, name="constr")

    # 6. Solve
    model.optimize()

    # 7. Robust Return
    # Check specifically for solution count to avoid crashes on 'model.X' access
    if model.SolCount > 0:
        return model.X, model.ObjVal
    
    # If Infeasible, Unbounded, or Error
    # print(f"No solution found. Status Code: {model.Status}")
    return np.zeros(len(c)), 0


# In[ ]:


def flow_free_solver_with_graph(board):

    # it turns out that introducing flow variables make the problem WAY, WAY HARDER to solve

    # -----------------------------------------------------------------------------------------------------------
    # decision variable 1: x_k_h_w: 1 if color k is in row h and col w, 0 otherwise
    # decision variable 2: we use the total indices instead of coordinates of a cell: 
    #                      for any cell k_h_w, the index is k * H * W + h * W + w
    #                      f_u_v: 1 if edge between cell u to cell v is used, 0 otherwise
    #                      do note that u and v are in the forms of get_index method defined below
    #                      we also have to make sure that if for some k, u and v, f_u_v = 1, then f_v_u is also 1
    #                      we can do this later by setting equations
    # -----------------------------------------------------------------------------------------------------------

    K = np.max(board)
    H, W = board.shape

    # helper methods
    def get_index(k, h, w):
        return k * H * W + h * W + w

    def is_within_bounds(k, h, w):
        return (0 <= k < K) and (0 <= h < H) and (0 <= w < W)

    # we only care about moving in one direction at the moment for easier calculations,
    # will be further optimized with slicing and vectorization 
    # output format: [u_index, v_index] where u_index is the current cell we stand on and v_index is the neighbor of ours
    neighbors_one_direction = [(0, 0, 1), (0, 1, 0)]
    def get_neighbor_indices_one_direction(k, h, w):
        temp = []
        for dk, dh, dw in neighbors_one_direction:
            if is_within_bounds(k + dk, h + dh, w + dw):
                temp.append([get_index(k, h, w), get_index(k + dk, h + dh, w + dw)])
        return temp

    # usage of helper methods
    flow_indices_first_half = []
    temp = []
    for h in range(H):
        for w in range(W):
            temp.extend(get_neighbor_indices_one_direction(0, h, w))
    for k in range(K):
        flow_indices_first_half.extend(np.array(temp) + k * H * W)

    flow_indices_first_half = np.array(flow_indices_first_half)
    flow_indices = np.concatenate((flow_indices_first_half, np.fliplr(flow_indices_first_half)))

    names_x = np.array(["x_{}_{}_{}".format(k+1, h+1, w+1) for k in range(K) for h in range(H) for w in range(W)])
    names_f = np.array(["f_{}_{}".format(u+1, v+1) for u, v in flow_indices])
    names = np.concatenate((names_x, names_f))

    total_x_vars = names_x.size
    total_f_vars = names_f.size

    types = np.repeat("B", total_x_vars + total_f_vars)
    c = np.repeat(0, total_x_vars + total_f_vars)

    # indices of given cells
    indices_of_terminals = np.array(board.nonzero()).T

    # lower bounds
    # we force the already-existing variables to be 1
    # for the edges, we have no clue about them at the moment, so we do not touch them
    l_x = np.zeros((K, H, W))
    for h, w in indices_of_terminals:
        l_x[board[h][w] - 1][h][w] = 1
    l_x = l_x.flatten()
    l_f = np.zeros((total_f_vars))
    l = np.concatenate((l_x, l_f))

    # upper bounds
    # for cells, we force other layers to be 0, kind of arbitrary
    # for edges, see below
    u_x = np.ones((K, H, W))
    for h, w in indices_of_terminals:
        for k in range(K):
            if board[h][w] != k+1:
                u_x[k][h][w] = 0

    u_x = u_x.flatten()

    # masking the invalid edges (f variables) here
    # logic: if both node u and v are eligible to contain color k, then f_u_v and f_v_u can be 1
    #        if either of them cannot contain color k, then they won't have an edge connecting them, so we force that variable to 0
    u_f = np.ones((total_f_vars))
    step = total_f_vars // 2
    u_nodes, v_nodes = flow_indices_first_half[:, 0], flow_indices_first_half[:, 1]
    valid_edges_mask = (u_x[u_nodes] == 1) & (u_x[v_nodes] == 1)
    indices_to_erase = np.where(~valid_edges_mask)[0]
    u_f[indices_to_erase], u_f[indices_to_erase + step] = 0, 0
    u = np.concatenate((u_x, u_f))

    # ----------------------------------------------
    # here, we transition into building the A matrix
    # ----------------------------------------------

    # constraint 1: each cell may contain only one color
    # sum of x over range(K) for h = 1, ..., H; w = 1, ..., W
    aij_1 = np.repeat(1, K * H * W)
    row_1 = np.repeat(range(H * W), K)
    col_1 = np.arange(K * H * W).reshape(K, -1).T.flatten()
    b_1 = np.repeat(1, H * W)
    senses_1 = np.repeat("E", H * W)

    # constraint 2: each terminal cell has to have only one connection
    # logic: we get the index of each terminal (which gives the u values),
    #        then we find the edges connected to it and add them to constraint

    # important from now on: row_index counter
    current_row_counter = H * W

    row_2 = []
    col_2 = []
    u_indices = []

    for h, w in indices_of_terminals:
        u_indices.append(get_index(board[h][w] - 1, h, w))
    u_indices = np.array(sorted(u_indices))

    # this gives the col indices actually, let's modify it so that we obtain row indices as well
    for u in u_indices:
        temp_indices = np.where(flow_indices[:, 0] == u)[0]
        repeat = temp_indices.size
        row_2.extend(np.repeat(current_row_counter, repeat))
        col_2.extend(temp_indices + total_x_vars)
        current_row_counter += 1

    row_2 = np.array(row_2)
    col_2 = np.array(col_2)
    aij_2 = np.repeat(1, row_2.size)
    b_2 = np.repeat(1, len(u_indices))
    senses_2 = np.repeat("E", len(u_indices))

    # constraint 3: each non-terminal cell with color k has to have exactly two neighbors
    # todo: make sure that the distinction between terminal and non-terminal arrays are crystal clear
    #       so that future heuristics will be working fine
    # logic: for each non-terminal cell x, we have -2x + sum of edges = 0

    # here, we get both the numerical indices
    numerical_indices_of_terminals = np.array(sorted(np.array([get_index(board[h][w] - 1, h, w) for h, w in indices_of_terminals])))
    empty_cells_mask = np.isin(np.arange(total_x_vars), numerical_indices_of_terminals)
    numerical_indices_of_nonterminals = np.arange(total_x_vars)[~empty_cells_mask]
   
    # handling x_values
    row_3_x = np.arange(numerical_indices_of_nonterminals.size) + current_row_counter
    col_3_x = numerical_indices_of_nonterminals
    aij_3_x = np.repeat(-2, numerical_indices_of_nonterminals.size)

    # handling flows
    # logic: we locate the flow variable in flow_indices_first_half, get the row index and add it to col_3_f

    row_3_f = []
    col_3_f = []

    for i, u in enumerate(numerical_indices_of_nonterminals):
        f_indices = np.where(flow_indices[:, 0] == u)[0]
        current_eq_row = current_row_counter + i
        repeat = f_indices.size
        row_3_f.extend(np.repeat(current_eq_row, repeat))
        col_3_f.extend(f_indices + total_x_vars)

    row_3_f = np.array(row_3_f)
    col_3_f = np.array(col_3_f)
    aij_3_f = np.repeat(1, row_3_f.size)
    b_3 = np.repeat(0, numerical_indices_of_nonterminals.size)
    senses_3 = np.repeat("E", numerical_indices_of_nonterminals.size)

    current_row_counter += numerical_indices_of_nonterminals.size

    # constraint 4: fixing f_u_v = f_v_u
    aij_4 = np.tile([1, -1], flow_indices_first_half.shape[0])
    row_4 = np.repeat(range(flow_indices_first_half.shape[0]), 2) + current_row_counter
    col_4 = np.arange(total_f_vars).reshape(2, -1).T.flatten() + total_x_vars
    b_4 = np.repeat(0, flow_indices_first_half.shape[0])
    senses_4 = np.repeat("E", flow_indices_first_half.shape[0])

    current_row_counter += flow_indices_first_half.shape[0]
    
    # -------------------------
    # constructing the A matrix
    # -------------------------

    aij = np.concatenate((aij_1, aij_2, aij_3_x, aij_3_f, aij_4))
    row = np.concatenate((row_1, row_2, row_3_x, row_3_f, row_4))
    col = np.concatenate((col_1, col_2, col_3_x, col_3_f, col_4))
    b = np.concatenate((b_1, b_2, b_3, b_4))
    senses = np.concatenate((senses_1, senses_2, senses_3, senses_4))

    A = sp.csr_matrix((aij, (row, col)), shape = (b.size, total_x_vars + total_f_vars))

    x_star, _ = mixed_integer_linear_programming("minimize", A, senses, b, c, l, u, types, names)
    x_star = np.argmax(np.array(x_star)[:total_x_vars].reshape(K, H, W), axis = 0) + 1
    return x_star

board = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 0, 0],
    [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 0, 6, 7, 0, 8, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
])

print(flow_free_solver_with_graph(board))
    


# In[ ]:


def flow_free_solver_with_heuristics(board):

    # ----------------------------------------------------------------------------
    # decision variables: x_k_h_w: 1 if color k is in row h and col w, 0 otherwise
    # ----------------------------------------------------------------------------

    # --------------
    # initialization
    # --------------

    K = np.max(board)
    H, W = board.shape
    colors = np.arange(K) + 1

    def get_index(k, h, w):
        return k * H * W + h * W + w
    
    def is_within_board_2d(h, w):
        return (0 <= h < H) and (0 <= w < W)

    def is_within_bounds(k, h, w):
        return (0 <= k < K) and (0 <= h < H) and (0 <= w < W)

    all_neighbors_on_board = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    all_neighbors = [(0, -1, 0), (0, 0, -1), (0, 0, 1), (0, 1, 0)]
    def get_neighbor_indices(index):
        current_neighbors = []
        k, h, w = index // (H * W), (index % (H * W)) // W, index % W
        for dk, dh, dw in all_neighbors:
            if is_within_bounds(k + dk, h + dh, w + dw):
                current_neighbors.append(get_index(k + dk, h + dh, w + dw))
        return current_neighbors

    total_vars = K * H * W

    # --------------
    # variable stuff
    # --------------

    names = np.array(["x_{}_{}_{}".format(k+1, h+1, w+1) for k in range(K) for h in range(H) for w in range(W)])
    types = np.repeat("B", total_vars)
    c = np.repeat(0, total_vars)
    l = np.zeros((total_vars))
    for h, w in np.argwhere(board != 0):
        l[get_index(board[h][w] - 1, h, w)] = 1
    u = np.ones((total_vars))

    
    # here, we save the indices of terminals
    all_terminal_indices = np.array(sorted([get_index(board[h][w] - 1, h, w) for h, w in np.argwhere(board != 0)]))

    # ----------------------------------------
    # heuristics to fasten the solving process (work in process)
    # ----------------------------------------

    # ----------------- bunch of helpers -----------------
    def make_slice_2d(dh, dw):
        def get_slice(d):
            if d == 0: return slice(None)
            return slice(None, -d) if d > 0 else slice(-d, None)
        return get_slice(dh), get_slice(dw)
     
    def make_slice_3d(dk, dh, dw):
        def get_slice(d):
            if d == 0: return slice(None)
            return slice(None, -d) if d > 0 else slice(-d, None)
        return get_slice(dk), get_slice(dh), get_slice(dw)

    def coordinate_to_index_on_board(h, w):
        return h * W + w
    def index_to_coordinate_on_board(index):
        return index // W, index % W
    
    def get_all_neighbors_on_board(index):
        all_neighbors_on_board = []
        h, w = index_to_coordinate_on_board(index)
        for dh, dw in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
            if is_within_bounds(0, h + dh, w + dw):
                all_neighbors_on_board.append(coordinate_to_index_on_board(h + dh, w + dw))
        return all_neighbors_on_board
    
    def get_empty_neighbors_on_board(current_board, index):
        current_neighbors_on_board = []
        h, w = index_to_coordinate_on_board(index)
        for dh, dw in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
            if is_within_bounds(0, h + dh, w + dw) and current_board[h + dh][w + dw] == 0:
                current_neighbors_on_board.append(coordinate_to_index_on_board(h + dh, w + dw))
        return current_neighbors_on_board

    def has_neighbor_with_the_same_color(current_board, index):
        h, w = index_to_coordinate_on_board(index)
        for neighbor_index in get_all_neighbors_on_board(index):
            nh, nw = index_to_coordinate_on_board(neighbor_index)
            if current_board[nh][nw] == current_board[h][w]:
                return True
        return False
    
    def has_two_neighbors_with_the_same_color(current_board, index):
        h, w = index_to_coordinate_on_board(index)
        neighbor_list = []
        for neighbor_index in get_all_neighbors_on_board(index):
            nh, nw = index_to_coordinate_on_board(neighbor_index)
            if current_board[nh][nw] == current_board[h][w]:
                neighbor_list.append(current_board[nh][nw])
        return len(neighbor_list) == 2

    # ----------------- end of helper methods -----------------

    # we save the terminals somewhere safe, so that we can access them in the heuristic methods later on
    terminals_map = {}
    for h, w in np.argwhere(board != 0):
        terminals_map[(h, w)] = board[h][w]

    # ----------------- heuristic method 1 -----------------
    # logic: check for every terminal at first, if there exists a forced move, then apply it.
    #        after that, run a check to see if the initial terminals have a connection.
    #        if they have connection, then we mark them as completed and treat the newly
    #        placed indices as non-terminal indices to avoid confusion.

    def apply_forced_moves(current_board):

        is_changed_temp = True
        actually_changed = False

        while is_changed_temp:
            is_changed_temp = False
            filled_indices = np.argwhere(current_board != 0)

            for h, w in filled_indices:
                color = current_board[h][w]
                source = coordinate_to_index_on_board(h, w)
                is_terminal = (h, w) in terminals_map
                amount_of_empties = len(get_empty_neighbors_on_board(current_board, source))
                should_grow = False

                if is_terminal and not has_neighbor_with_the_same_color(current_board, source) and amount_of_empties == 1:
                    should_grow = True
                elif not is_terminal and has_neighbor_with_the_same_color(current_board, source) and amount_of_empties == 1 and not has_two_neighbors_with_the_same_color(current_board, source):
                    should_grow = True
                
                if should_grow:
                    nh, nw = index_to_coordinate_on_board(get_empty_neighbors_on_board(current_board, source)[0])
                    current_board[nh][nw] = color
                    is_changed_temp = True    
                    actually_changed = True
                    
        return actually_changed

    # ----------------- heuristic method 2 -----------------
    # logic: if a terminal is at most two cells away from a corner and is at the edge,
    #        then it must fill the corner and move at most two more cells after turning

    def generate_roadmap(board):
        corners = [(0, 0), (0, W-1), (H-1, 0), (H-1, W-1)]
        deltas = [(-1, 0), (0, -1), (0, 1), (1, 0)]

        roadmap = {}

        for ch, cw in corners:
            if board[ch][cw] != 0: continue
            valid_directions = []
            for dh, dw in deltas:
                if is_within_board_2d(ch + dh, cw + dw):
                    valid_directions.append((dh, dw))
            (d_h1, d_w1), (d_h2, d_w2) = valid_directions
            current_inner_cell = (ch + d_h1 + d_h2, cw + d_w1 + d_w2)
            for r in range(1, 4):
                t_h1, t_w1 = ch + r * d_h1, cw + r * d_w1
                t_h2, t_w2 = ch + r * d_h2, cw + r * d_w2
                path_is_clear = True

                for k in range(1, r):
                    if board[ch + k * d_h1][cw + k * d_w1] != 0 or board[ch + k * d_h2][cw + k * d_w2] != 0:
                        path_is_clear = False
                
                if path_is_clear:
                    the_path = []
                    for k in range(r-1, -1, -1):
                        the_path.append((ch + k * d_h1, cw + k * d_w1))
                    for k in range(1, r):
                        the_path.append((ch + k * d_h2, cw + k * d_w2))
                    roadmap[(t_h1, t_w1)] = (the_path + [(t_h2, t_w2)], current_inner_cell)
                    roadmap[(t_h2, t_w2)] = (the_path[::-1] + [(t_h1, t_w1)], current_inner_cell)

        return roadmap

    # we need this roadmap for the upcoming heuristic
    roadmap = generate_roadmap(board)

    def apply_corner_moves_enhanced(current_board):

        filled_cells = {}
        for h, w in np.argwhere(current_board != 0):
            filled_cells[(h, w)] = current_board[h][w]
        
        # th: terminal height (as position)
        # tw: terminal width  (as position)
        for (th, tw) in roadmap:
            if (th, tw) not in filled_cells:
                continue
            color = current_board[th][tw]
            path = roadmap[(th, tw)][0]

            road_is_empty = True
            if len(path) <= 4:
                for i in range(len(path)):
                    if current_board[path[i][0]][path[i][1]] != 0:
                        road_is_empty = False
                if road_is_empty:
                    for i in range(len(path)):
                        current_board[path[i][0]][path[i][1]] = color
            else:
                inner_cell = roadmap[((th, tw))][1]
                for i in range(len(path)):
                    if current_board[path[i][0]][path[i][1]] != 0:
                        road_is_empty = False
                if road_is_empty and current_board[inner_cell[0]][inner_cell[1]] == 0:
                    for i in range(len(path)):
                        current_board[path[i][0]][path[i][1]] = color

    # ----------------- heuristic method 3 -----------------
    # logic: for each edge, we check for a specific condition: there exists two colors on the same edge with distance less than three
    # if that is the case, we end up filling the empty distance

    def fill_in_between(current_board):
        actually_changed = False
        
        edges = []
        edges.append([(0, w) for w in range(W)])   # top edge
        edges.append([(h, 0) for h in range(H)])   # left edge
        edges.append([(h, W-1) for h in range(H)]) # right edge
        edges.append([(H-1, w) for w in range(W)]) # bottom edge

        for edge_coords in edges:
            edge_values = [current_board[eh][ew] for eh, ew in edge_coords]
            for i in range(len(edge_values) - 2):
                color = edge_values[i]
                if color == 0: continue

                if edge_values[i+1] == 0 and edge_values[i+2] == color:
                    h_empty, w_empty = edge_coords[i+1]
                    if current_board[h_empty][w_empty] == 0:
                        current_board[h_empty][w_empty] = color
                        actually_changed = True

                if i+3 < len(edge_values):
                    if edge_values[i+1] == 0 and edge_values[i+2] == 0 and edge_values[i+3] == color:
                        h_empty_1, w_empty_1 = edge_coords[i+1]
                        h_empty_2, w_empty_2 = edge_coords[i+2]
                        if current_board[h_empty_1][w_empty_1] == 0 and current_board[h_empty_2][w_empty_2] == 0:
                            current_board[h_empty_1][w_empty_1] = color
                            current_board[h_empty_2][w_empty_2] = color
                            actually_changed = True
                            
        return actually_changed

    # ----------------- heuristic method 4 -----------------
    # dead-end logic: if an empty cell has zero or one empty neighbor, then it must inherit
    #                 a color of one of its neighbors. most likely, we won't encounter empty 
    #                 cells with zero empty neighbors

    def apply_bottleneck_moves(current_board):
        true_terminals_set = set(terminals_map.keys())
        is_changed_temp = True
        actually_changed = False
        
        def get_slices_for_source_and_neighbor(dh, dw):
            source_slice = make_slice_2d(dh, dw)
            neighbor_slice = make_slice_2d(-dh, -dw)
            return source_slice, neighbor_slice
        
        while is_changed_temp:
            is_changed_temp = False
            empty_mask = (current_board == 0)
            empty_neighbor_counts = np.zeros((H, W), dtype=int)

            for dh, dw in all_neighbors_on_board:
                source_slice, neighbor_slice = get_slices_for_source_and_neighbor(dh, dw)
                empty_neighbor_counts[source_slice] += empty_mask[neighbor_slice]

            bottleneck_mask = empty_mask & (empty_neighbor_counts <= 1)
            bottleneck_coords = np.argwhere(bottleneck_mask)

            if bottleneck_coords.size == 0: break

            for h, w in bottleneck_coords:
                potential_colors = []
                for dh, dw in all_neighbors_on_board:
                    nh, nw = h + dh, w + dw

                    if not is_within_bounds(0, nh, nw): continue

                    neighbor_color = current_board[nh][nw]
                    if neighbor_color == 0: continue

                    is_terminal = (nh, nw) in true_terminals_set

                    if is_terminal:
                        if not has_neighbor_with_the_same_color(current_board, coordinate_to_index_on_board(nh, nw)):
                            potential_colors.append(neighbor_color)
                    else:
                        if not has_two_neighbors_with_the_same_color(current_board, coordinate_to_index_on_board(nh, nw)):
                            potential_colors.append(neighbor_color)

                # we avoid modifying if there are multiple candidates
                unique_candidates = np.unique(potential_colors)
                if len(unique_candidates) == 1:
                    current_board[h][w] = unique_candidates[0]
                    is_changed_temp = True
                    actually_changed = True
                    
        return actually_changed

    # ----------------- end of heuristics -----------------
    
    # apply heuristics here
    global_change = True
    board_heuristics = np.array(board)

    # we only apply corner heuristic if the board is larger than a threshold
    # if not done, then it would be trying to fill two corners at the same time.
    if H > 7 and W > 7:
        apply_corner_moves_enhanced(board_heuristics)

    # here, we keep returning a boolean value to keep the loop running until no change occurs
    while global_change:
        is_changed_after_first = apply_forced_moves(board_heuristics)
        is_changed_after_third = fill_in_between(board_heuristics)
        is_changed_after_fourth = apply_bottleneck_moves(board_heuristics)
        global_change = is_changed_after_first or is_changed_after_third or is_changed_after_fourth

    # print(board_heuristics) # uncomment to see the current state after all of the heuristics

    # this is clever: for some boards, heuristics will fill the entire board.
    # in that case, instead of running the whole solver, we simply exit the program.
    if np.argwhere(board_heuristics == 0).size == 0:
        print("Taking the shortcut")
        plot_solution(board_heuristics, board_heuristics)
        return

    # in case the heuristics couldn't fill the entire board,
    # we simply accept the result and force the lower bounds
    for h, w in np.argwhere(board_heuristics != 0):
        l[get_index(board_heuristics[h][w] - 1, h, w)] = 1

    # -----------
    # constraints
    # -----------

    # constraint 1: each cell may contain only one color

    aij_1 = np.repeat(1, total_vars)
    row_1 = np.repeat(range(H * W), K)
    col_1 = np.arange(total_vars).reshape(K, -1).T.flatten()
    b_1 = np.repeat(1, H * W)
    senses_1 = np.repeat("E", H * W)

    current_row_counter = H * W

    # constraint 2: every terminal cell must have one neighbor with the same color

    row_2 = []
    col_2 = []

    for index in all_terminal_indices:
        current_neighbor_indices = get_neighbor_indices(index)
        repeat = len(current_neighbor_indices)
        row_2.extend(np.repeat(current_row_counter, repeat))
        col_2.extend(current_neighbor_indices)
        current_row_counter += 1

    aij_2 = np.repeat(1, len(row_2))
    row_2 = np.array(row_2)
    col_2 = np.array(col_2)
    b_2 = np.repeat(1, current_row_counter - H * W)
    senses_2 = np.repeat("E", b_2.size)

    # reset back the current_row_counter so it can be used again for the remaining part, for safety
    current_row_counter = H * W + all_terminal_indices.size

    # constraint 3: every non-terminal cell must have two neighbors with the same color if they possess that color
    # logic: this constraint has two parts: 
    #           part 1: -2*x + sum >= 0
    #           part 2:  2*x + sum <= 4
    #        with this logic, if x = 0, then the sum of neighbors does not matter for the given color
    #                         if x = 1, then sum is essentially 2, as 2 <= sum <= 2

    # stole the old and good vectorized constraint generation,
    # that is why namings are a bit off
    indices = np.arange(total_vars).reshape(K, H, W)

    all_adjacencies = []
    for dz, dx, dy in all_neighbors:
        covering = indices[make_slice_3d(dz, -dx, -dy)]
        covered = indices[make_slice_3d(dz, dx, dy)]
        test = np.concatenate((covering.flatten(), covered.flatten())).reshape(2, -1).T
        all_adjacencies.extend(test)
        
    all_adjacencies = np.array(all_adjacencies)

    sort_order = np.argsort(all_adjacencies[:, 0])
    all_adjacencies = all_adjacencies[sort_order]

    is_terminal_mask = np.isin(all_adjacencies[:, 0], all_terminal_indices)
    filtered_adjacencies = all_adjacencies[~is_terminal_mask]

    distinct_rows, new_row_indices = np.unique(filtered_adjacencies[:, 0], return_inverse=True)

    # constraint 3a: -2*x + sum >= 0
    row_3a_self = np.arange(distinct_rows.size) + current_row_counter
    col_3a_self = distinct_rows
    aij_3a_self = np.repeat(-2, row_3a_self.size)

    row_3a_neigh = new_row_indices + current_row_counter
    col_3a_neigh = filtered_adjacencies[:, 1]
    aij_3a_neigh = np.ones(row_3a_neigh.size)

    b_3a = np.zeros(distinct_rows.size)
    senses_3a = np.repeat("G", distinct_rows.size)

    current_row_counter += distinct_rows.size

    # constraint 3b: 2*x + sum <= 4
    row_3b_self = np.arange(distinct_rows.size) + current_row_counter
    col_3b_self = distinct_rows
    aij_3b_self = np.repeat(2, row_3a_self.size)

    row_3b_neigh = new_row_indices + current_row_counter
    col_3b_neigh = filtered_adjacencies[:, 1]
    aij_3b_neigh = np.ones(row_3b_neigh.size)

    b_3b = np.zeros(distinct_rows.size) + 4
    senses_3b = np.repeat("L", distinct_rows.size)

    # -------------------------
    # constructing the A matrix
    # -------------------------

    aij = np.concatenate((aij_1, aij_2, aij_3a_neigh, aij_3b_neigh, aij_3a_self, aij_3b_self))
    row = np.concatenate((row_1, row_2, row_3a_neigh, row_3b_neigh, row_3a_self, row_3b_self))
    col = np.concatenate((col_1, col_2, col_3a_neigh, col_3b_neigh, col_3a_self, col_3b_self))
    b = np.concatenate((b_1, b_2, b_3a, b_3b))
    senses = np.concatenate((senses_1, senses_2, senses_3a, senses_3b))

    A = sp.csr_matrix((aij, (row, col)), shape = (b.size, total_vars))

    # -----------------------
    # solution and formatting
    # -----------------------

    x_star, _ = mixed_integer_linear_programming("minimize", A, senses, b, c, l, u, types, names)
    x_star = np.argmax(np.array(x_star).reshape(K, H, W), axis = 0) + 1
    plot_solution(x_star, board_heuristics) # change board_heuristics with board to see the original terminal indices


board = np.array([ # 15 x 18 with 16 colors
    [0,  0,  0,  0,  0,  0,  0,  1,  2,  0,  0,  0,  0,  0,  3],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4],
    [0,  2,  3,  5,  6,  0,  0,  7,  1,  0,  0,  4,  0,  8,  9],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0, 10,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 11],
    [0,  0,  0,  0,  0,  0,  5,  0,  0,  0,  0,  0,  8,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0, 12,  0,  0,  0,  0,  0,  0],
    [0,  0,  0, 13,  0,  0,  0, 12,  0,  0,  0,  0,  0,  0, 11],
    [0,  0,  0, 10,  0, 14,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 13,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  7,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0, 14,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0, 15,  0,  0,  0, 15,  0,  0,  0, 16,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0, 16,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  9],
])

board = np.array([ # 8 x 8 with 8 colors, a test board I made to see if the program will take the shortcut
    [0, 0, 0, 1, 0, 0, 0, 2],
    [0, 3, 0, 2, 1, 0, 4, 5],
    [4, 0, 0, 0, 0, 0, 0, 0],
    [6, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 0],
    [0, 7, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 5],
    [0, 0, 0, 0, 6, 7, 0, 8]
])

flow_free_solver_with_heuristics(board)


# In[ ]:


# here are some of the boards to test the solver

# ------------
# empty boards
# ------------

board = np.array([ # 8 x 8
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

board = np.array([ # 12 x 12
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

board = np.array([ # 15 x 18
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
]) 

board = np.array([ # 12 x 15
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

board = np.array([ # 5 x 5
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])

# -----------------------------------------------
# filled boards that was used to test the program
# -----------------------------------------------

board = np.array([ # 5 x 5 with 3 colors
    [0, 1, 0, 0, 2],
    [0, 3, 0, 3, 1],
    [0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])

board = np.array([ # 12 x 12 with 13 colors
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 4, 0, 0, 0, 0, 0, 5, 0],
    [0, 0, 0, 0, 0, 0, 6, 7, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 9, 0, 0, 0, 0, 10, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 12, 0, 11, 0, 0, 0],
    [13, 0, 0, 9, 0, 0, 0, 0, 12, 10, 6, 0],
    [0, 13, 0, 0, 5, 0, 0, 0, 11, 7, 0, 0]
])

board = np.array([ # 15 x 18 with 13 colors
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
    [0, 0, 2, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0],
    [0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 8, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 9, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [12, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 1, 0, 0, 0],
    [13, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 12, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])

board = np.array([ # 12 x 15 with 8 colors
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 0, 0],
    [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 0, 6, 7, 0, 8, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
])

board = np.array([ # 8 x 8 with 8 colors
    [0, 0, 0, 0, 0, 0, 1, 2],
    [0, 3, 0, 0, 3, 4, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 5],
    [0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 6, 0, 0],
    [4, 6, 0, 0, 0, 0, 0, 7],
    [7, 5, 0, 8, 0, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

board = np.array([ # 12 x 12, testing stuff
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
])

board = np.array([ # 15 x 18 with 16 colors
    [0,  0,  0,  0,  0,  0,  0,  1,  2,  0,  0,  0,  0,  0,  3],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4],
    [0,  2,  3,  5,  6,  0,  0,  7,  1,  0,  0,  4,  0,  8,  9],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0, 10,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 11],
    [0,  0,  0,  0,  0,  0,  5,  0,  0,  0,  0,  0,  8,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0, 12,  0,  0,  0,  0,  0,  0],
    [0,  0,  0, 13,  0,  0,  0, 12,  0,  0,  0,  0,  0,  0, 11],
    [0,  0,  0, 10,  0, 14,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 13,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  7,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0, 14,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0, 15,  0,  0,  0, 15,  0,  0,  0, 16,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0, 16,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  9],
]) 

board = np.array([ # 8 x 8 with 8 colors
    [0, 0, 0, 1, 0, 0, 0, 2],
    [0, 3, 0, 2, 1, 0, 4, 5],
    [4, 0, 0, 0, 0, 0, 0, 0],
    [6, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 0],
    [0, 7, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 5],
    [0, 0, 0, 0, 6, 7, 0, 8]
])


# In[ ]:


def plot_solution(solution_grid, original_board):
    
    # this script is to visualize the solution, created by AI

    H, W = solution_grid.shape
    
    # Create a custom colormap
    unique_vals = np.unique(solution_grid)
    num_colors = len(unique_vals)
    cmap = plt.cm.get_cmap("tab20", num_colors) 
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot the grid
    im = ax.imshow(solution_grid, cmap=cmap, origin="upper")
    
    # Draw gridlines
    ax.set_xticks(np.arange(-.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-.5, H, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Remove major ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Annotate the original endpoints
    for r in range(H):
        for c in range(W):
            val = original_board[r, c]
            if val != 0:
                text = ax.text(c, r, str(val), ha="center", va="center", 
                               color="white", fontsize=12, fontweight='bold')
                # Use the explicitly imported 'pe' module here
                text.set_path_effects([pe.withStroke(linewidth=3, foreground='black')])

    plt.title("Solved Numberlink Board")
    plt.tight_layout()
    plt.show()

