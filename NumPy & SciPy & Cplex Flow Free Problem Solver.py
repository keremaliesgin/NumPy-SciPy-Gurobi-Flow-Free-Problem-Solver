#!/usr/bin/env python
# coding: utf-8

# In[18]:


# load libraries
import numpy as np
import scipy.sparse as sp
import math
import cplex as cp
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


# In[19]:


def mixed_integer_linear_programming(direction, A, senses, b, c, l, u, types, names):
    # create an empty optimization problem
    prob = cp.Cplex()

    # add decision variables to the problem including their coefficients in objective and ranges
    prob.variables.add(obj = c.tolist(), lb = l.tolist(), ub = u.tolist(), types = types.tolist(), names = names.tolist())

    # define problem type
    if direction == "maximize":
        prob.objective.set_sense(prob.objective.sense.maximize)
    else:
        prob.objective.set_sense(prob.objective.sense.minimize)

    # add constraints to the problem including their directions and right-hand side values
    prob.linear_constraints.add(senses = senses.tolist(), rhs = b.tolist())

    # add coefficients for each constraint
    row_indices, col_indices = A.nonzero()
    prob.linear_constraints.set_coefficients(zip(row_indices.tolist(), col_indices.tolist(), A.data.tolist()))

    print(prob.write_as_string())
    # solve the problem
    prob.solve()

    # check the solution status
    print(prob.solution.get_status())
    print(prob.solution.status[prob.solution.get_status()])

    # get the solution
    x_star = prob.solution.get_values()
    obj_star = prob.solution.get_objective_value()

    return(x_star, obj_star)


# In[ ]:


def flow_free_solver(board):
    
    # decision variable: x_k_i_j: 1 if the cell in i'th row and j'th column contains color k, 0 otherwise

    # initialization

    H, W = board.shape
    colors = np.arange(np.max(board)) + 1
    K = colors.size

    total_var = K * H * W

    # fixing variables:

    names = np.array(["x_{}_{}_{}".format(k+1, h+1, w+1) for k in range(K) for h in range(H) for w in range(W)])
    types = np.repeat("B", total_var)
    c = np.repeat(0, total_var)

    l = np.zeros((K, H, W))
    nonzero_coords = np.array(board.nonzero()).T
    for h, w in nonzero_coords:
        l[board[h][w] - 1][h][w] = 1

    u = np.repeat(1, total_var)

    # constraint 1: every cell must be filled: H * W rows

    aij1 = np.repeat(1, total_var)
    row1 = np.repeat(range(H * W), K)
    col1 = np.arange(total_var).reshape(K, -1).T.flatten()
    b1 = np.repeat(1, H * W)
    senses1 = np.repeat("E", H * W)

    # all neighbors:

    neighbors = [(0, -1, 0), (0, 0, -1), (0, 0, 1), (0, 1, 0)]

    # constraint 2: every terminal must have only one neighbor with the same color

    nonzero_coords = np.array(l.nonzero()).T
    nonzero_indices = np.array([k * H * W + h * W + w for k, h, w in nonzero_coords])
    l = l.flatten() # now, we flatten l after our job is done with it

    aij2 = []
    row2 = []
    col2 = []
    b2 = np.repeat(1, nonzero_coords.shape[0])
    senses2 = np.repeat("E", nonzero_coords.shape[0])

    current_row_index = np.max(np.array(row1)) + 1

    for k, h, w in nonzero_coords:
        repeat = 0
        index = k * H * W + h * W + w
        for dk, dh, dw in neighbors:
            if (0 <= h + dh < H) and (0 <= w + dw < W):
                neighbor = k * H * W + (h + dh) * W + (w + dw)
                col2.append(neighbor)
                repeat += 1
        row2.extend(np.repeat(current_row_index, repeat))
        current_row_index += 1

    aij2.extend(np.repeat(1, len(row2)))

    # constraint 3: every nonterminal cell must have two neighbors with the same color
    # -2_x + sum >= 0 and 2x + sum <= 4 at the same time, this allows flexibility

    indices = np.arange(total_var).reshape(K, H, W)

    def make_slice(dz, dx, dy):
        sz = slice(None, -dz) if dz > 0 else slice(-dz, None)
        sx = slice(None, -dx) if dx > 0 else slice(-dx, None)
        sy = slice(None, -dy) if dy > 0 else slice(-dy, None)
        return sz, sx, sy

    all_adjacencies = []

    for dz, dx, dy in neighbors:
        
        covering = indices[make_slice(dz, -dx, -dy)]
        covered = indices[make_slice(dz, dx, dy)]

        test = np.concatenate((covering.flatten(), covered.flatten())).reshape(2, -1).T

        all_adjacencies.extend(test)
        
    all_adjacencies = np.array(all_adjacencies)

    sort_order = np.argsort(all_adjacencies[:, 0])
    all_adjacencies = all_adjacencies[sort_order]

    is_terminal_mask = np.isin(all_adjacencies[:, 0], nonzero_indices)
    filtered_adjacencies = all_adjacencies[~is_terminal_mask]

    distinct_rows, new_row_indices = np.unique(filtered_adjacencies[:, 0], return_inverse=True)

    # constraint 3a: -2_x + sum >= 0

    row3_self_a = np.arange(distinct_rows.size) + current_row_index
    col3_self_a = distinct_rows
    aij3_self_a = np.repeat(-2, row3_self_a.size)

    row3_neigh_a = new_row_indices + current_row_index
    col3_neigh_a = filtered_adjacencies[:, 1]
    aij3_neigh_a = np.ones(row3_neigh_a.size)

    aij3_a = np.concatenate((aij3_neigh_a, aij3_self_a))
    row3_a = np.concatenate((row3_neigh_a, row3_self_a))
    col3_a = np.concatenate((col3_neigh_a, col3_self_a))

    b3_a = np.zeros(distinct_rows.size)
    senses3_a = np.repeat("G", distinct_rows.size)

    current_row_index += distinct_rows.size

    # constraint 3b: 2x + sum <= 4

    row3_self_b = np.arange(distinct_rows.size) + current_row_index
    col3_self_b = distinct_rows
    aij3_self_b = np.repeat(2, row3_self_a.size)

    row3_neigh_b = new_row_indices + current_row_index
    col3_neigh_b = filtered_adjacencies[:, 1]
    aij3_neigh_b = np.ones(row3_neigh_b.size)

    aij3_b = np.concatenate((aij3_neigh_b, aij3_self_b))
    row3_b = np.concatenate((row3_neigh_b, row3_self_b))
    col3_b = np.concatenate((col3_neigh_b, col3_self_b))

    b3_b = np.zeros(distinct_rows.size) + 4
    senses3_b = np.repeat("L", distinct_rows.size)

    # finalization

    aij = np.concatenate((aij1, aij2, aij3_a, aij3_b))
    row = np.concatenate((row1, row2, row3_a, row3_b))
    col = np.concatenate((col1, col2, col3_a, col3_b))

    b = np.concatenate((b1, b2, b3_a, b3_b))
    senses = np.concatenate((senses1, senses2, senses3_a, senses3_b))

    A = sp.csr_matrix((aij, (row, col)), shape = (b.size, total_var))

    x_star, obj_star = mixed_integer_linear_programming("minimize", A, senses, b, c, l, u, types, names)
    x_star = np.argmax(np.array(x_star).reshape(K, H, W), axis = 0) + 1
    
    return x_star, obj_star


# In[21]:


# this script is to visualize the solution, created by AI

def plot_solution(solution_grid, original_board):
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



# In[ ]:


board = np.array([ # empty board to set up problems easily, 12 x 15
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
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    
])

board = np.array([ # CPLEX solved this problem in about 0.2 seconds
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

board = np.array([ # empty board to set up problems easily, 15 x 18
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    
])

board = np.array([ # CPLEX solved this problem in about 3 minutes, this has 3510 variables and 7264 constraints
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

x_star, obj_star = flow_free_solver(board)


# In[25]:


plot_solution(x_star, board) # obtain the solution from the previous cell

