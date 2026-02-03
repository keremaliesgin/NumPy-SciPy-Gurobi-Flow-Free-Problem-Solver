This script formulates a Numberlink/Flow Free problem as a mixed-integer linear programming model, and solves it by pre-processing the board and handing the updated board to Gurobi.

How it works:

  First of all, I used Gurobi as it is more up-to-date than CPLEX and performed way better than CPLEX.

  The Gurobi solver function: mixed_integer_linear_programming. Here is what each parameter represents:
  
        # direction: in linear/integer programming models, we either minimize or maximize something
        # A: the matrix of constraints, each row represents a constraint and each column represents a decision variable
        # senses: the vector of equalities, less thans and greater thans
        # b: the vector of right hand sides
        # c: the objective vector
        # l: lower bounds of all of the decision variables
        # u: upper bounds of all of the decision variables
        # types: each decision variable is either "continuous", "integer" or "binary"; we define it with this vector.
        # names: the vector to name all of the decision variables

  Gurobi takes all of these arguments and uses them to define the problem and solves it.

  Before we move into how we craft the model, let's analyze the general function and understand its structure:
  
      --> The function starts with a couple of initializations. We have K as the amount of colors, H as the height of the board and W as the width of the board. We define a couple of helper functions to use later on. 
      --> Then, we move onto initializing the decision variables. We force the terminals' lower bounds to be 1 so that Gurobi knows which points are terminals. And then we save the original terminals somewhere safe.
      --> Then, we move onto the heuristic methods. They have their own helper functions and they process the board before the model is crafted. Sometimes, they are able to fill the entire board (mostly in smaller boards) so that Gurobi does not even have to run.
      --> If the board is not filled completely during the heuristic methods, then we force the lower bounds once again, and continue crafting the model.
      --> We craft the model and hand it to the Gurobi with appropiate parameters.
      
  Now that we understand the general structure of the script, let's analyze the model first of all:
  
      --> We have the decision variables x_k_h_w: 1 if color k is in the intersection of row h and column w, 0 otherwise. This allows for layering. 
          # normally, one would expect this problem as a graph such as G = (V, E), however, in this case, the amount of decision variables becomes way too much. in fact, I ran the flow_free_solver_with_graph for a board, and it ran more than 10 minutes without providing a solution. that is why I used the vertex-only model.
      --> Now, we define our objective. Our goal is to connect the dots appropiately, which is not really related to maximizing or minimizing something, so our objective function is arbitrary.
      --> And now, we define our constraints:
          1: Every cell must be filled with exactly one color.
          2: Every terminal cell must have exactly one neighbor with the same color.
          3: Every non-terminal cell must have exactly two neighbors with the same color.
  
  Now that we have the entire model planned, let's craft it.
  Decision variables: x_k_i_j: 1 if color k is in the intersection of row h and column w, 0 otherwise.
  
        # names: a list comprehension
        # types: "B" for all of the variables
        # l: 0 for the empty cells, 1 for the all of the known variables
        # u: 1 for all of the known variables
        
  Objective: as it is arbitrary, we just set it to be complete zeros.
      
        # c: full of zeros
      
  Constraints: We generate a huge A matrix in the format sp.csr_matrix((aij, (row, col)), shape = (b.size, total_vars)).
      
          Constraint 1: Every cell must be filled with exactly one color.
              --> sum over k in {1, ..., K} x_k_h_w = 1 for h in {1, ..., H} and w in {1, ..., W}
                  # as we can see, we have H * W rows with K elements in each. every element has the coefficient 1. for the col indices, we take a range; reshape, transpose and flatten it.
          Constraint 2: Every terminal cell must have exactly one neighbor with the same color.
              --> sum of neighbors with same colors = 1 for every terminal
                  # luckily, we already had the locations of terminals as numerical values, with the help of helper functions, we quickly iterate on the terminals, gather the exact col, row and aij values. one thing to note is how we use current_row_counter to not overwrite the existing constraints.
          Constraint 3: Every non-terminal cell must have exactly two neighbors with the same color.
              --> this constraint has two parts (let me denote the empty cell with the desired color as 'x', with its neighbors' sum as 'sum'):
                    # x = 1: we want the sum to be exactly two, as x is a flow
                    # x = 0: we do not care how many neighbors x will have with the desired colors
                  do note that, we have to do this for EVERY empty cell for EVERY color layer, if we were to do this manually, it would be taking a lot of time to just to create the matrix. instead, we use numpy vectorization to quickly get all of the adjacencies.
                  now, let's define the actual constraint:
                    3a: -2x + sum >= 0
                    3b: 2x + sum <= 4
                  this is clever, as if x = 1, then from 3a, sum >= 2 and from 3b, sum <= 2, effectively squeezing the sum to 2. if x = 0, 0 <= sum <= 4, as it should be.
                  we handle these constraints in two parts: generating the aij, row and col indices for x values and neighbors separately. we generate senses and b in one go though
        
  Now that we have finished               
  
















  
        
