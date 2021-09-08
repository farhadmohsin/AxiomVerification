from ortools.sat.python import cp_model

class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        for v in self.__variables:
            print('%s=%i' % (v, self.Value(v)), end=' ')
        print()

    def solution_count(self):
        return self.__solution_count

class VarArraySolutionCollector(cp_model.CpSolverSolutionCallback):

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.solution_list = []

    def on_solution_callback(self):
        self.solution_list.append([self.Value(v) for v in self.__variables])

def SearchForAllSolutionsSampleSat():
    """Showcases calling the solver to search for all solutions."""
    # Creates the model.
    model = cp_model.CpModel()

    # Creates the variables.
    num_vals = 3
    x = []
    upper_bounds = [5, 8, 5]

    for i in range(num_vals):
        x.append(model.NewIntVar(0, upper_bounds[i], f'x[{i}]'))

    # Ax <= h are all the constraints
    A = [[1,1,1], [3,-1,2], [5,2,3]]
    h = [5, 2, 10]

    # Create the constraints.
    
    for i in range(len(h)):
        constraint_expr = [A[i][k] * x[k] for k in range(num_vals)]
        model.Add(sum(constraint_expr) <= h[i])
    
    # Create a solver and solve.
    solver = cp_model.CpSolver()
    # solution_printer = VarArraySolutionPrinter(x)
    # status = solver.SearchForAllSolutions(model, solution_printer)
    
    solution_collector = VarArraySolutionCollector(x)
    status = solver.SearchForAllSolutions(model, solution_collector)

    print('Status = %s' % solver.StatusName(status))
    print('All solutions:', solution_collector.solution_list)
    print('Number of solutions found: %i' % len(solution_collector.solution_list))

if __name__ == '__main__':
    SearchForAllSolutionsSampleSat()
 