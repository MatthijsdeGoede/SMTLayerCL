import z3


class VisualAdditionScorer:

    # Method to generate new instances of the solver, as no constraints can be removed
    def __sat(self, s1_dom, s2_dom, y_dom):
        # If one of the domains is empty, there can't be a feasible solution
        if len(s1_dom) == 0 or len(s2_dom) == 0 or len(y_dom) == 0:
            return False
        # Create a Z3 solver instance
        solver = z3.Solver()

        # Define variables, for simplicity we work directly with integer variables
        s1 = z3.Int('s1')
        s2 = z3.Int('s2')
        y = z3.Int('y')

        # Ensure domain bounds, can also be used to enforce equality when passing domain of cardinality 1
        solver.add(z3.And(min(s1_dom) <= s1, s1 <= max(s1_dom)))
        solver.add(z3.And(min(s2_dom) <= s2, s2 <= max(s2_dom)))
        solver.add(z3.And(min(y_dom) <= y, y <= max(y_dom)))

        # Add sum constraint
        solver.add(s1 + s2 == y)

        # Return whether a satisfiable solution exists with the given domains
        return solver.check() == z3.sat

    def score(self, s1_dom, s2_dom, y_dom):
        # Keep the scores for the y values in a dictionary
        y_vals = dict()

        for v in y_dom:
            # Initialize both input domains to the full domain
            s1_dom_ = s1_dom.copy()
            s2_dom_ = s2_dom.copy()

            # First check values in the s1 domain
            for u in s1_dom:
                if not self.__sat({u}, s2_dom_, {v}):
                    s1_dom_.remove(u)

            # Then check values in the s2 domain
            for u in s2_dom:
                if not self.__sat(s1_dom_, {u}, {v}):
                    s2_dom_.remove(u)

            # Calculate the y_vals[v] as the product of the input cardinalities
            y_vals[v] = len(s1_dom_) * len(s2_dom_)

        # Return the scores for each of the y values
        return y_vals


if __name__ == '__main__':
    scorer = VisualAdditionScorer()
    s1_dom = {s1 for s1 in range(10)}
    s2_dom = {s2 for s2 in range(10)}
    y_dom = {y for y in range(19)}
    scores = scorer.score(s1_dom, s2_dom, y_dom)
    print(scores)