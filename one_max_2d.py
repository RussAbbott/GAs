#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.


#    example which maximizes the number or 1s in a 2D array.

# Taken from: https://deap.readthedocs.io/en/master/examples/ga_onemax.html
# Code from: https://github.com/DEAP/deap/blob/454b4f65a9c944ea2c90b38a75d384cddf524220/examples/ga/onemax.py

from deap import base
from Population_and_Utils import Population, Utils

import numpy as np
from random import choice

"""
Typically we need an Individual class and a Population class.
An individual is a list of "genes". These are what get mutated.
The Population is a list of individuals. 
"""


class Individual(list):
    """
    For this problem we want an Individual to be a 2-D array.
    Even so, we have to define it as a subclass of list and
    convert it to an array when we want to treat is as an array.
    """

    rows = None
    cols = None

    def __init__(self):
        # DEAP stores fitness values in a Fitness class, which offers some basic operations. (See base.py.)
        # Must set the "weights" before instantiating the Fitness class.
        # Since we will have only a single fitness value, the weights, which must be a tuple,
        # will always be either (1,) (to maximize) or (-1, ) (to minimize).
        # (Or to minimize can use negative weights as in set_fitness below.)
        base.Fitness.weights = (1, )
        self.fitness = base.Fitness()

        self.parent_1 = self.parent_2 = self.pre_mutation = None

        # Individual is a subclass of list. So must have a list.
        the_list = [choice([0, 1]) for _ in range(Individual.rows*Individual.cols)]
        super().__init__(the_list)

    def __str__(self):
        """
        Convert this list to a grid for printing.
        """
        if not self.fitness.valid:
            self.set_fitness()
        # ar = self.to_np_array()
        ar = Utils.list_to_array(self, Individual.rows, Individual.cols)
        st = '\n    ' +  \
             '\n    '.join([             # Put a '\n' between rows
                            ' '.join([   # Put a ' ' between elements in a row
                                      f'{ar[row, col]}' for col in range(self.cols)])
                                                        for row in range(self.rows)] )
        return st + f'         Fitness: {self.fitness.values[0]}'

    def set_fitness(self):
        """
        Compute the fitness of this individual and store it at self.fitness.values.
        In this (simple) case, fitness is the number of 1's.

        np.sum of a 2-D array returns a list containing the sums of the columns.
        We want the sum of those sums. ( We could simply have taken sum(self). )

        Again, recall that the fitness is a tuple.
        """
        sums_of_cols = np.sum(Utils.list_to_array(self, Individual.rows, Individual.cols))
        fit = np.sum(sums_of_cols)
        self.fitness.values = (fit, )


def main(verbose=True):

    Individual.rows = 5
    Individual.cols = 6

    # create an initial population.
    pop = Population(pop_size=10, max_gens=200,
                     individual_generator=Individual,
                     verbose=verbose)
    pop.run_evolution(Individual.rows*Individual.cols)


if __name__ == "__main__":
    main()
