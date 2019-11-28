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


#    Generate a 3x3 Magic Square

from deap import base
from Population_and_Utils import Population, Utils

import numpy as np
from random import sample

"""
Typically we need an Individual class and a Population class.
An individual is a list of "genes".
The Population is a list of individuals. 
"""


class Individual(list):
    """
    For this problem we want an Individual to be a 2-D array.
    Even so, we have to define it as a subclass of list and
    convert it to an array when we want to treat is as an array.
    """

    side = None


    def __init__(self):
        # DEAP stores fitness values in a Fitness class, which offers some basic operations. (See base.py.)
        # Must set the "weights" before instantiating the Fitness class.
        # Since we will have only a single fitness value, the weights, which must be a tuple,
        # will always be either (1,) (to maximize) or (-1, ) (to minimize).
        # (Or to minimize can use negative weights as in set_fitness below.)
        base.Fitness.weights = (-1, )
        self.fitness = base.Fitness()

        self.parent_1 = self.parent_2 = self.pre_mutation = None

        # Individual is a subclass of list. So must have a list.
        the_list = sample(list(range(1, 10)), 9)
        super().__init__(the_list)

    def __str__(self):
        """
        Convert this list to a grid for printing.
        """
        if not self.fitness.valid:
            self.set_fitness()
        ar = Utils.list_to_array(self.to_ms_seq(list(self)), Individual.side, Individual.side)
        st = '\n    ' +  \
             '\n    '.join([             # Put a '\n' between rows
                            ' '.join([   # Put a ' ' between elements in a row
                                      f'{ar[row, col]}' for col in range(self.side)])
                                                        for row in range(self.side)] )
        return st + f'    Sum of mismatches: {int(self.fitness.values[0])}'

    def set_fitness(self):
        """
        Compute the fitness of this individual and store it at self.fitness.values.
        """
        original_list = list(self)
        best_fit = float('inf')
        # Try starting at all positions in our list and wrapping around.
        # Rotation ignores the interpretation of the sequence as a magic square.
        # This essentially allows us additional mutations on the fly.
        for i in range(len(original_list)):
            # Try all rotations of self
            rotation = Utils.rotate_by(original_list, i)
            # An individual is interpreted as containing elements of a square in this order.
            # (1, 1), (0, 0), (2, 2), (0, 1), (2, 1), (0, 2), (2, 0), (1, 0), (1, 2).
            # In other words, if an individual is [a, b, c, d, e, f, g, h, i], this is taken
            # to mean the square: b d f
            #                     h a i
            #                     g e c
            # In particular, the function self.to_ms_seq, transforms an individual from
            # [a, b, c, d, e, f, g, h, i] to [b, d, f, h, a, i, g, e, c], which represents the
            # array above. Looked at "backwards, the array
            #                     a b c
            #                     d e f
            #                     g h i
            # is encoded as the list: [e, a, i, b, h, c, g, d, f] In other words, the center cell
            # of the array is first in the list. That is followed by the pairs of cells opposite
            # each other.
            sq = self.to_ms_seq(rotation)

            ar = Utils.list_to_array(sq, Individual.side, Individual.side)
            #       col sums, row sums, [major diag, minor diag]
            sums = (list(np.sum(ar, axis=0).tolist()) +
                    list(np.sum(ar, axis=1).tolist()) +
                    [np.sum(ar.diagonal()), np.sum(np.fliplr(ar).diagonal())])
            avg = round(sum(sums)/len(sums))
            # Fitness is defined as the sum of the differences from the average.
            fit = int(sum([abs(n-avg) for n in sums]))

            if fit < best_fit:
                best_fit = fit
                self[:] = rotation

        self.fitness.values = (best_fit, )

    @staticmethod
    def to_ms_seq(rotation):
        """
        Converts our specialized format for individuals to a more standard sequence for a 3x3 array.
        The middle element is first followed by the pairs of elements on opposite sides of the middle.
        This was intended to make exchanging pairs easier to write as a mutation. May not have been
        worth the trouble.
        :param:  [a, b, c, d, e, f, g, h, i]
        :return: [b, d, f, h, a, i, g, e, c]
        """
        ms_seq = [rotation[i] for i in [1, 3, 5, 7, 0, 8, 6, 4, 2]]
        return ms_seq


def main(verbose=True):

    Individual.side = 3

    # create an initial population.
    pop = Population(pop_size=100, max_gens=50,
                     individual_generator=Individual,
                     mate=Utils.cx_all_diff, CXPB=0.7,
                     mutate=Utils.mut_swap_pairs, MUTPB=0.5,
                     verbose=verbose)

    pop.run_evolution(0)


if __name__ == "__main__":
    main()
