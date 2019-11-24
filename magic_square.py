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

from deap import base, tools

import numpy as np
from random import choice, random, sample

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
        ar = Utils.to_array(self.to_ms_seq(list(self)))
        st = '    ' +  \
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

            ar = Utils.to_array(sq)
            #       col sums, row sums, [major diag, minor diag]
            sums = (list(np.sum(ar, axis=0).tolist()) +
                    list(np.sum(ar, axis=1).tolist()) +
                    [np.sum(ar.diagonal()), np.sum(np.fliplr(ar).diagonal())])
            avg = round(sum(sums)/len(sums))
            # Fitness is defined as the sum of the squares of the differences from the average.
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


class Population(list):
    """
    The Population holds the collection of Individuals that will undergo evolution.
    """

    # Which generation this is.
    gen = None

    # An expressions that when executed generates an individual.
    individual_generator = None

    # The size of the population
    pop_size = None

    # The maximum number of generations to allow.
    max_gens = None

    # Probabilities of crossover and mutation.
    CXPB = None
    MUTPB = None

    verbose = None

    toolbox = base.Toolbox( )

    best_ind = None
    former_best_ind = None


    def __init__(self, pop_size, max_gens, individual_generator,
                 mate=None, CXPB=0.7,
                 mutate=None, MUTPB=0.5,
                 select=tools.selTournament, verbose=True):
        Population.gen = 0
        # individual_generator is a function that when executed returns an individual.
        # See its use in generating the population at the end of __init__.
        Population.individual_generator = individual_generator
        Population.pop_size = pop_size
        Population.max_gens = max_gens
        Population.CXPB = CXPB
        Population.MUTPB = MUTPB
        Population.prev_best_fitness = float('inf')
        Population.verbose = verbose

        # Choose the genetic operators.

        # Select a crossover operator. We are using our own crossover operator.
        self.toolbox.register("mate", mate)

        # Select a mutation operator. We are using our own mutation operator.
        self.toolbox.register("mutate", mutate)

        # operator for selecting individuals for breeding the next
        # generation: each individual of the current generation
        # is replaced by the 'fittest' (best) of <n> individuals
        # drawn randomly from the current generation.
        # I like tournament selection. Notice the difference tournament size makes.
        self.toolbox.register("select", select, tournsize=7)

        # Create a list of Individuals as the initial population.
        pop = [Population.individual_generator() for _ in range(pop_size)]
        super().__init__(pop)

    def eval_all(self):
        for ind in self:
            if not ind.fitness.valid:
                ind.set_fitness( )

        Population.former_best_ind = Population.best_ind
        Population.best_ind = tools.selBest(self, 1)[0]

    def generate_next_generation(self):
        Population.gen += 1

        # Select the next generation individuals
        # Use tournament selection to select half the base population.
        # So these are the elite elements of the population.
        # pop-size of these elite elements are selected.
        # Will almost certainly include duplicates.
        # But no guarantee that the best is kept.
        # In addition, fill the remainder of the population with new random elements.
        # At the end we add the best of the current population.
        offspring = self.toolbox.select(self, self.pop_size//2) + \
                    [Population.individual_generator() for _ in range(self.pop_size//2)]

        # Now make each element a clone of itself.
        # We do that because the genetic operators modify the elements in place.
        # (It is not functional.)
        offspring = list(map(self.toolbox.clone, offspring))

        # offspring will be the new population.
        # Apply crossover and mutation to the offspring.
        # Mark those that are the result of crossover or mutation as having invalid fitnesses.

        # Pair the offspring off even-odd.
        for (child_1, child_2) in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability CXPB
            # Set the parents to None so that we can tell whether crossover occurred
            child_1.parent_1 = child_1.parent_2 = None
            child_2.parent_1 = child_2.parent_2 = None
            # Keep track of what the elements were before crossover.
            c_1 = list(child_1)
            c_2 = list(child_2)
            if random( ) < self.CXPB:
                # Set the parents to c_1 and c_2 so that we can tell that crossover occurred
                child_1.parent_1 = child_2.parent_1 = c_1
                child_1.parent_2 = child_2.parent_2 = c_2
                self.toolbox.mate(child_1, child_2)

                if child_1 == child_1.parent_1 or child_1 == child_1.parent_2:
                    child_1.parent_1 = child_1.parent_2 = None
                if child_2 == child_2.parent_1 or child_2 == child_2.parent_2:
                    child_2.parent_1 = child_2.parent_2 = None

                # fitness values of the children must be recalculated later
                del child_1.fitness.values
                del child_2.fitness.values

        for mutant in offspring:
            mutant.pre_mutation = None
            # mutate an individual with probability MUTPB
            if random( ) < self.MUTPB:
                mutant.pre_mutation = list(mutant)
                # Utils.mut_swap_pairs(mutant)
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

        # The new generation replaces the current one.
        self[:] = offspring
        # Elitism. Add the best of the current generation to the new generation.
        self[0] = Population.best_ind

        Utils.print_stats(self)


class Utils:

    @staticmethod
    def cx_all_diff(ind_1, ind_2):
        """
        Perform crossover between ind_1 and ind_2 without violating all_different.
        This is our own crossover operator. Don't change ind_1 until after the
        second cx_over.
        """
        ind_1_new = Utils.cx_over(ind_1, ind_2)
        ind_2[:] = Utils.cx_over(ind_2, ind_1)
        ind_1[:] = ind_1_new

    @staticmethod
    def cx_over(ind_1, ind_2):
        # This ensures that the rotations are non-trivial.
        inner_indices = [2, 3, 4, 5]
        ind_1r = Utils.rotate_by(ind_1, choice(inner_indices))
        ind_2r = Utils.rotate_by(ind_2, choice(inner_indices))
        indx = choice(inner_indices)

        child = ind_1r[: indx] + [item for item in ind_2r if item not in ind_1r[: indx]]
        return child

    @staticmethod
    def mut_swap_pairs(ind):
        """
        This is our own mutation operator. It swaps twp pairs.
        """
        # Ensures that the two index positions are different.
        [pair_1, pair_2] = sample([1, 3, 5, 7], 2)
        (ind[pair_1], ind[pair_1+1]),    (ind[pair_2], ind[pair_2+1]) = \
        (ind[pair_2], ind[pair_2+1]),    (ind[pair_1], ind[pair_1+1])

    @staticmethod
    def print_best(best_ind):
        cx_segment = f'{Utils.to_ind(best_ind.parent_1)}\n+\n{Utils.to_ind(best_ind.parent_2)} \n=>\n' \
                                                                                   if best_ind.parent_1 else ''
        mutation_segment = f'{Utils.to_ind(best_ind.pre_mutation)} \n=>\n' if best_ind.pre_mutation else ''
        print(cx_segment + mutation_segment + str(best_ind))

    @staticmethod
    def print_stats(pop):
        # Evaluate the individuals (in the new generation) with invalid fitnesses.
        pop.eval_all()

        if pop.verbose and Population.best_ind != Population.former_best_ind:
            print(f"\n-- Generation {pop.gen} --")
            # Gather all the fitnesses in one list and print the stats.
            # Again, ind.fitness.values is a tuple. We want the first value.
            fits = [ind.fitness.values[0] for ind in pop]
            mean = sum(fits) / pop.pop_size
            sum_sq = sum(x * x for x in fits)
            std = abs(sum_sq / pop.pop_size - mean ** 2) ** 0.5
            print(f"   Min: {int(min(fits))}; Mean {round(mean, 2)}; Max {int(max(fits))}; Std {round(std, 2)}")
            Utils.print_best(Population.best_ind)

    @staticmethod
    def rotate_by(lst, amt):
        return lst[amt:] + lst[:amt]

    @staticmethod
    def to_array(lst):
        """
        Creates a side x cols 2-dimensional numpy array.
        Create a 1-D numpy array and then turn it into a side x cols array.
        """
        ar = np.reshape(np.array(lst), (Individual.side, Individual.side))
        return ar
    
    @staticmethod
    def to_ind(lst):
        ind = Individual()
        ind[:] = lst
        ind.set_fitness()
        return ind


def main(verbose=True):

    Individual.side = 3

    # create an initial population.
    pop = Population(pop_size=100, max_gens=50,
                     individual_generator=Individual,
                     mate=Utils.cx_all_diff, CXPB=0.7,
                     mutate=Utils.mut_swap_pairs, MUTPB=0.5,
                     verbose=verbose)
    Utils.print_stats(pop)
    prefix = 'Unsuccessful'
    for _ in range(Population.max_gens):
        best_fit = Population.best_ind.fitness.values[0]
        if best_fit == 0:
            prefix = 'Successful'
            break
        pop.generate_next_generation()
    print(f"-- {prefix} evolution. {Population.gen} generations. --")


if __name__ == "__main__":
    main()
