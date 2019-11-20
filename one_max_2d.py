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

from deap import base, tools

import numpy as np
from random import choice, random  #, seed

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

    parent_1 = parent_2 = pre_mutation = None

    def __init__(self):
        # DEAP stores fitness values in a Fitness class, which offers some basic operations. (See base.py.)
        # Must set the "weights" before instantiating the Fitness class.
        # Since we will have only a single fitness value, the weights, which must be a tuple,
        # will always be either (1,) (to maximize) or (-1, ) (to minimize).
        # (Or to minimize can use negative weights as in set_fitness below.)
        base.Fitness.weights = (1, )
        self.fitness = base.Fitness()

        # Individual is a subclass of list. So must have a list.
        the_list = [choice([0, 1]) for _ in range(Individual.rows*Individual.cols)]
        super().__init__(the_list)

    def __str__(self):
        """
        Convert this list to an array of elements for printing.
        """
        ar = self.to_np_array()
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
        sums_of_cols = np.sum(self.to_np_array())
        fit = np.sum(sums_of_cols)
        self.fitness.values = (fit, )

    def to_np_array(self):
        """
        Creates a rows x cols 2-dimensional numpy array.
        Create a 1-D numpy array and then "reshape" it into a rows x cols array.
        """
        ar0 = np.array(self)
        ar = np.reshape(ar0, (self.rows, self.cols))
        return ar


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

    count = 0
    prev_best_fitness = None

    verbose = None

    # Using the toolbox allows us to select some arguments in advance.
    toolbox = base.Toolbox( )

    def __init__(self, pop_size, max_gens, individual_generator, CXPB=0.5, MUTPB=0.2, verbose=True):
        Population.gen = 0
        # individual_generator is a function that when executed returns an individual.
        # See its use in generating the population at the end of __init__.
        Population.individual_generator = individual_generator
        Population.pop_size = pop_size
        Population.max_gens = max_gens
        Population.CXPB = CXPB
        Population.MUTPB = MUTPB
        Population.prev_best_fitness = 0
        Population.verbose = verbose

        # Choose the genetic operators.

        # Select a crossover operator. (Note use of named parameter.)
        # I tend to like uniform-crossover.
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)

        # register a mutation operator with a probability to
        # flip each attribute/gene of 0.05
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

        # operator for selecting individuals for breeding the next
        # generation: each individual of the current generation
        # is replaced by the 'fittest' (best) of <n> individuals
        # drawn randomly from the current generation.
        # I like tournament selection. Notice the difference tournament size makes.
        self.toolbox.register("select", tools.selTournament, tournsize=7)

        # Create a list of Individuals as the initial population.
        pop = [Population.individual_generator() for _ in range(pop_size)]
        super().__init__(pop)

    def eval_all(self):
        Population.count = 0
        for ind in self:
            if not ind.fitness.valid:
                ind.set_fitness( )
                Population.count += 1

        best_ind = tools.selBest(self, 1)[0]
        return best_ind

    def generate_next_generation(self):
        Population.gen += 1

        # Select the next generation individuals
        # Use tournament selection to select the base population.
        # So these are the elite elements of the population.
        # pop-size of these elite elements are selected.
        # Will almost certainly include duplicates.
        # But no guarantee that the best is kept.
        offspring = self.toolbox.select(self, self.pop_size)

        # Now make each element a clone of itself.
        # We do that because the genetic operators modify the elements in place.
        # (It is not functional.)
        offspring = list(map(self.toolbox.clone, offspring))

        # offspring will be the new population.
        # Apply crossover and mutation to the offspring.
        # Mark those that are the result of crossover or mutation as having invalid fitnesses.

        # Pair the offspring off even-odd.
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

        # Evaluate the individuals with invalid fitnesses.
        # Return the best individual and its fitness.
        best_ind = self.eval_all()
        best_fit = best_ind.fitness.values[0]

        if self.verbose and best_fit > Population.prev_best_fitness:
            self.print_stats(best_ind)

        return best_ind

    def print_stats(self, best_ind):
        print(f"\n-- Generation {self.gen} --")
        print(f"   Evaluated {Population.count} individuals")
        best_fit = best_ind.fitness.values[0]
        Population.prev_best_fitness = best_fit
        # Gather all the fitnesses in one list and print the stats.
        # Again, ind.fitness.values is a tuple. We want the first value.
        fits = [ind.fitness.values[0] for ind in self]
        mean = sum(fits) / self.pop_size
        sum_sq = sum(x * x for x in fits)
        std = abs(sum_sq / Population.pop_size - mean ** 2) ** 0.5
        print(f"   Min: {min(fits)}; Mean {round(mean, 2)}; Max {best_fit}; Std {round(std, 2)}", end='')
        Population.print_best(best_ind)

    @staticmethod
    def print_best(best_ind):
        cx_segment = f'{Population.to_ind(best_ind.parent_1)}\n\n+ (crossed with) \n' \
                     f'{Population.to_ind(best_ind.parent_2)} \n=>' if best_ind.parent_1 else ''
        mutation_segment = f'{Population.to_ind(best_ind.pre_mutation)}\n\n=> (mutated to) \n' \
                                                                    if best_ind.pre_mutation else ''
        print(cx_segment + mutation_segment + str(best_ind))

    @staticmethod
    def to_ind(lst):
        ind = Individual( )
        ind[:] = lst
        ind.set_fitness()
        return ind


def main(verbose=True):

    Individual.rows = 5
    Individual.cols = 6

    # create an initial population.
    pop = Population(pop_size=10, max_gens=200,
                     individual_generator=Individual,
                     verbose=verbose)
    best_ind = pop.eval_all()
    if verbose:
        pop.print_stats(best_ind)
    prefix = 'unsuccessful'
    for _ in range(Population.max_gens):
        best_fit = best_ind.fitness.values[0]
        if best_fit == Individual.rows*Individual.cols:
            prefix = 'successful'
            break
        best_ind = pop.generate_next_generation()

    # We consider the evoluation successful if max_fit indicates that best_ind is all 1's,
    # prefix = "" if best_fit == best_ind.rows*best_ind.cols else "un"
    print(f"\n-- After {pop.gen} generations, end of {prefix} evolution --")


if __name__ == "__main__":
    main()
