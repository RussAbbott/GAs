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

from deap import base, tools

import numpy as np
from random import choice, random, sample

"""
Typically we need an Individual class and a Population class.
An individual is a list of "genes".
The Population is a list of individuals. 
"""


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

    select_pct_random = None

    verbose = None

    toolbox = base.Toolbox( )

    def __init__(self, pop_size, max_gens, individual_generator,
                 mate=tools.cxUniform, CXPB=0.7,
                 mutate=tools.mutFlipBit, MUTPB=0.5,
                 select=tools.selTournament,
                 select_pct_random=0.5,
                 verbose=True):
        Population.gen = 0
        # individual_generator is a function that when executed returns an individual.
        # See its use in generating the population at the end of __init__.
        Population.individual_generator = individual_generator
        Population.pop_size = pop_size
        Population.max_gens = max_gens
        Population.CXPB = CXPB
        Population.MUTPB = MUTPB
        Population.select_pct_random = select_pct_random
        Population.verbose = verbose

        self.best_ind = None
        self.former_best_ind = None

        # Choose the genetic operators.

        # Select a crossover operator. We are using our own crossover operator.
        self.toolbox.register("mate", mate, indpb=0.5)
        # self.toolbox.register("mate", mate)

        # Select a mutation operator. We are using our own mutation operator.
        self.toolbox.register("mutate", mutate, indpb=0.05)
        # self.toolbox.register("mutate", mutate)

        # operator for selecting individuals for breeding the next
        # generation: each individual of the current generation
        # is replaced by the 'fittest' (best) of <n> individuals
        # drawn randomly from the current generation.
        # I like tournament selection. Notice the difference tournament size makes.
        self.toolbox.register("select", select, tournsize=7)

        # Create a list of Individuals as the initial population.
        pop = [Population.individual_generator() for _ in range(pop_size)]
        super().__init__(pop)
        self.eval_all( )
        if verbose:
            Utils.print_stats(self)

    def eval_all(self):
        for ind in self:
            if not ind.fitness.valid:
                ind.set_fitness( )

        self.former_best_ind = self.best_ind
        self.best_ind = tools.selBest(self, 1)[0]

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
        nbr_random = round(self.pop_size * Population.select_pct_random)
        randoms = [Population.individual_generator() for _ in range(nbr_random)]
        best = self.toolbox.select(self, self.pop_size - nbr_random)
        offspring = best + randoms

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
            if random( ) < self.CXPB:
                # Set the parents to None so that we can tell whether crossover occurred
                child_1.parent_1 = child_1.parent_2 = None
                child_2.parent_1 = child_2.parent_2 = None
                # Keep track of what the elements were before crossover.
                c_1 = list(child_1)
                c_2 = list(child_2)
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

        # noinspection PyTypeChecker
        self.eval_all()
        # Elitism. Add the best of the current generation to the new generation.
        self[0] = self.best_ind

        if (self.verbose and
            # wvalues are the fitness values times the weights. MagicSquare has a negative weight.
            # A smaller fitness value the better. Multiplied by -1, a larger (but negative) fitness value is better.
            (not self.former_best_ind or self.best_ind.fitness.wvalues[0] > self.former_best_ind.fitness.wvalues[0])):
            Utils.print_stats(self)

    def run_evolution(self, fitness_target, acceptable_discrepancy=0):
        prefix = 'Unsuccessful'
        for _ in range(Population.max_gens):
            best_fit = self.best_ind.fitness.values[0]
            if abs(best_fit - fitness_target) <= acceptable_discrepancy:
                prefix = 'Successful'
                break
            self.generate_next_generation( )
        print(f"-- {prefix} evolution. {Population.gen} generations. --")


class Utils:

    # noinspection PyUnusedLocal
    @staticmethod
    # The third parameter makes this consistent with the other crossover operations
    def cx_all_diff(ind_1, ind_2, indpb=None):
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
    def list_to_array(lst, rows, cols):
        """
        Creates a side x cols 2-dimensional numpy array.
        Create a 1-D numpy array and then turn it into a side x cols array.
        """
        ar = np.reshape(np.array(lst), (rows, cols))
        return ar

    @staticmethod
    def list_to_ind(Ind_class, lst):
        ind_shell = Ind_class()
        ind_shell[:] = lst
        ind_shell.set_fitness()
        return ind_shell

    # noinspection PyUnusedLocal
    @staticmethod
    # The second parameter makes this consistent with the other crossover operations
    def mut_move_elt(ind, indpb=None):
        """
        This is our own mutation operator. It moves an element from one place to another in the list.
        """
        # Ensures that the two index positions are different.
        [indx_1, indx_2] = sample(list(range(len(ind))), 2)
        ind.insert(indx_2, ind.pop(indx_1))

    # noinspection PyUnusedLocal
    @staticmethod
    # The second parameter makes this consistent with the other crossover operations
    def mut_swap_pairs(ind, indpb=None):
        """
        This is our own mutation operator. It swaps twp pairs.
        """
        # Ensures that the two index positions are different.
        [pair_1, pair_2] = sample([1, 3, 5, 7], 2)
        (ind[pair_1], ind[pair_1+1]),    (ind[pair_2], ind[pair_2+1]) = \
        (ind[pair_2], ind[pair_2+1]),    (ind[pair_1], ind[pair_1+1])

    # noinspection PyPep8Naming
    @staticmethod
    def print_best(best_ind):
        Ind_class = best_ind.__class__
        cx_segment = f'{Utils.list_to_ind(Ind_class, best_ind.parent_1)}\n\n + (crossed with) \n' \
                     f'{Utils.list_to_ind(Ind_class, best_ind.parent_2)}\n\n =>\n ' if best_ind.parent_1 else ''
        mutation_segment = f'{Utils.list_to_ind(Ind_class, best_ind.pre_mutation)}\n\n => (mutated to) \n' \
                                                                    if best_ind.pre_mutation else ''
        print(cx_segment + mutation_segment + str(best_ind))

    @staticmethod
    def print_stats(pop):
        print(f"\n-- Generation {pop.gen} --")
        # Gather all the fitnesses in one list and print the stats.
        # Again, ind.fitness.values is a tuple. We want the first value.
        fits = [ind.fitness.values[0] for ind in pop]
        mean = sum(fits) / pop.pop_size
        sum_sq = sum(x * x for x in fits)
        std = abs(sum_sq / pop.pop_size - mean ** 2) ** 0.5
        print(f"   Min: {int(min(fits))}; Mean {round(mean, 2)}; Max {int(max(fits))}; Std {round(std, 2)}")
        Utils.print_best(pop.best_ind)

    @staticmethod
    def rotate_by(lst, amt):
        return lst[amt:] + lst[:amt]
