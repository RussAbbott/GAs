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

"""
In the puzzle truculent: https://www.transum.org/Software/Puzzles/Truculent.asp
the 8 nodes are connected in a single cycle. Numbering clockwise starting
at the top right, the cycle is: 1, 4, 7, 2, 5, 8, 3, 6.

We can think of this as a simple list of length 8. From this point on, we can
talk in terms of such a list and not talk about the actual nodes.

A solution to the puzzle will be a list of length 7, called solution, of some
permutation of 7 distinct elements of 1..8.

solution[i] represents the position where one (a) puts the token at step i and then
(b) slides it to the left. (solution is of length 7 since we only have to put down 7 tokens.)

Here is some pseudo-code that is fleshed out in Individual.set_fitness.

for i in range(8):
  if positions[solution[i]] is empty and positions[solution[i]+1] is empty:
    # put a counter at positions[solution[i]] and slide it to the right (mod 8)
    put a counter at positions[solution[i]]
  else:
    break

The fitness of the given solution is the number of tokens successfully put into the positions.

"""


from deap import base, tools

from random import choice, random, sample

"""
Typically we need an Individual class and a Population class.
An individual is a list of "genes". These are what get mutated.
"Genes" are typically primitive elements like ints or bools. In Truculent,
the "genes" are indexes into the positions list.

The Population is a list of individuals. 
"""


class Individual(list):
    positions = 8
    indices = list(range(positions))

    def __init__(self):
        # DEAP stores fitness values in a Fitness class, which offers some basic operations. (See base.py.)
        # Must set the "weights" before instantiating the Fitness class.
        # The weights which must be a tuple. We use it to keep track of
        # the actual fitness along with the rotation and the ending positions.
        base.Fitness.weights = (1, )
        self.fitness = base.Fitness()

        # Define these here to keep PyCharm happy
        self.best_positions = None
        self.parent_1 = self.parent_2 = self.pre_mutation = None

        # The class Individual is a subclass of list. So must have a list.
        # Every candidate solution will be a permutation of range(8).
        the_list = sample(Individual.indices, Individual.positions)
        super().__init__(the_list)

    def set_fitness(self):
        """
        Compute the fitness of this individual and store it at self.fitness.values.
        """
        original_list = list(self)
        self.best_positions = ['_'] * 8
        # To keep PyCharm happy
        positions = None
        # Try starting at all positions in our list and wrapping around.
        for i in Individual.indices:
            # Try all rotations of self
            rotation = Utils.rotate_by(original_list, i)
            positions = ['_'] * 8

            for j in range(len(rotation)):
                indx = rotation[j]
                # Put a token at positions[indx] if it and the position
                # either to its left or to its right are unoccupied.
                # No need to worry about indx-1 being -1 since that is equivalent to index 7.
                if positions[indx] == '_' and (positions[indx - 1] == '_' or positions[(indx + 1) % 8] == '_'):
                    positions[indx] = 'abcdefg'[j]
                else:
                    # If we can't place any more tokens we have reached the fitness of this individual.
                    break

            # Determine which rotation is best. We use this later to display the results.
            if (sum(positions[j] != '_' for j in range(len(positions)))
                > sum(self.best_positions[j] != '_' for j in range(len(self.best_positions)))):
                # Replace this individual with its best rotation and save the best positions record.
                self[:] = rotation
                self.best_positions = positions

        self.fitness.values = (sum([self.best_positions[j] != '_' for j in range(len(positions))]), )


class Population(list):
    """
    The Population holds the collection of Individuals that will undergo evolution.
    """

    # Which generation this is.
    gen = 0

    # An expressions that when executed generates an individual.
    individual_generator = None

    # The size of the population
    pop_size = None

    # The maximum number of generations to allow.
    max_gens = None

    # The probabilities of performing crossover and mutation respectively.
    CXPB = None
    MUTPB = None

    # Whether to print intermediate output. (The default is True.)
    verbose = None

    # A reference to the DEAP toolbox.
    toolbox = base.Toolbox( )

    def __init__(self, pop_size, max_gens, individual_generator, CXPB=0.5, MUTPB=0.2, verbose=True):
        # individual_generator is a function that when executed returns an individual.
        # See its use in generating the population at the end of __init__.
        Population.individual_generator = individual_generator

        Population.pop_size = pop_size
        Population.max_gens = max_gens
        Population.CXPB = CXPB
        Population.MUTPB = MUTPB
        Population.verbose = verbose

        # Choose the genetic operators.

        # In the following we are using the DEAP mechanism that allows us to name operators
        # in the toolbox. It's a bit hokey, but it's a major aspect of how DEAP works.

        # Select a crossover operator. We are using our own crossover operator defined below.
        # We could use a built-in operator.
        self.toolbox.register("mate", Utils.cx_all_diff)

        # Select a mutation operator. Again, we are using our own mutation operator.
        self.toolbox.register("mutate", Utils.mut_move_elt)

        # Select the operator for selecting individuals for the next generation.
        # In Tournament selection each individual of the next generation is selected
        # as the 'fittest' of <n> individuals drawn randomly from the current generation.
        # A larger <n> results in a more elitist selection process.
        # A tournament size of 2 will repeatedly select two random elements of the current
        # population put the most fit into the next population.
        self.toolbox.register("select", tools.selTournament, tournsize=2)

        # Create a list of Individuals as the initial population.
        pop = [Population.individual_generator() for _ in range(pop_size)]
        super().__init__(pop)

    def eval_all(self):
        """
        Evaluate all elements in the population and return (one of) the fittest.
        """
        count = 0
        for ind in self:
            if not ind.fitness.valid:
                ind.set_fitness( )
                count += 1

        if self.verbose:
            print(f"   Evaluated {count} individuals")

        best_ind = tools.selBest(self, 1)[0]
        return best_ind

    def generate_next_generation(self):
        """
        Generate the next generation. It is not returned; it replaces the current generation.
        """
        Population.gen += 1
        if self.verbose:
            print(f"\n-- Generation {Population.gen}/{Population.max_gens} --")

        # Select the next generation of individuals to use as the starting point.
        # Use tournament selection to select the base population.
        # Since these are the elite elements of the population and since pop-size
        # are selected we will almost certainly include duplicates.
        # But there is no guarantee that the best is kept.
        offspring = self.toolbox.select(self, self.pop_size)

        # Now make each element a clone of itself.
        # We do this because the genetic operators modify the elements in place.
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

                # fitness values of the children must be recalculated later
                del child_1.fitness.values
                del child_2.fitness.values

        for mutant in offspring:
            mutant.pre_mutation = None
            # mutate an individual with probability MUTPB
            if random( ) < self.MUTPB:
                mutant.pre_mutation = list(mutant)
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

        # The new generation replaces the current one.
        # Recall that self is the population.
        self[:] = offspring

        # Evaluate all individuals -- but don't re-evaluate those that haven't changed.
        # Return the best individual.
        best_ind = self.eval_all()

        if self.verbose:
            Utils.print_stats(self, best_ind)

        return best_ind

class Utils:

    @staticmethod
    def cx_all_diff(ind_1, ind_2):
        """
        Perform crossover between ind_1 and ind_2 without violating all_different.
        This is our own crossover operator.
        """
        ind_1_new = Utils.cx_over(ind_1, ind_2)
        ind_2[:] = Utils.cx_over(ind_2, ind_1)
        ind_1[:] = ind_1_new

    @staticmethod
    def cx_over(ind_1, ind_2):

        inner_indices = [2, 3, 4, 5]
        ind_1r = Utils.rotate_by(ind_1, choice(inner_indices))
        ind_2r = Utils.rotate_by(ind_2, choice(inner_indices))
        indx = choice(inner_indices)

        child = ind_1r[: indx] + [item for item in ind_2r if item not in ind_1r[: indx]]
        return child

    @staticmethod
    def mut_move_elt(ind):
        """
        This is our own mutation operator. It moves an element from one place to another in the list.
        """
        # Ensures that the two index positions are different.
        [indx_1, indx_2] = sample(Individual.indices, 2)
        ind.insert(indx_2, ind.pop(indx_1))

    @staticmethod
    def print_best(best_ind, label=''):
        cx_segment = f'{best_ind.parent_1} + {best_ind.parent_2} => ' if best_ind.parent_1 else ''
        mutation_segment = f'{best_ind.pre_mutation} => ' if best_ind.pre_mutation else ''
        print('   ' + cx_segment + mutation_segment + str(best_ind))
        print(f'   Best {label}individual: {best_ind}, '
              f'[{", ".join(best_ind.best_positions)}], '
              f'{best_ind.fitness.values[0]}'
              )

    @staticmethod
    def print_stats(pop, best_ind):
        # Gather all the fitnesses in a list and print the stats.
        # Again, ind.fitness.values is a tuple. We want the first value.
        fits = [ind.fitness.values[0] for ind in pop]
        mean = sum(fits) / pop.pop_size
        sum_sq = sum(x * x for x in fits)
        std = abs(sum_sq / pop.pop_size - mean ** 2) ** 0.5
        print(f"   Min: {min(fits)}; Mean {round(mean, 2)}; Max {max(fits)}; Std {round(std, 2)}")
        Utils.print_best(best_ind)


    @staticmethod
    def rotate_by(lst, amt):
        return lst[amt:] + lst[:amt]


def main(verbose=True):

    # Create an initial population. We are using a *tiny* population so that
    # the answer doesn't appear immediately. Even so the initial population
    # often contains a solution--no evolution required.
    pop = Population(pop_size=3, max_gens=50,
                     individual_generator=Individual,
                     verbose=verbose)
    # Evaluate the fitness of all population elements and return the best.
    best_ind = pop.eval_all()
    if verbose:
        print()
        Utils.print_best(best_ind, 'initial ')
        print("\nStart evolution")

    # We have succeeded when the fitness shows that we can place 7 tokens.
    # Generate new populations until then or until we reach max_gens.
    success = 'unsuccessful'
    for _ in range(Population.max_gens):
        best_fit = best_ind.fitness.values[0]
        if best_fit == 7:
            success = 'successful'
            break
        best_ind = pop.generate_next_generation()

    print(f"\n-- After {Population.gen} generations, end of {success} evolution --")


if __name__ == "__main__":
    main()
