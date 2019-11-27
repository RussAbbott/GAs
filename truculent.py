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


from deap import base

from Population_and_Utils import Population, Utils

from random import sample

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

    def __str__(self):
        return f'\n   [{", ".join(map(str, self))}] -> [{", ".join(self.best_positions)}], {self.fitness.values[0]}'

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


def main(verbose=True):

    # Create an initial population. We are using a *tiny* population so that
    # the answer doesn't appear immediately. Even so the initial population
    # often contains a solution--no evolution required.
    pop = Population(pop_size=3, max_gens=50,
                     individual_generator=Individual,
                     mate=Utils.cx_all_diff, CXPB=0.5,
                     mutate=Utils.mut_move_elt, MUTPB=0.2,
                     verbose=verbose)
    # Evaluate the fitness of all population elements and return the best.
    pop.eval_all()
    if verbose:
        Utils.print_stats(pop)  #, best_ind)
        # Utils.print_best(best_ind, 'initial ')
        # print("\nStart evolution")

    # We have succeeded when the fitness shows that we can place 7 tokens.
    # Generate new populations until then or until we reach max_gens.
    success = 'unsuccessful'
    for _ in range(Population.max_gens):
        best_fit = pop.best_ind.fitness.values[0]
        if best_fit == 7:
            success = 'successful'
            break
        pop.generate_next_generation()

    print(f"\n-- After {Population.gen} generations, end of {success} evolution --")


if __name__ == "__main__":
    main()
