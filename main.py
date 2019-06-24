import os.path
import pickle

import numpy as np
from sklearn import preprocessing

FILEPATH = 'C:\\Users\\rober\\Desktop\\text-generator-markov-master\\file.txt'
STATESPATH = 'C:\\Users\\rober\\Desktop\\text-generator-markov-master\\states.txt'
SAVEFILEPATH = 'C:\\Users\\rober\\Desktop\\text-generator-markov-master\\matrix.txt'
GENERATEDSTUFF = 'C:\\Users\\rober\\Desktop\\text-generator-markov-master\\generatedstuff.txt'


class MarkovThing:
    states = []
    states_dict = {}
    transition_matrix = None

    def prepare_data(self):
        # method for iterating all data and making
        # mapped dictionary
        if not os.path.exists(STATESPATH):
            print("States array not found. Generating...")
            if os.path.exists(FILEPATH):
                with open(FILEPATH) as f:
                    words = f.read()
                    #for i in f.read():
                    for i in words.split():
                        if i not in self.states:
                            self.states.append(i)
            else:
                print("You have to put your data in file.txt!")
                exit()
            #self.states.sort()
            with open(STATESPATH, 'wb') as f:
                pickle.dump(self.states, f)
                print("Saving states array to a file...")
        else:
            print("Found states array file. Loading...")
            with open(STATESPATH, 'rb') as f:
                self.states = pickle.load(f)
                print("Done.")

        # all states mapped state : position
        print("Creating states dictionary...")
        for i in range(len(self.states)):
            self.states_dict[self.states[i]] = i
        print("Done.")

    def update_transition_matrix(self):
        # transition matrix updater:
        # s[f][t] a | b | c ("from"/current state)
        #     a | 1 | 2 | 3
        #     b | 4 | 5 | 6
        #     c | 7 | 8 | 9
        # ("to"/next state)
        # where a, b, c - states
        # numbers - probabilities
        # 1. find index of current state in the dict
        # 2. find index of next state in the dict
        # 3. update probability in transition matrix[current][next]
        # 4. repeat for all length of a file
        # 5. save matrix to a file
        buffer = []
        if not os.path.exists(SAVEFILEPATH):
            print("Transition matrix not found. Generating...")
            self.transition_matrix = np.zeros((len(self.states), len(self.states)))
            with open(FILEPATH) as f:
                #for i in f.read():
                words = f.read()
                for i in words.split():
                    buffer.append(i)
                    if len(buffer) == 2:
                        current_state = self.states_dict.get(buffer[0])
                        next_state = self.states_dict.get(buffer[1])
                        self.transition_matrix[current_state][next_state] += 1
                        buffer.pop(0)
                print("Done.")

            # part responsible for change matrix from occurrences to actual probabilities
            print("Normalizing transition matrix...")
            self.transition_matrix = preprocessing.normalize(self.transition_matrix, norm='l1')

            print("Saving transition matrix to a file...")
            np.savetxt(SAVEFILEPATH, self.transition_matrix, delimiter=', ')
            print("Done.")
        else:
            print("Loading transition matrix from file...")
            self.transition_matrix = np.loadtxt(SAVEFILEPATH, delimiter=', ')
            print("Done.")

    def next_state(self, current_state):
        # method for choosing the next state from the transition matrix
        return np.random.choice(
            self.states,
            p=self.transition_matrix[self.states_dict[current_state], :]
        )

    def generate_states(self, current_state, number):
        # method for generating new data
        print("Generating new states...")
        generated = []
        for i in range(number):
            n_state = self.next_state(current_state)
            generated.append(n_state)
            current_state = n_state
        print("Done.")

        with open(GENERATEDSTUFF, 'w') as f:
            print("Saving generated states to a file...")
            f.write(' '.join(generated))
            print("Done.")


def main():
    x = MarkovThing()

    x.prepare_data()
    x.update_transition_matrix()
    x.generate_states(current_state=x.states[0], number=10000)


if __name__ == '__main__':
    main()
