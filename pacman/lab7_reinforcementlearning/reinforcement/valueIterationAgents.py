# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # Initializing our states
        # The fact that I'm not sure if we can initialize a self.states makes
        # this kind of annoying :(
        states = self.mdp.getStates()

        # We are going to loop as many times as we are told based on the
        # iterations variable
        for i in range(0,self.iterations):

            # We need a holder for our new values that we are going to replace
            # the originals with, so we are initializing this dict
            data = util.Counter()

            # We're going to iterate through each state on the board and then
            # evaluate the qvalue for those states
            for state in states:

                # temp is going to be a holder for the variable we insert into
                # the dict we initialized earlier
                temp = None

                # We're gonna get all of our possible actions at our current
                # position and calculate the qvalue at said position
                actions = self.mdp.getPossibleActions(state)
                for action in actions:

                    # We're gonna use curr to hold the qvalue return. We will
                    # change temp accordingly if curr yields a new max
                    curr = self.computeQValueFromValues(state,action)
                    if temp == None or temp < curr:
                        temp = curr

                # If our temp value hasn't changed, we haven't found a
                # noteworthy max, so we'll return 0
                if temp == None:
                    temp = 0

                # Update our dict to retain the information we just found for
                # the current state
                data[state] = temp

            # Change our values to reflect our calculations of the new state
            # dict
            self.values = data


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # This one was fun! The simplicity made me feel confident :D
        # Okay, initializing our return variable at 0, standard stuff
        qval = 0

        # We are given a state and an action. We want to get the possible states
        # sprime that would result from that action, as well as their
        # probabilities
        stateprobs = self.mdp.getTransitionStatesAndProbs(state,action)

        # We want to iterate through each possible state sprime and the
        # associated probability
        for sprime, prob in stateprobs:

            # We want to calculate the rewards from the move it would take to
            # get from state to sprime
            r = self.mdp.getReward(state, action, sprime)

            # In accordance with the qvalue function, we want to construct the
            # current qval using the qvals of each possible sprime
            qval += prob * (r + (self.discount * self.values[sprime]))

        # Return 8)
        return qval

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Imagine an italian accent
        # Okay so we're gonna do some backwards of the working capiche?

        # Getting the possible actions from our current state; if there are none
        # then no actions could possible be associated with any inputted state
        # or intended value
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            return None

        # We make a holder for the qval at the given state as well as for the
        # action that we will return
        qval = float('-inf')
        maxact = None

        # We're gonna iterate through each action and see if it returns the max
        # qval
        for action in actions:

            # If it does indeed return the max qval, let's update our qval and
            # returns accordingly
            if self.computeQValueFromValues(state, action) > qval:
                qval = self.computeQValueFromValues(state, action)
                maxact = action

        # By this point, we will have found the action associated with the max
        # qval, which is the one we want, so let's return it
        return maxact

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
