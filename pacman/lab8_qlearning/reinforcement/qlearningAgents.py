# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # Figure I'll initialize the states to hold qvals like we did in the
        # valueIterationAgents.py
        self.qvalues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # If we have seen this state and action pair, then let's return the
        # appropriate q value
        if (state,action) in self.qvalues:
            return self.qvalues[(state,action)]

        # Otherwise, let's return 0.0
        else:
            return 0.0


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # Let's get the available actions at the current state, and then we can
        # loop through (s,a) for a in actions to find the correct pair
        actions = self.getLegalActions(state)

        # If we have no available actions, we must be at a terminal state, so
        # we'll return 0.0
        if len(actions) == 0:
            return 0.0

        # If we're not at a terminal state, the value may be something other
        # than 0.0, so let's figure it out and return it
        else:

            # We'll create a holder of all the qvalues based on each action
            qvalues = []

            # We'll loop through each action and keep track of the corresponding
            # qvalue
            for action in actions:
                qvalues.append(self.getQValue(state,action))

            # In accordance with the qvalue function, let's return the max
            # qvalue
            return max(qvalues)


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # We're importing random now so that we can use it later, as you'll see
        import random

        # Let's get our available actions so we can loop through each
        # (state, action) pair again
        actions = self.getLegalActions(state)

        # If there aren't any actions available, then we've reached a terminal
        # state, so there really isn't any appropriate action that we should
        # return lol
        if len(actions) == 0:
            return None

        # If we're not at a terminal state, then there is an appropriate action
        # to return, so let's find it
        else:

            # We'll initialize a list to keep track of our (action, qvalue)
            # pairs and another to keep track of the individual qvalues.
            allpairs = []
            allqvals = []

            # Now, we'll loop through each action and append said action and
            # the attached qvalue to allpairs, and append the qvalue to the
            # allqvals list.
            for action in actions:
                allpairs.append((action, self.getQValue(state,action)))
                allqvals.append(self.getQValue(state,action))

            # Let's find what the max qval is
            maxVal = max(allqvals)

            # There's a possibility that multiple actions hold the max qval, and
            # we should leave a chance to explore each one, so we'll initialize
            # a list of the maxactions and append the appropriate pairs
            maxactions = []
            for pair in allpairs:
                if pair[1] == maxVal:
                    maxactions.append(pair)

            # We'll use our imported random package so we can make a random
            # choice among which max action to return
            return random.choice(maxactions)[0]

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        "*** YOUR CODE HERE ***"
        # We want to choose a random action for P(epsilon). We'll simulate
        # P(epsilon) using flipcoin(epsilon).
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)

        # flipcoin(epsilon) returns true for P(epsilon), so it returns false
        # the remaining amount of times. Thus, in accordance with the problem
        # prompt, we'll choose our known best action for P(not epsilon)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # We can just follow the equation established in the video lecture
        # slides

        # We use (1-alpha)*Q(s,a) to offer a lower priority to old data
        olddata = (1 - self.alpha) * self.getQValue(state, action)

        # We use alpha(reward + lambda * Q(s',a')) to offer more priority to new
        # data
        newdata = self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))

        # We sum the two to find our new qvalue
        self.qvalues[(state,action)] = olddata + newdata

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # The features make up the different metrics for our qvalue, so let's
        # get the feats and create a holder for our sum qvalue
        features = self.featExtractor.getFeatures(state, action)
        qvalue = 0

        # Haha feet but feat
        # Get the value associated with each feature, multiply by its
        # corresponding weight, and then sum them up to get our qvalue
        for feat in features:

            # Features feet but feat hahaha
            qvalue += self.weights[feat] * features[feat]

        # Okay enough jokes, serious return time 8)
        return qvalue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # Okay okay, we need da features so we can do da loop de loop
        # Just realized it is called self feet extractor. Am I the only one that
        # thinks computer scientists are low key funny?
        feats = self.featExtractor.getFeatures(state, action)

        # Okay okay, we will follow da code template in da lectures
        diff = (reward + self.discount*self.getValue(nextState)) - self.getQValue(state, action)

        # Loop tru da feats so that we can update da weight associated with
        # each feat, following da same equation given in the problem page
        for feat in feats:
            self.weights[feat] += self.alpha * diff * feats[feat]

        # I fear that with each passing line of code, I am descending into
        # madness. My writing skills devolve, I feel the need to articulate
        # myself in a manner similar to a cartoon in order to feel real. Is
        # this what I have become? Is this who I am destined to be? Joking
        # about the apparati of ambulation in an artificial intelligence
        # programming assignment? Does my air of whimsy and silly-goofy nature
        # know no bounds? I fear my hilarity has gotten beyond me, God save us
        # all. -GSR

        # Disclaimer: the above comment block was a joke born of boredom. I
        # do not really fear any descent into madness, nor do I detest my
        # silly nature. I figure the seriousness of that note would provide a
        # comedically fruitful juxtaposition to the nonsensical and sophomoric
        # jokes that I have been making in other portions of the assignment.
        # Fear not, Papa is okay. -Also GSR

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
