# a Gazi Rahman original

# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # We wan't to set a baseline for whether or not we should make the
        # move that we are processing
        makeMove = True

        # Let's figure out if we die, in which case we don't wanna make the
        # move

        # I'm not factoring the ghost positions into my score because a) that
        # sounds hard to consider the risk based on distance b) some moves are
        # incredibly lucrative, even if it means getting super close to the
        # ghost. figuring out whether a move is lucrative based on the nuances
        # also sounds really hard

        # For some reason, my newPos doesn't work unless it's a tuple
        for ghost in newGhostStates:
            gpos = ghost.getPosition()
            if gpos == (newPos):
                makeMove = False

        # I tried working off of newFood.asList(), but it takes a lot of time
        # for the code to calculate the newFood list for each successor, and
        # then evaluate the points in each respective newFood list, in contrast
        # to just working off of the current board
        foodPos = currentGameState.getFood().asList()

        # Basically, our evaluation function will work off of the food distance.
        # We'll see how far the closest food is for this successor, and then
        # we'll assign points 1:1 accordingly.
        dists = []
        for food in foodPos:
            dist = manhattanDistance(food, (newPos))
            dists.append(dist)

        # If we get eaten, then we won't even consider the move. We assign a
        # really low value in consideration of the minimax process, assigning
        # a very low min here will get rid of it as our agent maximizes it's
        # selection for a viable move
        if makeMove == False:
            return -10000

        # We want to return negative value in anticipation of the minimax
        # shenanigans
        return -min(dists)



def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # Importing infinity from math so that we can do miniMax like we did
        # in the slides
        from math import inf

        # We'll use miniMax to judge our maximizing depths, which make decisions
        # for pacman
        def miniMax(gameState, depth):

            # If we win or lose, the game is over, so let's end our recursion
            if gameState.isWin() or gameState.isLose():

                # We'll return our evaluation function, which judges our current
                # state, and the move we should make, which is no move
                return(self.evaluationFunction(gameState), None)

            # If we reach the end of our game tree, then let's end our recursion
            if depth == self.depth:

                # Same as last time lol
                return(self.evaluationFunction(gameState),None)

            # Let's set our max to infinity so that any possible move is greater
            tempMax = -inf

            # We need to get all the possible moves for pacman, whose index is 0
            actions = gameState.getLegalActions(0)

            # We're gonna iterate through each action, which functions as a
            # a branch. As we move through each branch, we will collect data
            # on the values they return
            for action in actions:

                # Let's create the branch
                branch = gameState.generateSuccessor(0, action)

                # Let's take the miniMin from that branch. We need the miniMin
                # since the branch is moving on to the next depth, which
                # requires minimization
                branch = miniMin(branch, depth, 1)

                # If we have a new max, let's adapt accordingly
                if branch[0] > tempMax:
                    tempMax, move = branch[0], action

            # At this point, the data we've stored corresponds to the optimal
            # moves so lets return it accordingly
            return(tempMax, move)

        # We'll use miniMin for minimizing depths, which pick moves for the
        # ghosts
        def miniMin(gameState, depth, agentIndex):

            # Let's get a list of the possible actions, this time for the
            # ghost that we are considering
            actions = gameState.getLegalActions(agentIndex)

            # If our ghost can't move then we have to end the recursion
            # Didn't know this was possible but that's what the TA is saying lol
            if len(actions) == 0:

                # Again, return our score for the current state and the lack
                # of the move we should make lol
                return(self.evaluationFunction(gameState), None)

            # This time tempMin holds our min value, set to infinity
            tempMin = inf

            # We'll iterate through each action, this time a little differently
            for action in actions:

                # Again, let's consider the branch that comes from the action
                branch = gameState.generateSuccessor(agentIndex, action)

                # If we approach the final agentIndex, then we are running out
                # of ghosts, which means that we can move on to the next depth.
                # If we move on to the next depth, then we maximize
                if agentIndex == gameState.getNumAgents() - 1:
                    branch = miniMax(branch, depth + 1)

                # So long as we don't reach the end of this depth, we want to
                # keep running miniMin
                else:
                    branch = miniMin(branch, depth, agentIndex+1)

                # Let's check if we have a new min and proceed accordingly
                if branch[0] < tempMin:
                    tempMin, move = branch[0], action

            # At this point we have our min and the move that leads to it so
            # let's return
            return (tempMin, move)

        # We essentially start at the top of the game tree and our recursion
        # flips between miniMin and miniMax, working down to the bottom
        # and then once it gets information for the whole tree, works back up
        # until we can return the appropriate action
        return miniMax(gameState, 0)[1]





class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Importing infinity
        from math import inf

        # The structure for miniMax is the exact same, it will just adapt to
        # expectiPredict, which predicts the values of branches and leaves
        def expectiMax(gameState, depth):

            # If we win or lose, the game is over, so let's end our recursion
            if gameState.isWin() or gameState.isLose():

                # We'll return our evaluation function, which judges our current
                # state, and the move we should make, which is no move
                return(self.evaluationFunction(gameState), None)

            # If we reach the end of our game tree, then let's end our recursion
            if depth == self.depth:

                # Same as last time lol
                return(self.evaluationFunction(gameState),None)

            # Let's set our max to infinity so that any possible move is greater
            tempMax = -inf

            # We need to get all the possible moves for pacman, whose index is 0
            actions = gameState.getLegalActions(0)

            # We're gonna iterate through each action, which functions as a
            # a branch. As we move through each branch, we will collect data
            # on the values they return
            for action in actions:

                # Let's create the branch
                branch = gameState.generateSuccessor(0, action)

                # Let's take expectiPredict from that branch
                branch = expectiPredict(branch, depth, 1)

                # If we have a new max, let's adapt accordingly
                if branch[0] > tempMax:
                    tempMax, move = branch[0], action

            # At this point, the data we've stored corresponds to the optimal
            # moves so lets return it accordingly
            return(tempMax, move)

        # This time instead of finding and then evaluating minimum values,
        # We will create expected values at the minimizing depths of the game
        # tree

        # Also this name is so annoying lmaooo, I love it and am keeping it
        def expectiPredict(gameState, depth, agentIndex):

            # Same as miniMin, gotta get the actions early so that we can see
            # if the ghost can make no actions, because that's apparently
            # possible lol
            actions = gameState.getLegalActions(agentIndex)

            # If the ghost can't act then we return the evaluation function and
            # the lack of action lmao
            if len(actions) == 0:
                return (self.evaluationFunction(gameState), None)

            # Let's create a holder for our predicted value, which we will
            # build up from 0 with weighted probability calculations
            tempGuess = 0

            # We are going to iterate through each action, each of which
            # represents a branch in our game tree, and we are going to create
            # our value prediction weighted by each branch
            for action in actions:

                # Again, let's make the branch
                branch = gameState.generateSuccessor(agentIndex, action)

                # Similar to last time, if we reach the last ghost then let's
                # move onto the next depth, where we are now maximizing
                if agentIndex == gameState.getNumAgents() - 1:
                    branch = expectiMax(branch, depth + 1)

                # If not, then let's keep calculating until we can round out our
                # prediction
                else:
                    branch = expectiPredict(branch, depth, agentIndex + 1)

                # Now that we have the value given by said branch, we can divide
                # by the number of actions to give it a weighted value.
                # Then we can add this weighted value to our guess to create
                # expected values for each branch as we alternate between
                # predictions and maximizing depths
                tempGuess += branch[0] / len(actions)

            # Let's return our guess. With expectiPredict we don't need to
            # return a move because the move is determined by expectiMax,
            # but we should return a None for uniformity in return types
            return (tempGuess,None)

        # We start at the top of our game tree using expectiMax, which will
        # automatically switch between expectiPredict and expectiMax with
        # each level of recursion until we fill out the info in our tree,
        # after which we will make judgements using the info as we move back
        # up the tree.
        return expectiMax(gameState,0)[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # Let's get some basic info real quick with some cool, sleek, quick maths
    pos = currentGameState.getPacmanPosition()
    boostList = currentGameState.getCapsules()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()

    # Let's give credit for how much the score already is
    score = currentGameState.getScore()

    # We should offer credit for how many capsules we've consumed, but we
    # really don't want to incentivize it too much; we don't want pacman
    # making a risk to get the capsule and end up dying in the process
    score -= 10 * len(boostList)

    # We want to give a lot of credit for eating food since that is our main
    # way for us to win right now. By subtracting points for how much food is
    # left, we incentivize the consumption of food so there's less food left
    # to cut from our score
    score -= 20 * len(foodList)

    # It's possible we ate every food without eating every pellet apparently?
    # Crazy world.
    if len(foodList) > 0:

        # Creating a holder for the distances of each food pellet
        foodDists = []

        # We will iterate through each pellet to determine which one is the
        # closest
        for food in foodList:
            dist = manhattanDistance(pos, food)
            foodDists.append(dist)
        minDist = min(foodDists)

        # We want to incentivize food consumption but not too much. In other
        # words, pacman keeps dying if I don't make minDist the denominator lol
        score += 1.0/minDist

    # We're checking through each ghost, and seeing whether the ghost is scared
    # or not, and then how close the ghost is
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        dist = manhattanDistance(pos,ghostPos)

        # If the ghost is scared, then we want to consider eating it
        if ghost.scaredTimer:

            # However, if the scared ghost is too far away, it might become
            # active again by the time we reach it, so we really should only
            # incentivize consumption within a certain distance. Otherwise,
            # let's not think about it too much
            if dist < 5:

                # If the ghost is right next to us, the odds are that we could
                # consume it without much risk so let's just do that >:^D
                if dist < 2:
                    score += 300

                # If the ghost is between the 2-4 distance units threshold, then
                # we really don't want to incentivize the risk too much :(
                score += 30

        # If the ghost isn't scared, let's still pay attention to it
        else:

            # If the ghost is active and close, RUN!!!!
            if dist < 2:

                # Let's incentivize staying away from the spooky wooky :(
                score -= 300

    # Return the *better* evaluation function >:^)
    # Now that I'm really looking at it I can't believe my OG evaluation
    # function even working lmao
    return score

# Abbreviation
better = betterEvaluationFunction
