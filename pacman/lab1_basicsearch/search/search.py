# a gazi rahman original production 8)
# what the code look like if we made pacman explore every dead end and wrong
# moves before finally getting to the goal o.O

# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # "*** YOUR CODE HERE ***"

    # creating stack, lists to track spots that it's already been at and
    # how it has moved so far to implement graph search
    stack = util.Stack()
    closed =[]
    moves = []

    # adding start state to the stack
    stack.push((problem.getStartState(),[]))

    # figure the search should stop and give up if the stack is empty and it
    # hasn't gotten anywhere lol
    while not stack.isEmpty():

        # assigning variables to the current state, making sure we don't revisit
        # any places we've already been by adding position to closed list
        pos,moves = stack.pop()
        closed.append(pos)

        # self explanatory, but if we have gotten to where we need to go then
        # we can just state how we got there
        if problem.isGoalState(pos):
            return moves

        # figuring out next move while making sure we don't revisit old spots
        next = problem.getSuccessors(pos)
        for node in next:
            if node[0] not in closed:
                new = moves + [node[1]]
                stack.push((node[0],new))

    # can i just say how proud i am to have gotten this, i was stuck for two
    # hours because i forgot to add the start state to the stack

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # same as dfs except using a queue instead of a stack
    # need the set q because a Queue() is not iterable so we need a set version
    # of it where we can track the frontier and check for the successors in
    queue = util.Queue()
    q = set()
    closed = set()
    moves = []

    # add start state to the queue
    queue.push((problem.getStartState(),[]))

    # same logic as dfs, figure if our queue somehow becomes empty we should
    # forfeit the agent and figure out what happened
    while not queue.isEmpty():

        # assign variables to our state so we can change and work with them
        pos,moves = queue.pop()
        closed.add(pos)

        # if we're done then let's return our moves
        if problem.isGoalState(pos):
            return moves

        # if we're not done, let's figure out what to do next and make sure
        # not to go where we've already been
        next = problem.getSuccessors(pos)
        for node in next:
            if node[0] not in closed and node[0] not in q:
                new = moves + [node[1]]
                queue.push((node[0],new))
                q.add(node[0])

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
