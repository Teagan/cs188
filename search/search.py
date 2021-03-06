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
    "*** YOUR CODE HERE ***"
    start = problem.getStartState() #(5,5)
    s = util.Stack()
    closed = set()


    starting_moves = problem.getSuccessors(start) #[((5,4), 'South', 1), ((4,5), 'West', 1)]
    for move in starting_moves:
        trail = (move, [move[1]])
        s.push(trail)
    closed.add(start)

    while not s.isEmpty():
        cell_path = s.pop()
        if problem.isGoalState(cell_path[0][0]):
            return cell_path[1]
        if cell_path[0][0] not in closed:
            closed.add(cell_path[0][0])
            successors = problem.getSuccessors(cell_path[0][0])
            for move in successors:
                updated_path = list(cell_path[1])
                updated_path.append(move[1])
                trail = (move, updated_path)
                s.push(trail)

    return None

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState() #(5,5)
    q = util.Queue()
    closed = set()


    starting_moves = problem.getSuccessors(start) #[((5,4), 'South', 1), ((4,5), 'West', 1)]
    for move in starting_moves:
        trail = (move, [move[1]])
        q.push(trail)
    closed.add(start)

    while not q.isEmpty():
        cell_path = q.pop()
        if problem.isGoalState(cell_path[0][0]):
            return cell_path[1]
        if cell_path[0][0] not in closed:
            closed.add(cell_path[0][0])
            successors = problem.getSuccessors(cell_path[0][0])
            for move in successors:
                updated_path = list(cell_path[1])
                updated_path.append(move[1])
                trail = (move, updated_path)
                q.push(trail)

    return None

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState() #(5,5)
    pq = util.PriorityQueue()
    closed = set()


    starting_moves = problem.getSuccessors(start) #[((5,4), 'South', 1), ((4,5), 'West', 1)]
    for move in starting_moves:
        trail = (move, [move[1]])
        pq.push(trail, problem.getCostOfActions([move[1]]))
    closed.add(start)

    while not pq.isEmpty():
        cell_path = pq.pop()
        if problem.isGoalState(cell_path[0][0]):
            return cell_path[1]
        if cell_path[0][0] not in closed:
            closed.add(cell_path[0][0])
            successors = problem.getSuccessors(cell_path[0][0])
            for move in successors:
                updated_path = list(cell_path[1])
                updated_path.append(move[1])
                trail = (move, updated_path)
                pq.push(trail, problem.getCostOfActions(updated_path))

    return None

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState() #(5,5)
    pq = util.PriorityQueue()
    closed = set()


    starting_moves = problem.getSuccessors(start) #[((5,4), 'South', 1), ((4,5), 'West', 1)]
    for move in starting_moves:
        trail = (move, [move[1]])
        h_plus_c = heuristic(move[0], problem) + problem.getCostOfActions([move[1]])
        pq.push(trail, h_plus_c)
    closed.add(start)

    while not pq.isEmpty():
        cell_path = pq.pop()
        if problem.isGoalState(cell_path[0][0]):
            return cell_path[1]
        if cell_path[0][0] not in closed:
            closed.add(cell_path[0][0])
            successors = problem.getSuccessors(cell_path[0][0])
            for move in successors:
                updated_path = list(cell_path[1])
                updated_path.append(move[1])

                trail = (move, updated_path)
                h_plus_c = heuristic(move[0], problem) + problem.getCostOfActions(updated_path)
                pq.push(trail, h_plus_c)

    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
