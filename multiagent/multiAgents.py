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


from heapq import heapify
from util import manhattanDistance, matrixAsList
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

    def __init__(self):
        self.last_moves = []


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
        get_score_weight = 1
        food_dist_weight = 10
        closest_food_weight = 20
        ghost_dist_weight = 1
        weak_ghost_weight = 100
        revisit_weight = 20



        revisit_penalty = 1 if (newPos in self.last_moves) else 0

        self.last_moves.append(newPos)
        if len(self.last_moves) > 10: # number of moves it keeps track of
            self.last_moves = self.last_moves[1:len(self.last_moves)]


        # edible_ghosts = sum(newScaredTimes)
        # ghost_dist_weight = -ghost_dist_weight if edible_ghosts > 0 else ghost_dist_weight


        food_dist = [[None for j in range(newFood.width)] for i in range(newFood.height)]
        binary = lambda x: 1 if x else 0
        sum_food_dist = 0
        closest_food_dist = newFood.width * newFood.height

        for row in range(newFood.width):
            for col in range(newFood.height):
                val = binary(newFood[row][col])*manhattanDistance((row,col), newPos)
                sum_food_dist += val
                closest_food_dist = val if ((val < closest_food_dist) and (val != 0)) else closest_food_dist
                food_dist[col][row] = val

        sum_food_dist = 1 if (sum_food_dist == 0) else sum_food_dist



        ghost_distances = []

        for ghost in newGhostStates:
            ghost_distances.append(manhattanDistance(ghost.configuration.pos, newPos))
        ghost_dist = sum(ghost_distances)


        # add increasing penalty if the pacman doesn't move
        # stuck_penalty = manhattanDistance(newPos, currentGameState.pos )


        # add increasing penalty if pacman doesnt get food 
        # (include factor that accounts for closest food distance)


        # add big bonus to decreasing dist between ghost and pacman when ghost is vulnerable


        score = successorGameState.getScore() * get_score_weight \
            + 1/sum_food_dist * food_dist_weight \
            + 1/closest_food_dist * closest_food_weight \
            + ghost_dist * ghost_dist_weight \
            - revisit_penalty * revisit_weight \

        # print("Current evaluation function  :   ", score)
        # print("Score             :  ", successorGameState.getScore())
        # print("Total food dist   :  1/", sum_food_dist)
        # print("Closest food dist :  1/", closest_food_dist)
        # print("Ghost distances   :  -", ghost_dist)
        # print("Revisited?        :  -", revisit_penalty)

        return score

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

    def max_value(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState)
        max_val = -float('inf')
        max_action = None
        for action in gameState.getLegalActions(0):
            for ghost_index in range(1, gameState.getNumAgents()):
                (new_action, new_val) = self.min_value(ghost_index, gameState.generateSuccessor(ghost_index, action), depth)
                if new_val > max_val:
                    max_val = new_val
                    max_action = new_action
        return (max_val, max_action)


    def min_value(self, ghost_index, gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState)
        min_val = float('inf')
        min_action = None
        for action in gameState.getLegalActions(ghost_index): 
            (new_action, new_val) = self.min_value(ghost_index, gameState.generateSuccessor(ghost_index, action), depth)
            if new_val < min_val:
                min_val = new_val
                min_action = new_action
        return (min_val, min_action)


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

        # first is maximization of pacman
        # then go through one layer of minimization per ghost
        # repeat this self.depth times or until gamestate.iswin() or gamestate.islose()
        # float('inf') is the biggest number in python
        # if there are no legal actions for an agent on a certain board state of your minimax 
        # tree you should return evaluationFunction() called on the board state corresponding 
        # to that node. I was returning 0 initially and that turned out to cause issues.
        # when calculating the value of a leaf node, use evaluationFunction()


        #adjust depending on desired depth
        return self.max_value(gameState, self.depth)





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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
