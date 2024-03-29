# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import math

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        #print "---"*10
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
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action) # game state after action
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        minDist = float("inf")
        minFood = []
        for food in newFood.asList():
            dist = directDistance(newPos, food)
            if dist < minDist:
                minFood = [food] # clear and add new food in the list
                minDist = dist
            elif dist == minDist:  # food at same distance
                minFood.append(food)
        newGhostStates = successorGameState.getGhostStates()
        # Calculate total ghost distance
        ghostDist = 0
        for n in newGhostStates:
            ghostDist += directDistance(n.configuration.pos, newPos)

        ## Heuristic ##
        ## Just to try different values ##
        A = 1 # How scared pacman is from ghost
        B = 1 # How attracted pacman is to food
        ###############
        if ghostDist == 0:
            finalValue = successorGameState.getScore()  + B * (1/minDist)
        else:
            finalValue = successorGameState.getScore()  - A * (1/ghostDist) + B * (1/(minDist))
        
        return finalValue

def directDistance(pointA, pointB):

    return math.sqrt(abs((pointA[0]-pointB[0])**2 + (pointA[1] - pointB[1])**2))

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
      Minimax Agent (question 2)
      We are considering total depth = depth from command line * total number of agents
    """
    # This function will be called recursively
    def maxMin(self, gameState, depth, action): # GameState, Depth, Action
        # Terminal State
        if depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        else: # Not terminal state, hence call recursion
            actions = gameState.getLegalActions( (depth % gameState.getNumAgents()) )
            if 'Stop' in actions:
                actions.remove('Stop')
            ###### PACMAN AGENT ###########
            if depth % gameState.getNumAgents() == 0: # Check if it is pacman Agent
                maximum = float("-inf")
                for action in actions:
                    succGameState =gameState.generateSuccessor(depth % gameState.getNumAgents(), action)
                    val = self.maxMin(succGameState, depth + 1, action)
                    maximum = max(maximum, val)
                return maximum
            else:
            ###### GHOST AGENT ###########
                minimum = float("inf")
                for action in actions:
                    succGameState =gameState.generateSuccessor(depth % gameState.getNumAgents(), action)
                    val = self.maxMin(succGameState, depth + 1, action)
                    minimum = min(minimum, val)
                return minimum

    def getAction(self, gameState):
        maxValue = float("-inf")
        maxAction = None
        actions = gameState.getLegalActions()
        if 'Stop' in actions:
            actions.remove('Stop')
        for action in actions:
            succGameState = gameState.generateSuccessor(0, action)
            maxMinValue = self.maxMin( succGameState, 1, action)
            if maxValue < maxMinValue:
                maxValue = maxMinValue
                maxAction = action
            #print "Action : ", action, "Height: 0", "Final Value : ", maxValue, "action:", maxAction
        #print "="*20
        return maxAction
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """ 
      Your minimax agent with alpha-beta pruning (question 3)
    """
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    '''
    Alpha Beta can only happen between min and max layers but cannot happen 
    between two min layers. Hence we have to take care of this case 
    and simply perform operations normally for it.
    '''
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        move = self.maximize(gameState, float("-inf"), float("inf"), self.depth)
        return move[1]
    
    def maximize(self, gameState, alpha, beta, d):
        # Check if we are at winning or losing point
        if d == 0 or gameState.isWin() or gameState.isLose():
          return (self.evaluationFunction(gameState), Directions.STOP)
        value = float("-inf")
        # For every legal action see get the maximum value from minimum values of all successors.
        # Keep track of maximum value and that maximizing action and return these
        # Here there is only one maximizing agent so, we need to do this only once for every legal actions
        # unlike our multiple minimizing agents
        for action in gameState.getLegalActions(0):
          tmp = max(value, self.minimize(gameState.generateSuccessor(0, action), alpha, beta, d, 1))
          if tmp > value:
            final = action
            value = tmp
          if value > beta:
            return (value, final)
          alpha = max(alpha, value)
        return (value, final)
    
    def minimize(self, gameState, alpha, beta, d, player):
        # Check if we are at winning or losing point
        if d == 0 or gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        value = float("inf")
        # For every legal action see get the minimum value from maximum values of all successors.
        # Keep track of minimum value and that minimizing action and return these
        # Here I am checking if we are at last minimizing agent i.e. ghost and if we are then
        # maximize function will be called and if we are not at the last agent then minimizing function
        # is recursed for next agent i.e. agent + 1 till we reach the last agent
        if player == gameState.getNumAgents() - 1:
          for action in gameState.getLegalActions(player):
            tmp = min(value, self.maximize(gameState.generateSuccessor(player, action), alpha, beta, d - 1)[0])
            if tmp <= value:
              value = tmp
            if value < alpha:
              return value
            beta = min(beta, value)
        else:
          for action in gameState.getLegalActions(player):
            tmp = min(value, self.minimize(gameState.generateSuccessor(player, action), alpha, beta, d, player + 1))
            if tmp <= value:
              value = tmp
            if value < alpha:
              return value
            beta = min(beta, value)
        return value

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

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

