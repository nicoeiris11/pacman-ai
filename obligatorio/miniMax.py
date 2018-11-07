import random
import math
from game import Directions
from numpy import power

statesValues = dict()

def evaluationFunctionRec(currentGameState, agentIndex, depth=0, maxDepth=7):  
    
    if currentGameState.isWin() or currentGameState.isLose():
        return currentGameState.getScore()
    
    if  depth > maxDepth:
        pacmanX, pacmanY = currentGameState.data.agentStates[0].configuration.pos
        minDistance = math.inf
        
        for x in range(currentGameState.data.food.width):
            for y in range(currentGameState.data.food.height):
                if currentGameState.data.food[x][y]:
                    distance = math.sqrt(power(pacmanX - x, 2) + power(pacmanY - y, 2))
                    if distance < minDistance:
                        minDistance = distance
        
        return currentGameState.getScore() - minDistance

    states = []

    for action in currentGameState.getLegalActions(agentIndex):
        sucessor = currentGameState.generateSuccessor(agentIndex, action)
        stateHash = hash(sucessor) #* 10 + agentIndex

        if (stateHash in statesValues.keys()):
          states.append(statesValues[stateHash])
        else:
          nextAgent = (agentIndex + 1) % currentGameState.getNumAgents()
          stateValue = evaluationFunctionRec(sucessor, nextAgent, depth + 1)
          statesValues[stateHash] = stateValue
          states.append(stateValue)

    if agentIndex == 0:
      return max(states)
    else:
      return min(states)

def evaluationFunction(currentGameState, d):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
  """
    return evaluationFunctionRec(currentGameState, 1, 0, d)

def getAction(gameState, d):
    legalActions = gameState.getLegalActions(0)
    bestScore = -999999999999999999
    bestAction = Directions.STOP
    for action in legalActions:
        score = evaluationFunction(gameState.generateSuccessor(0, action), d)
        if (score > bestScore):
            bestScore = score
            bestAction = action
    return bestAction
    """
      Returns the minimax action from the current gameState using self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

    """

