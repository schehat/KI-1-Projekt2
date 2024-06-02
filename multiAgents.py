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


import random

import util
from game import Agent, Directions
from pacman import GameState
from util import manhattanDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
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
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        The evaluation functions checks the position of the new state if there is a ghost then
        avoid it, if there is food take it. If none of the both cases then estimate distance to
        closest food with Manhattan Distance dist and take the action with the least distance.
        The evaluation functions returns -dist to ensure that higher values are better
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Check if ghost in successor position then avoid it
        ghostPositions = successorGameState.getGhostPositions()
        for ghostP in ghostPositions:
            if ghostP == newPos:
                return float("-inf")

        # Check if food in successor position then take it
        if newFood[newPos[0]][newPos[1]]:
            return float("inf")

        # Estimate distance to nearest food according to current GameState
        minDist = float("inf")
        food = currentGameState.getFood().asList()
        for f in food:
            dist = util.manhattanDistance(f, newPos)
            if dist < minDist:
                minDist = dist

        # Return -minDist because higher values are better
        return -minDist


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        # For Pacman
        def maxScore(gameState, currentDepth):
            # Terminal Condition
            if gameState.isWin() or gameState.isLose() or currentDepth >= self.depth:
                return self.evaluationFunction(gameState)

            score = float("-inf")
            actions = gameState.getLegalActions(0)
            for action in actions:
                successor = gameState.generateSuccessor(0, action)
                score = max(score, minScore(successor, currentDepth, 1))
            return score

        # For all ghosts
        def minScore(gameState, currentDepth, currentAgent):
            # Terminal condition for recursion by reaching terminal state or max depth
            if gameState.isWin() or gameState.isLose() or currentDepth >= self.depth:
                return self.evaluationFunction(gameState)

            score = float("inf")
            # Iterate trough actions of currentAgent which is a ghost
            actions = gameState.getLegalActions(currentAgent)
            for action in actions:
                successor = gameState.generateSuccessor(currentAgent, action)
                # Check if iterated trough all ghosts
                if currentAgent < (gameState.getNumAgents() - 1):
                    # Ghosts left, call MIN of next ghost (currentAgent + 1) and take min() compared
                    # to current score
                    score = min(
                        score, minScore(successor, currentDepth, currentAgent + 1)
                    )
                else:
                    # Iterated trough all ghost, next turn is Pacman. Call MAX and increase depth
                    score = min(score, maxScore(successor, currentDepth + 1))

            return score

        # Pacman/MAX starts
        actions = gameState.getLegalActions(0)
        currentScore = float("-inf")
        currentAction = ""
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            # Next layer is MIN from ghosts to minimize scores
            nextScore = minScore(nextState, 0, 1)
            # Select action that is maximum of the scores of the successors
            if nextScore > currentScore:
                currentAction = action
                currentScore = nextScore

        return currentAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # For Pacman
        def maxScore(gameState, currentDepth, alpha, beta):
            # Terminal Condition
            if gameState.isWin() or gameState.isLose() or currentDepth >= self.depth:
                return self.evaluationFunction(gameState)

            score = float("-inf")
            actions = gameState.getLegalActions(0)
            for action in actions:
                successor = gameState.generateSuccessor(0, action)
                score = max(score, minScore(successor, currentDepth, 1, alpha, beta))
                if score > beta:
                    return score
                alpha = max(alpha, score)
            return score

        # For all ghosts
        def minScore(gameState, currentDepth, currentAgent, alpha, beta):
            # Terminal condition for recursion by reaching terminal state or max depth
            if gameState.isWin() or gameState.isLose() or currentDepth >= self.depth:
                return self.evaluationFunction(gameState)

            score = float("inf")
            # Iterate trough actions of currentAgent which is a ghost
            actions = gameState.getLegalActions(currentAgent)
            for action in actions:
                successor = gameState.generateSuccessor(currentAgent, action)
                # Check if iterated trough all ghosts
                if currentAgent < (gameState.getNumAgents() - 1):
                    # Ghosts left, call MIN of next ghost (currentAgent + 1) and take min() compared
                    # to current score
                    score = min(
                        score,
                        minScore(
                            successor, currentDepth, currentAgent + 1, alpha, beta
                        ),
                    )
                else:
                    # Iterated trough all ghost, next turn is Pacman. Call MAX and increase depth
                    score = min(
                        score, maxScore(successor, currentDepth + 1, alpha, beta)
                    )
                if score < alpha:
                    return score
                beta = min(beta, score)

            return score

        # Pacman/MAX starts
        actions = gameState.getLegalActions(0)
        currentScore = float("-inf")
        currentAction = ""
        alpha = float("-inf")
        beta = float("inf")
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            # Next layer is MIN from ghosts to minimize scores
            nextScore = minScore(nextState, 0, 1, alpha, beta)

            # Select action that is maximum of the scores of the successors
            if nextScore > currentScore:
                currentAction = action
                currentScore = nextScore

            if nextScore > beta:
                return currentAction
            alpha = max(alpha, nextScore)

        return currentAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
