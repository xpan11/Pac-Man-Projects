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
import random, util, math

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
        # return successorGameState.getScore()
        foods = currentGameState.getFood().count()
        possiblefood = newFood.asList()
        length = len(possiblefood)
        ans = 0
        #print(len(possiblefood))


        foods = currentGameState.getFood().count()
        possiblefood = newFood.asList()
        length = len(possiblefood)
        ans = 0
        #print(len(possiblefood))
        if foods == length:
            compare = []
            for i in possiblefood:
                geffect = 0
                for j in newGhostStates:
                    numstate= len(newGhostStates)
                    geffect = geffect + manhattanDistance(i,j.getPosition())/numstate

                compare.append(manhattanDistance(i,newPos)+0.01*geffect)
            ans =0- min(compare)

        else:
            compare = []
            for i in possiblefood:
                getfood = manhattanDistance(i,newPos)
                compareg = []
                for j in  newGhostStates:
                    foodgetg = manhattanDistance(j.getPosition(), i)
                    compareg.append(foodgetg)
                compare.append((sum(compareg)/len(compareg)) - getfood)
            if(len(compare)>0):
                ans =1+max(compare)
        return ans
        



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
        self.pacmanIndex = 0

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
        #util.raiseNotDefined()
        ans = self.evaluate(gameState ,0,0)
        return ans[1]
    def evaluate(self, gameState, indexx, depth):
        if indexx >= gameState.getNumAgents():
            indexx = 0
            depth = depth+1

        if self.depth == depth or gameState.isWin():
            ans = self.evaluationFunction(gameState) 
        elif indexx == 0:
            ans = self.minmax(gameState, indexx, depth, "max")
        elif indexx >0:
            ans = self.minmax(gameState, indexx, depth, "min")
        return ans
        
    def minmax(self, gameState, indexx, depth, instruction):
        compare = []
        action = []
        count = 0
        #index = 0
        #curdepth  =(indexx)//gameState.getNumAgents()

        #ans = (0, "action")

       # if indexx>=gameState.getNumAgents():
        #    index = indexx%gameState.getNumAgents()
        #if indexx< gameState.getNumAgents():
         #   index = indexx

        if not gameState.getLegalActions(indexx):
            count = self.evaluationFunction(gameState)
            return count
        else:
            for i in gameState.getLegalActions(indexx):

                if i!="Stop":
                    #indexx = indexx+1
                    shortname = self.evaluate((gameState.generateSuccessor(indexx, i)), indexx+1, depth)
                    if type(shortname)==tuple:
                        compare.append(shortname[0])
                    else:
                        compare.append(shortname)
                    action.append(i)
        #print(len(compare))
        #print(compare[0])
        if(instruction == "max"):
            ans = (max(compare), action[int(compare.index(max(compare)))])
        elif(instruction=="min"):
            ans = (min(compare), action[int(compare.index(min(compare)))])

        return ans


"""
    def minn(self, gameState, indexx, depth):
        #ans = (0, "action")
        compare = []
        action = []
        count = 0
       # index =0 
       # curdepth  =(indexx)//gameState.getNumAgents()

        #if indexx>=gameState.getNumAgents():
         #   index = indexx%gameState.getNumAgents()
        #if indexx< gameState.getNumAgents():
         #   index = indexx
        if not gameState.getLegalActions(indexx):
            count = self.evaluationFunction(gameState)
            return count
        else:
            for i in gameState.getLegalActions(indexx):
                if i!="Stop":
                    #indexx = indexx+1
                    shortname = self.evaluate((gameState.generateSuccessor(indexx, i)), indexx+1, depth)
                    print(type(shortname))
                    if type(shortname)==tuple:
                        compare.append(shortname[0])
                    else:
                        compare.append(shortname)
                    action.append(i)
        #print(len(compare))
        #print(compare[0])
        ans = (min(compare), action[compare.index(min(compare))])
        return ans
"""
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = 0-float('inf')
        beta = float('inf')
        ans = self.evaluate(gameState ,0,0,alpha,beta)

        return ans[1]
    def evaluate(self, gameState, indexx, depth, alpha, beta):
        if indexx >= gameState.getNumAgents():
            indexx = 0
            depth = depth+1

        if self.depth == depth:
            ans = self.evaluationFunction(gameState) 
        elif indexx == 0:
            ans = self.minmax(gameState, indexx, depth, alpha, beta, "max")
        elif indexx >0:
            ans = self.minmax(gameState, indexx, depth, alpha, beta, "min")
        return ans
        
    def minmax(self, gameState, indexx, depth, alpha, beta, instruction):
        compare = []
        action = []
        count = 0
        ansmax = (0-float('inf'), "act");
        ansmin = (float('inf'),"act")
        #index = 0
        curvalue = 0
        #curdepth  =(indexx)//gameState.getNumAgents()

        #ans = (0, "action")

       # if indexx>=gameState.getNumAgents():
        #    index = indexx%gameState.getNumAgents()
        #if indexx< gameState.getNumAgents():
         #   index = indexx

        if not gameState.getLegalActions(indexx):
            count = self.evaluationFunction(gameState)
            return count
        else:
            for i in gameState.getLegalActions(indexx):

                if i!="Stop":
                    #indexx = indexx+1
                    shortname = self.evaluate((gameState.generateSuccessor(indexx, i)), indexx+1, depth, alpha, beta)
                    if type(shortname)==tuple:
                        curvalue = shortname[0];   
                    else:
                        curvalue = shortname;
                    if(instruction == "max"):
                        curmax= max(curvalue, ansmax[0])
                        if(curmax != ansmax[0]):
                            ansmax =(curmax, i)
                        if(ansmax[0]>beta):
                            return ansmax  
                        alpha = max(alpha, ansmax[0])                      
                            #ans = (curvalue, i)                            
                    elif(instruction=="min"):
                        curmin= min(curvalue, ansmin[0])
                        if(curmin != ansmin[0]):
                            ansmin =(curmin, i)
                        if(ansmin[0]<alpha):
                            return ansmin  
                        beta = min(beta, ansmin[0])
                        #if(curvalue<=alpha):
                            #ans = (curvalue,i)
                            
            if(instruction == "max"):
                return ansmax
            elif(instruction == "min"):
                return ansmin
                                #compare.append(shortname)
                    
                    
        #print(len(compare))

        #print(compare[0])
        #if(instruction == "max"):

        #if(instruction=="max"):
         #   ans = (max(compare), action[int(compare.index(max(compare)))]) 
          #  alpha = max(alpha, ans[0])
        #elif(instruction=="min"):
         #   ans = (min(compare), action[int(compare.index(min(compare)))]) 
          #  beta=min(beta, ans[0])
        #elif(instruction=="min"):
         #   ans = (min(compare), action[int(compare.index(min(compare)))])

        #return ans

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
        #util.raiseNotDefined()
        alpha = 0-float('inf')
        beta = float('inf')
        return self.evaluate(gameState, 0,0,alpha,beta)[1]
    def evaluate(self, gameState, indexx, depth, alpha, beta):
        if indexx >= gameState.getNumAgents():
            indexx = 0
            depth = depth+1

        if self.depth == depth:
            ans = self.evaluationFunction(gameState) 
        elif indexx == 0:
            ans = self.minmax(gameState, indexx, depth, alpha, beta, "max")
        elif indexx >0:
            ans = self.minmax(gameState, indexx, depth, alpha, beta, "expect")
        return ans
        
    def minmax(self, gameState, indexx, depth, alpha, beta, instruction):
        compare = []
        action = []
        count = 0
        ansmax = (0-float('inf'), "act");
        ansexp = (0,"act")
        #index = 0
        p=0
        curvalue = 0
       
        #curdepth  =(indexx)//gameState.getNumAgents()

        #ans = (0, "action")

       # if indexx>=gameState.getNumAgents():
        #    index = indexx%gameState.getNumAgents()
        #if indexx< gameState.getNumAgents():
         #   index = indexx

        if not gameState.getLegalActions(indexx):
            count = self.evaluationFunction(gameState)
            return count
        else:
            p = len(gameState.getLegalActions(indexx))
            for i in gameState.getLegalActions(indexx):

                if i!="Stop":
                    #indexx = indexx+1
                    shortname = self.evaluate((gameState.generateSuccessor(indexx, i)), indexx+1, depth, alpha, beta)
                    if type(shortname)==tuple:
                        curvalue = shortname[0];   
                    else:
                        curvalue = shortname;
                    if(instruction == "max"):
                        curmax= max(curvalue, ansmax[0])
                        if(curmax != ansmax[0]):
                            ansmax =(curmax, i)
                        if(ansmax[0]>beta):
                            return ansmax  
                        alpha = max(alpha, ansmax[0])                      
                            #ans = (curvalue, i)                            
                    elif(instruction=="expect"):
                        curexp = ansexp[0] + (curvalue)/p
                        ansexp = (curexp, i)

                        #curmin= min(curvalue, ansmin[0])
                        #if(curmin != ansmin[0]):
                         #   ansmin =(curmin, i)
                        #if(ansmin[0]<alpha):
                         #   return ansmin  
                        #beta = min(beta, ansmin[0])
                        #if(curvalue<=alpha):
                            #ans = (curvalue,i)
                        #evaluationFunction()
                            
            if(instruction == "max"):
                return ansmax
            elif(instruction == "expect"):
                return ansexp

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    I try several methods and the performance of those didnt go better than "chase me or I am not going to run" with build in score function
    """
    "*** YOUR CODE HERE ***"

 #successorGameState = currentGameState.generatePacmanSuccessor(actio*n)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newCapsule = currentGameState.getCapsules()
    newScore = currentGameState.getScore()
    food =[]
    #scaretiem = 0 
    
        #print(manhattanDistance(newPos, i))
    #food = []
    for i in newFood.asList():
        x2 = (i[0]-newPos[0])*(i[0]-newPos[0])
        y2 = (i[1]-newPos[1])*(i[1]-newPos[1])
        dis = math.sqrt(x2+y2)
        dis0 = manhattanDistance(i,newPos)
        food.append((dis+dis0)/2+random.randint(0,5))
         
    g =[]
    for j in newGhostStates:
        g.append(manhattanDistance(newPos, j.getPosition()))
    c= []
    for x in newCapsule:
        c.append(manhattanDistance(newPos, x))
    numGs = 0
    for i in newGhostStates:
        if i.scaredTimer>0:
            numGs=numGs+1
    closeGindex = g.index(min(g))
    closeGtime = newScaredTimes[closeGindex]
    gd = min(g)
    #fd = min(food)
    #if len(newFood.asList())!=0:
    #   fd=(manhattanDistance(newPos, newFood[0]))



    
    
    ans = newScore+gd
    #min(food)-100*len(food)+gd+newScore
    #newScore+gd
    return ans










   # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
