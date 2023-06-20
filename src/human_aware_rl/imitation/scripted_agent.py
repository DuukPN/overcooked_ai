import random

from overcooked_ai_py.agents.agent import (
    Agent,
)
from overcooked_ai_py.mdp.actions import (
    Direction,
    Action,
)
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


class DummyAI(Agent):
    """
    Randomly samples actions. Used for debugging
    """

    def __init__(self, id, layout_name):
        self.id = id
        self.roomName = layout_name
        # print("Initialization")

    dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    oppositeDirs = [[0, -1], [0, 1], [-1, 0], [1, 0]]

    # This needs to be changed if not the last version is used
    def GetRemainingTimeOfObject(self, object):
        cookingTime = object.cook_time
        if cookingTime == None:
            cookingTime = object.recipe.time

        if (object.cooking_tick == -1):  # The soup has not started cooking yet
            return 100000

        return cookingTime - object.cooking_tick

    # End

    def getDirectionTo(self, x, y):

        if (self.distances[y][x] == 0):
            return None

        if (self.distances[y][x] > 5000):
            return None

        while self.distances[y][x] != 1:
            for i in range(0, len(self.dirs)):
                xNew = x + self.dirs[i][0]
                yNew = y + self.dirs[i][1]
                xNew2 = x + self.dirs[i][0] + self.dirs[i][0]
                yNew2 = y + self.dirs[i][1] + self.dirs[i][1]

                if self.insideBounds(xNew, yNew) and self.distances[yNew][xNew] == self.distances[y][
                    x] - 1 and self.isWalkable(xNew, yNew, False):
                    if self.distances[yNew2][xNew2] != 0:
                        if self.insideBounds(xNew2, yNew2) and self.distances[yNew2][xNew2] == self.distances[y][
                            x] - 2 and self.isWalkable(xNew2, yNew2, False):
                            x = xNew2
                            y = yNew2
                            break

                if self.insideBounds(xNew, yNew) and self.distances[yNew][xNew] == self.distances[y][
                    x] - 1 and self.isWalkable(xNew, yNew, False):
                    x = xNew
                    y = yNew
                    break
        # print([x,y])
        for i in range(0, len(self.dirs)):
            xNew = x + self.dirs[i][0]
            yNew = y + self.dirs[i][1]
            if self.insideBounds(xNew, yNew) and self.distances[yNew][xNew] == self.distances[y][
                x] - 1 and self.isWalkable(xNew, yNew, False):
                # print(i)
                return i

        # selectedDir = -1

        # if selectedDir==-1: # in case of no path
        #    return 1000000

        # x = x + self.dirs[selectedDir][0]
        # y = y + self.dirs[selectedDir][1]
        # if(self.distances[y][x] == 0):
        #    return selectedDir
        # selectedSecondDir = None
        # selectedPossibleDir = None

        # for i in range(0, len(self.dirs)):
        #    xNew = x+self.dirs[i][0]
        #    yNew = y+self.dirs[i][1]
        #    if self.insideBounds(xNew, yNew) and self.distances[yNew][xNew] == self.distances[y][x]-1 and self.isWalkable(xNew, yNew, False):
        #        if i == selectedDir:
        #            selectedSecondDir = i
        #        selectedPossibleDir = i
        # if selectedSecondDir == None:
        #    selectedSecondDir = selectedPossibleDir
        # return selectedSecondDir

    def distanceTo(self, x, y, ignoreAlly=False):
        if (self.distances[y][x] == 0):
            return 0
        if (self.distances[y][x] == 10000):
            return 10000

        selectedDir = -1
        for i in range(0, len(self.dirs)):
            xNew = x + self.dirs[i][0]
            yNew = y + self.dirs[i][1]
            if self.insideBounds(xNew, yNew) and self.distances[yNew][xNew] == self.distances[y][
                x] - 1 and self.isWalkable(xNew, yNew, False):
                selectedDir = i

        if selectedDir == -1:  # in case of no path
            return 10000

        x = x + self.dirs[selectedDir][0]
        y = y + self.dirs[selectedDir][1]
        if (self.distances[y][x] == 0):
            return 1
        if (self.distances[y][x] == 10000):
            return 10000

        selectedSecondDir = None
        selectedPossibleDir = -1
        for i in range(0, len(self.dirs)):
            xNew = x + self.dirs[i][0]
            yNew = y + self.dirs[i][1]
            if self.distances[yNew][xNew] == self.distances[y][x] - 1 and self.isWalkable(xNew, yNew, False):
                if i == selectedDir:
                    selectedSecondDir = i
                selectedPossibleDir = i
        if selectedPossibleDir == -1:
            return 10000
        if selectedSecondDir == None:
            selectedSecondDir = selectedPossibleDir

        x = x + self.dirs[selectedSecondDir][0]
        y = y + self.dirs[selectedSecondDir][1]

        if selectedSecondDir == selectedDir:
            return 2 + self.distanceToHelper(x, y, ignoreAlly)
        else:
            return 3 + self.distanceToHelper(x, y, ignoreAlly)

    def distanceToHelper(self, x, y, ignoreAlly):
        if (self.distances[y][x] == 0):
            return 0
        selectedDir = -1
        for i in range(0, len(self.dirs)):
            xNew = x + self.dirs[i][0]
            yNew = y + self.dirs[i][1]
            if self.distances[yNew][xNew] == self.distances[y][x] - 1 and self.isWalkable(xNew, yNew, ignoreAlly):
                selectedDir = i
        if (selectedDir == -1):
            return 10000
        x = x + self.dirs[selectedDir][0]
        y = y + self.dirs[selectedDir][1]
        return 1 + self.distanceToHelper(x, y, ignoreAlly)

    def managePositon(self, x, y, ignoreAlly, firstTime=False):

        for i in range(0, len(self.dirs)):
            posToConsider = [x + self.dirs[i][0], y + self.dirs[i][1]]
            if not self.insideBounds(posToConsider[0], posToConsider[1]):
                continue
            if not self.isWalkable(posToConsider[0], posToConsider[1], ignoreAlly):
                if not firstTime:
                    self.distances[posToConsider[1]][posToConsider[0]] = min(self.distances[y][x] + 1,
                                                                             self.distances[posToConsider[1]][
                                                                                 posToConsider[0]])
                continue
            if not self.checkedPositions[posToConsider[1]][posToConsider[0]] == 0:
                continue

            self.positions.append(posToConsider)
            self.checkedPositions[posToConsider[1]][posToConsider[0]] = 1
            self.distances[posToConsider[1]][posToConsider[0]] = self.distances[y][x] + 1

    def printDistances(self):
        for x in range(0, len(self.distances)):
            print(self.distances[x])

    def computePathingFrom(self, x, y, ignoreAlly=False, useFirstTime=True):
        self.distances = [[10000] * self.mapWidth for i in range(0, self.mapHeight)]
        self.checkedPositions = [[0] * self.mapWidth for i in range(0, self.mapHeight)]

        self.distances[y][x] = 0
        self.checkedPositions[y][x] = 1

        index = 0
        self.positions = []
        self.positions.append([x, y])
        firstTime = useFirstTime
        while index < len(self.positions):
            self.managePositon(self.positions[index][0], self.positions[index][1], ignoreAlly, firstTime)
            firstTime = False
            index = index + 1

    def isPlaceabale(self, x, y):
        return self.map[y][x] == 'X'

    def isWalkable(self, x, y, ignoreAlly):
        if (ignoreAlly):
            return self.map[y][x] == ' ' or self.map[y][x] == 'x' or self.map[y][x] == chr(self.allyId + 48) or \
                   self.map[y][x] == chr(self.id + 48)
        else:
            return self.map[y][x] == ' ' or self.map[y][x] == chr(self.id + 48)

    def initialization(self):
        try:
            self.init
        except AttributeError:

            ######## CUSTOMIZABLE PARAMETERS
            self.rangeToConsiderForAllyMovement = 1  # NOT USED ANYWHERE
            self.maximumWaitTimeForSoup = 3  # Might arrive to pick up sooner with 3 timesteps
            self.putSoupOnCounterBias = 1
            # print("map name is " + str(self.roomName))
            self.biasInPuttingOnionsWhereThereAreMultipleOnions = 7
            #######  END OF CUSTOMIZABLE PARAMETERS

            self.init = None
            self.mdp = OvercookedGridworld.from_layout_name(
                self.roomName
            )
            self.allyId = 1 - self.id
            self.map = self.mdp.terrain_mtx

            # This can be used if an older version of overcooked is used,but this has changed
            # self.cookingTime = self.mdp.soup_cooking_time

            self.mapWidth = len(self.map[0])
            self.mapHeight = len(self.map)
            self.stoves = []
            self.onionDispensers = []
            self.delivery = []
            self.dishDispensers = []
            self.counters = []
            for y in range(0, len(self.map)):
                for x in range(0, len(self.map[y])):
                    if self.map[y][x] == 'P':
                        self.stoves.append([x, y])
                    if self.map[y][x] == 'S':
                        self.delivery.append([x, y])
                    if self.map[y][x] == 'O':
                        self.onionDispensers.append([x, y])
                    if self.map[y][x] == 'D':
                        self.dishDispensers.append([x, y])
                    if self.map[y][x] == 'X':
                        self.counters.append([x, y, False, 100000,
                                              100000])  # x,y, isSomethingOnIt, distance to closest delivery, distance to closest stove

    def insideBounds(self, x, y):
        if (x < 0 or y < 0):
            return False
        if (x >= self.mapWidth or y >= self.mapHeight):
            return False
        return True

    def takeVariablesFromState(self, state):
        self.x = state.players[self.id].position[0]
        self.y = state.players[self.id].position[1]
        self.orientation = state.players[self.id].orientation
        self.heldObject = state.players[self.id].held_object

        self.allyX = state.players[self.allyId].position[0]
        self.allyY = state.players[self.allyId].position[1]
        self.allyOrientation = state.players[self.allyId].orientation
        self.allyHeldObject = state.players[self.allyId].held_object

        # This exception happens only in the first go through this function
        try:
            self.map[self.allyLastY][self.allyLastX] = ' '
            self.map[self.lastY][self.lastX] = ' '
            self.map[self.lastXy][self.lastXx] = ' '
        except AttributeError:
            self.map

        self.lastY = self.y
        self.lastX = self.x
        self.allyLastY = self.allyY
        self.allyLastX = self.allyX

        self.map[self.allyY][self.allyX] = chr(self.allyId + 48)

        orientationY = self.allyY + self.allyOrientation[1]
        orientationX = self.allyX + self.allyOrientation[0]
        if (self.map[orientationY][orientationX] == ' '):
            self.lastXx = orientationX
            self.lastXy = orientationY
            if (random.randint(0, 100) < 50):
                self.map[orientationY][orientationX] = 'x'

        self.map[self.y][self.x] = chr(self.id + 48)

    def front(self):
        return [self.x + self.orientation[0], self.y + self.orientation[1]]

    def TakeOnionScore(self, x, y, state):
        if (self.distances[y][x] > 5000):
            return 0

        if self.map[y][x] == 'O':
            return max(50 - self.distances[y][x], 0)
        if self.IsSoupReadyToStart(state, x, y):
            return 47
        if ((x, y) in state.objects and state.objects[(x, y)].name == "onion"):
            return 50 - self.distanceTo(x, y)

        # for i in range (0, len(self.onions)):
        #     if (self.onions[i].position[0] == x and self.onions[i].position[1] == y):
        #         myDist = 0
        #         self.computePathingFrom(x, y)
        #         myDist = myDist + self.distances[self.y][self.x]

        #         if self.getDirectionTo(self.x, self.y) == None:
        #            self.computePathingFrom(self.x, self.y, False, False)
        #            return 0

        #         expectedX = x + self.getDirectionTo(self.x, self.y)[0]
        #         expectedY = y + self.getDirectionTo(self.x, self.y)[1]

        #         closestDistanceToStove = 10000
        #         counterDist = 10000
        #         for j in range(0, len(self.stoves)):
        #             self.computePathingFrom(self.stoves[j][0], self.stoves[j][1])
        #             newDist = self.distances[expectedY][expectedX]
        #             newCounterDist = self.distances[y][x]
        #             if newDist < closestDistanceToStove:
        #                 closestDistanceToStove = newDist
        #             if newCounterDist < counterDist:
        #                 counterDist = newCounterDist

        #         myDist = myDist + closestDistanceToStove

        #         self.computePathingFrom(self.x, self.y, False, False)

        #         if(myDist > counterDist - self.putSoupOnCounterBias):
        #             return 0

        #         return max(48 - self.distances[y][x], 0)

        return 0

    def GoodSoupPlacementScore(self, x, y, state):
        # print("calculating score for " + str(x) + " " + str(y))
        self.computePathingFrom(self.x, self.y, False, False)
        for i in range(0, len(self.delivery)):
            if (self.delivery[i][0] == x and self.delivery[i][
                1] == y):  # if it's a stove
                return max(50 - self.distances[y][x], 0)

        oneOtherPossibleLocation = False
        for i in range(0, len(self.counters)):
            xc = self.counters[i][0]
            yc = self.counters[i][1]
            if (self.counters[i][3] < 5000 and (xc, yc) in state.objects and state.objects[(xc, yc)].name == "soup"):
                oneOtherPossibleLocation = True

        if oneOtherPossibleLocation == True:
            return 0

        if (self.map[y][x] == 'X' and not (x, y) in state.objects):
            for i in range(0, len(self.counters)):
                if (self.counters[i][0] == x and self.counters[i][1] == y):
                    return max(28 - self.counters[i][3], 0)

            return max(28 - self.distances[y][x], 0)

        return 0

    def GoodOnionPlacementScore(self, x, y, state):
        # print("calculating score for " + str(x) + " " + str(y))
        self.computePathingFrom(self.x, self.y, False, False)
        for i in range(0, len(self.stoves)):
            if (self.stoves[i][0] == x and self.stoves[i][1] == y):  # if it's a stove
                if (not (x, y) in state.objects) or len(state.objects[(x, y)].ingredients) < 3:

                    bias = 0
                    if (x, y) in state.objects:
                        bias = bias + len(
                            state.objects[(x, y)].ingredients) * self.biasInPuttingOnionsWhereThereAreMultipleOnions

                    return max(50 - self.distances[y][x], 0) + bias

        oneOtherPossibleLocation = False
        for i in range(0, len(self.counters)):
            xc = self.counters[i][0]
            yc = self.counters[i][1]
            if (self.counters[i][4] < 5000 and (xc, yc) in state.objects and state.objects[(xc, yc)].name == "onion"):
                oneOtherPossibleLocation = True

        if oneOtherPossibleLocation == True:
            return 0

        if (self.map[y][x] == 'X' and not (x, y) in state.objects):
            for i in range(0, len(self.counters)):
                if (self.counters[i][0] == x and self.counters[i][1] == y):
                    return max(28 - self.counters[i][4], 0)
            # myDist = 0
            # self.computePathingFrom(x, y)
            # myDist = myDist + self.distances[self.y][self.x]

            # try:

            #     if (self.getDirectionTo(self.x, self.y) == None): # if there is no path
            #         self.computePathingFrom(self.x, self.y,False, False)
            #         return 0

            #     expectedX = x + self.getDirectionTo(self.x, self.y)[0]
            #     expectedY = y + self.getDirectionTo(self.x, self.y)[1]
            # except TypeError:
            #     self.computePathingFrom(self.x, self.y,False, False)
            #     return 0

            # closestDistanceToStove = 10000
            # counterDist = 10000
            # for j in range(0, len(self.stoves)):
            #     self.computePathingFrom(self.stoves[j][0], self.stoves[j][1])
            #     newDist = self.distances[expectedY][expectedX]
            #     newCounterDist = self.distances[y][x]
            #     if newDist < closestDistanceToStove:
            #         closestDistanceToStove = newDist
            #     if newCounterDist < counterDist:
            #         counterDist = newCounterDist

            # myDist = myDist + closestDistanceToStove

            # self.computePathingFrom(self.x, self.y, False, False)

            # if(myDist < counterDist + self.putSoupOnCounterBias):
            #     return 0

            return max(28 - self.distances[y][x], 0)

        return 0

    def actionFromDir(self, dir):
        if dir == 0:
            return Direction.SOUTH
        if dir == 1:
            return Direction.NORTH
        if dir == 2:
            return Direction.EAST
        if dir == 3:
            return Direction.WEST
        return Action.STAY

    def oppositeActionFromDir(self, dir):
        if dir == 0:
            return Direction.NORTH
        if dir == 1:
            return Direction.SOUTH
        if dir == 2:
            return Direction.WEST
        if dir == 3:
            return Direction.EAST
        return Action.STAY

    def TimeToPickUpAndDeliverSoup(self, state, x, y, heldObject):
        if (heldObject != None and heldObject.name == "soup"):
            return [10000, 1, 1]

        time = 0

        self.computePathingFrom(x, y, False, False)

        closestCounter = 10000
        bestX = -1
        bestY = -1
        hasOnion = False
        hasNothing = False

        timeTilCounter = 0
        timeTilDish = 0
        TimeTilSoup = 0

        counterx = -1
        countery = -1
        dishx = -1
        dishy = -1
        soupx = -1
        soupy = -1

        # print ("x initial: " + str(x) + " y initial " + str(y))
        if (heldObject != None and heldObject.name == "onion"):
            time = time + 1  # adding the time to perform the action of dropping the onion
            hasOnion = True
            for i in range(0, len(self.stoves)):
                if (self.IsSoupMissingOnions(state, i)):
                    stoveX = self.stoves[i][0]
                    stoveY = self.stoves[i][1]
                    if self.distances[stoveY][stoveX] < closestCounter:
                        closestCounter = self.distances[stoveY][stoveX]
                        bestX = stoveX
                        bestY = stoveY

            # self.printDistances()
            for i in range(0, len(self.counters)):
                counterX = self.counters[i][0]
                counterY = self.counters[i][1]
                if ((counterX, counterY) not in state.objects):  # if the counter is empty
                    if self.distances[counterY][counterX] < closestCounter:
                        closestCounter = self.distances[counterY][counterX]
                        bestX = counterX
                        bestY = counterY

            timeTilCounter = closestCounter
            counterx = bestX
            countery = bestY
            if bestX == -1:
                # print ("crying, there is no place to put the onion")
                return [10000, 1,
                        1]  # cry, there is no place to put the onion, and we cry because we hold the onion and there is nothing we can do
            time = time + closestCounter

            bestDist = 5000

            for i in range(0, len(self.dirs)):
                newX = bestX + self.dirs[i][0]
                newY = bestY + self.dirs[i][1]
                if not self.insideBounds(newX, newY) or not self.isWalkable(newX, newY, True):
                    continue
                if (self.distances[newY][newX] < bestDist):
                    bestDist = self.distances[newY][newX]
                    x = newX
                    y = newY

        closestDish = 10000
        bestX = -1
        bestY = -1

        self.computePathingFrom(x, y, False, False)  # x and y might have new values
        # self.printDistances()
        if (heldObject == None or hasOnion):
            time = time + 1  # adding the time to perform the action of picking the dish
            hasNothing = True
            for i in range(0, len(self.dishDispensers)):
                dishX = self.dishDispensers[i][0]
                dishY = self.dishDispensers[i][1]
                if (self.distances[dishY][dishX] < closestDish):
                    closestDish = self.distances[dishY][dishX]
                    bestX = dishX
                    bestY = dishY

            for i in range(0, len(self.counters)):
                dishX = self.counters[i][0]
                dishY = self.counters[i][1]
                if ((dishX, dishY) in state.objects and state.objects[
                    (dishX, dishY)].name == "dish"):  # if there is a dish on a counter
                    if self.distances[dishY][dishX] < closestDish:
                        closestDish = self.distances[dishY][dishX]
                        bestX = dishX
                        bestY = dishY

            timeTilDish = closestDish
            dishx = bestX
            dishy = bestY

            if bestX == -1:
                # print ("crying, there is no place to take dish from")
                return [10000, 1, 1]  # cry
            time = time + closestDish

            bestDist = 10000
            for i in range(0, len(self.dirs)):
                newX = bestX + self.dirs[i][0]
                newY = bestY + self.dirs[i][1]
                if not self.insideBounds(newX, newY) or not self.isWalkable(newX, newY, True):
                    continue
                if (self.distances[newY][newX] < bestDist):
                    bestDist = self.distances[newY][newX]
                    x = newX
                    y = newY

        self.computePathingFrom(x, y, False, False)  # x and y might have different values

        if (len(self.startedSoups) == 0):  # there is no soup to be picked up
            return [10000, 1, 1]

        bestDist = 10000
        bestX = -1
        bestY = -1
        # self.printDistances()
        isThereASoup = False
        for i in range(0, len(self.startedSoups)):
            newX = self.startedSoups[i].position[0]
            newY = self.startedSoups[i].position[1]

            if self.map[newY][newX] != 'P':  # ignore soups that are not on pots
                continue
            isThereASoup = True
            if (state.objects[(newX, newY)].cook_time_remaining < bestDist):
                bestDist = state.objects[(newX, newY)].cook_time_remaining
                bestX = newX
                bestY = newY

        if bestX == -1:
            # print ("crying, there is no soup to take")
            return [10000, 1, 1]  # cry

        if not isThereASoup:
            return [10000, 1, 1]

        bestDist = 10000
        for i in range(0, len(self.dirs)):
            newX = bestX + self.dirs[i][0]
            newY = bestY + self.dirs[i][1]
            if not self.insideBounds(newX, newY) or not self.isWalkable(newX, newY, True):
                continue
            if (self.distances[newY][newX] < bestDist):
                bestDist = self.distances[newY][newX]
                x = newX
                y = newY

        # Time until taking the soup, soup location
        return [time + bestDist, [bestX, bestY],
                [[timeTilCounter, (counterx, countery)], [timeTilDish, (dishx, dishy)]]]

    def FindTargetForDelivery(self, state, x, y):
        self.computePathingFrom(x, y, True, False)
        canDeliverToDelivery = False

        bestDelivery = 10000

        bestX = -1
        bestY = -1
        for i in range(0, len(self.delivery)):
            dist = self.distances[self.delivery[i][1]][self.delivery[i][0]]
            if dist < bestDelivery:
                bestX = self.delivery[i][0]
                bestY = self.delivery[i][1]
                canDeliverToDelivery = True

        if canDeliverToDelivery:
            return [bestX, bestY]

        bestCounter = 10000

        for i in range(0, len(self.counters)):
            if (self.distances[self.counters[i][1]][self.counters[i][0]]) > 4000:
                continue
            if (self.counters[i][3] < bestCounter):
                bestX = self.counters[i][0]
                bestY = self.counters[i][1]

        return [bestX, bestY]

    def TimeToPickUpSoup(self, state, x, y):
        print("Time to pick up")

    def canDeliverSoup(self, state):

        self.computePathingFrom(self.x, self.y, True, False)

        for j in range(0, len(self.delivery)):
            dist = self.distances[self.delivery[j][1]][self.delivery[j][0]]
            if dist < 5000:
                return True

    def decideMyBehaviour(self, state):
        self.decidedAction = Action.STAY

        self.behaviour = "BringOnion"

        [timeUntilICanDeliverSoup, mySoupLocation, _] = self.TimeToPickUpAndDeliverSoup(state, self.x, self.y,
                                                                                        self.heldObject)
        [timeUntilAllyCanDeliverSoup, allySoupLocation, _] = self.TimeToPickUpAndDeliverSoup(state, self.allyLastX,
                                                                                             self.allyY,
                                                                                             self.allyHeldObject)

        if timeUntilICanDeliverSoup < 5000:
            soupAtPos = state.objects[(mySoupLocation[0], mySoupLocation[1])]
            timeUntilSoupIsDone = soupAtPos.cook_time_remaining
            # print("TIME UNTIL SOUP IS DONE IS " + str(timeUntilSoupIsDone))
            if (mySoupLocation == allySoupLocation):
                if (timeUntilICanDeliverSoup < timeUntilAllyCanDeliverSoup):
                    if (timeUntilICanDeliverSoup >= timeUntilSoupIsDone):
                        self.behaviour = "PickUpSoup"
                elif (timeUntilAllyCanDeliverSoup == timeUntilICanDeliverSoup):
                    if (
                            self.id == 0):  # in case of a tie, the lower id will try to bring the soup, this is only for when playing with another ai
                        if (timeUntilICanDeliverSoup >= timeUntilSoupIsDone):
                            self.behaviour = "PickUpSoup"

            else:
                if (timeUntilICanDeliverSoup < 4000):
                    if (timeUntilICanDeliverSoup >= timeUntilSoupIsDone):
                        self.behaviour = "PickUpSoup"

        # print("My time to pick and deliver soup: " + str(timeUntilICanDeliverSoup))
        # print("Ally time to pick and deliver soup: " + str(timeUntilAllyCanDeliverSoup))

        for i in range(0, len(self.startedSoups)):

            if self.startedSoups[i].cook_time_remaining > 10:
                continue
            newX = self.startedSoups[i].position[0]
            newY = self.startedSoups[i].position[1]

            if self.map[newY][newX] != 'P':  # ignore soups that are not on pots
                continue

            self.computePathingFrom(newX, newY, True, True)
            isThereDishForSoup = False
            for j in range(0, len(self.dishDispensers)):
                x = self.dishDispensers[j][0]
                y = self.dishDispensers[j][1]
                if (self.distances[y][x] < 4000):
                    isThereDishForSoup = True
                    break
            for j in range(0, len(self.dishes)):
                x = self.dishes[j].position[0]
                y = self.dishes[j].position[1]
                if (self.distances[y][x] < 4000):
                    isThereDishForSoup = True
                    break
            if isThereDishForSoup == False:
                self.computePathingFrom(self.x, self.y, True, False)
                isThereDishNearMe = False
                for j in range(0, len(self.dishDispensers)):
                    x = self.dishDispensers[j][0]
                    y = self.dishDispensers[j][1]
                    if (self.distances[y][x] < 4000):
                        isThereDishNearMe = True
                        break
                for j in range(0, len(self.dishes)):
                    x = self.dishes[j].position[0]
                    y = self.dishes[j].position[1]
                    if (self.distances[y][x] < 4000):
                        isThereDishNearMe = True
                        break
                if isThereDishNearMe and (
                        self.heldObject == None or self.heldObject.name == "dish" or self.heldObject.name == "onion"):
                    self.behaviour = "BringPlate"

        if self.heldObject != None:
            if self.heldObject.name == "dish":
                if self.behaviour != "PickUpSoup":
                    self.behaviour = "BringPlate"

        if self.heldObject != None:
            if (self.heldObject.name == "soup"):
                self.behaviour = "DeliverSoup"

    def IsSoupMissingOnions(self, state, i):
        x = self.stoves[i][0]
        y = self.stoves[i][1]
        if (x, y) in state.objects and len(state.objects[(x, y)].ingredients) == 3:
            return False

        return True

    def IsSoupReadyToStart(self, state, x, y):
        for i in range(0, len(self.stoves)):
            if (self.stoves[i][0] == x and self.stoves[i][1] == y):  # if we found the correct stove
                if (x, y) in state.objects and len(state.objects[(x, y)].ingredients) == 3 and state.objects[
                    (x, y)].is_cooking == False:  # here we can check if it's the correct order
                    if state.objects[(x, y)].is_ready:
                        return False
                    return True

        return False

    def takeAction(self, state):

        if random.randint(1, 15) == 1:
            return Action.STAY

        self.computePathingFrom(self.x, self.y, False, False)

        if self.heldObject == None and self.IsSoupReadyToStart(state, self.front()[0], self.front()[1]):
            # print("doing here1")
            return Action.INTERACT

        # print("Behavior is: " + self.behaviour)

        if self.behaviour == "BringOnion":

            if (self.heldObject == None):
                if self.IsSoupReadyToStart(state, self.front()[0], self.front()[1]):
                    # print("doing here2")
                    return Action.INTERACT

                bestScore = 0
                targetX = None
                targetY = None
                for y in range(0, len(self.map)):
                    for x in range(0, len(self.map[0])):
                        newScore = self.TakeOnionScore(x, y, state)
                        # print(str(x) + " " + str(y) + " Obtained " + str(newScore))
                        if newScore > bestScore:
                            bestScore = newScore
                            targetX = x
                            targetY = y
                if self.front()[0] == targetX and self.front()[1] == targetY:
                    # print("doing here3")
                    return Action.INTERACT

                for i in range(0, len(self.dirs)):
                    if self.x + self.dirs[i][0] == targetX and self.y + self.dirs[i][1] == targetY:
                        return self.actionFromDir(i)

                if (targetX != None):
                    return self.oppositeActionFromDir(self.getDirectionTo(targetX, targetY))

            if (self.heldObject != None):
                if (self.heldObject.name == "onion"):
                    bestScore = 0
                    targetX = None
                    targetY = None
                    for y in range(0, len(self.map)):
                        for x in range(0, len(self.map[0])):
                            newScore = self.GoodOnionPlacementScore(x, y, state)
                            # print("Obtained " + str(newScore))
                            if newScore > bestScore:
                                bestScore = newScore
                                targetX = x
                                targetY = y
                    if self.front()[0] == targetX and self.front()[1] == targetY:
                        # print("doing here4")
                        return Action.INTERACT

                    for i in range(0, len(self.dirs)):
                        if self.x + self.dirs[i][0] == targetX and self.y + self.dirs[i][1] == targetY:
                            return self.actionFromDir(i)

                    if (targetX != None):
                        return self.oppositeActionFromDir(self.getDirectionTo(targetX, targetY))

        if self.behaviour == "BringPlate":
            targetX = -1
            targetY = -1
            if self.heldObject == None:

                bestDist = 10000
                self.computePathingFrom(self.x, self.y, False, False)
                for j in range(0, len(self.dishDispensers)):
                    x = self.dishDispensers[j][0]
                    y = self.dishDispensers[j][1]
                    if (self.distances[y][x] < bestDist):
                        targetX = x
                        targetY = y
                        bestDist = self.distances[y][x]
                        break
                for j in range(0, len(self.dishes)):
                    x = self.dishes[j].position[0]
                    y = self.dishes[j].position[1]
                    if (self.distances[y][x] - 5 < bestDist):
                        targetX = x
                        targetY = y
                        bestDist = self.distances[y][x] - 5
                        break

            if self.heldObject != None and self.heldObject.name == "dish":
                self.computePathingFrom(self.x, self.y, False, False)
                bestDist = 10000
                for i in range(0, len(self.counters)):
                    x = self.counters[i][0]
                    y = self.counters[i][1]
                    if (x, y) not in state.objects and self.distances[y][x] * 4 + self.counters[i][4] < bestDist:
                        bestDist = self.distances[y][x]
                        targetX = x
                        targetY = y

            if self.heldObject != None and self.heldObject.name == "onion":
                self.computePathingFrom(self.x, self.y, False, False)
                bestDist = 10000
                for i in range(0, len(self.counters)):
                    x = self.counters[i][0]
                    y = self.counters[i][1]
                    if (x, y) not in state.objects and self.distances[y][x] < bestDist:
                        bestDist = self.distances[y][x]
                        targetX = x
                        targetY = y

            if self.front()[0] == targetX and self.front()[1] == targetY:
                return Action.INTERACT

            for i in range(0, len(self.dirs)):
                if self.x + self.dirs[i][0] == targetX and self.y + self.dirs[i][1] == targetY:
                    return self.actionFromDir(i)

            return self.oppositeActionFromDir(self.getDirectionTo(targetX, targetY))

        if self.behaviour == "PickUpSoup":
            # self.printDistances()
            [timeTilSoup, soupLocation, params] = self.TimeToPickUpAndDeliverSoup(state, self.x, self.y,
                                                                                  self.heldObject)
            self.computePathingFrom(self.x, self.y, False, False)
            # print ([timeTilSoup, soupLocation, params])

            if self.heldObject != None and self.heldObject.name == "onion":
                targetX = params[0][1][0]
                targetY = params[0][1][1]

            if self.heldObject == None:
                targetX = params[1][1][0]
                targetY = params[1][1][1]

            if self.heldObject != None and self.heldObject.name == "dish":
                targetX = soupLocation[0]
                targetY = soupLocation[1]

            # print ("Target for delivery is " + str(targetX) + " " + str(targetY))

            if self.front()[0] == targetX and self.front()[1] == targetY:
                return Action.INTERACT

            for i in range(0, len(self.dirs)):
                if self.x + self.dirs[i][0] == targetX and self.y + self.dirs[i][1] == targetY:
                    return self.actionFromDir(i)
            # print("we got here...")
            return self.oppositeActionFromDir(self.getDirectionTo(targetX, targetY))

        if self.behaviour == "DeliverSoup":
            bestScore = 0
            targetX = None
            targetY = None
            for y in range(0, len(self.map)):
                for x in range(0, len(self.map[0])):
                    newScore = self.GoodSoupPlacementScore(x, y, state)
                    # print("Obtained " + str(newScore))
                    if newScore > bestScore:
                        bestScore = newScore
                        targetX = x
                        targetY = y
            if self.front()[0] == targetX and self.front()[1] == targetY:
                return Action.INTERACT

            for i in range(0, len(self.dirs)):
                if self.x + self.dirs[i][0] == targetX and self.y + self.dirs[i][1] == targetY:
                    return self.actionFromDir(i)

            if (targetX != None):
                return self.oppositeActionFromDir(self.getDirectionTo(targetX, targetY))
        return Action.STAY
        # if(self.heldObject.name == "onion"):

    def checkObjects(self, state):

        self.soups = []
        self.startedSoups = []
        self.dishes = []
        self.onions = []
        for i in range(0, len(self.counters)):
            self.counters[i][2] = False
            self.counters[i][3] = 100000
            self.counters[i][4] = 100000

        for i in range(0, len(state.objects.keys())):
            obj = state.objects[list(state.objects.keys())[i]]
            if (obj.name == "soup"):
                self.soups.append(obj)
                if (obj.is_cooking or obj.is_ready):
                    self.startedSoups.append(obj)
            if (obj.name == "dish"):
                self.dishes.append(obj)
            if (obj.name == "onion"):
                self.onions.append(obj)

            for i in range(0, len(self.counters)):
                if (self.counters[i][0] == obj.position[0] and self.counters[i][1] == obj.position[1]):
                    self.counters[i][2] = True

        for i in range(0, len(self.counters)):
            if (self.counters[i][2] == True):
                continue
            self.computePathingFrom(self.counters[i][0], self.counters[i][1], True)

            bestDelivery = 100000
            for j in range(0, len(self.delivery)):
                dist = self.distances[self.delivery[j][1]][self.delivery[j][0]]
                if dist < bestDelivery:
                    bestDelivery = dist
            self.counters[i][3] = bestDelivery

            bestDelivery = 100000
            for j in range(0, len(self.stoves)):
                dist = self.distances[self.stoves[j][1]][self.stoves[j][0]]
                if dist < bestDelivery:
                    bestDelivery = dist
            self.counters[i][4] = bestDelivery

        # obj = state.objects["onion"]
        # print(obj)

    def action(self, states, actions):
        state = states[-1]
        self.initialization()
        self.takeVariablesFromState(state)

        # for x in range (0, len(self.map)):
        # print (self.map[x])

        # print("Tking action knowing " + str(state))

        self.computePathingFrom(self.x, self.y)
        self.checkObjects(state)

        # print("STARTED SOUPS:")
        # print(self.startedSoups)

        self.decideMyBehaviour(state)

        action = self.takeAction(state)
        # print("Doing " + str(action))

        return action, {}

        [action] = random.sample(
            [
                Action.STAY,
                Direction.NORTH,
                Direction.SOUTH,
                Direction.WEST,
                Direction.EAST,
                Action.INTERACT,
            ],
            1,
        )
        return action, {}

    def reset(self):
        pass

