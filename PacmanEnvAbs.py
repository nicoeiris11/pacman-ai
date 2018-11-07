import gym
from gym import error, spaces, utils
from gym.utils import seeding
from functools import reduce
import copy
import numpy as np
from pacman import readCommand, ClassicGameRules
from game import Game, Directions
import sys
import random
from keyboardAgents import KeyboardAgent
from ghostAgents import DirectionalGhost, RandomGhost
import textDisplay
from layout import getLayout
import graphicsDisplay

_directions = {Directions.NORTH: (0, 1),
               Directions.SOUTH: (0, -1),
               Directions.EAST:  (1, 0),
               Directions.WEST:  (-1, 0),
               Directions.STOP:  (0, 0)}

_directionsAsList = list(_directions.items())


class GameExtended(Game):
    def init(self):
        # try:
        #     import boinc
        #     _BOINC_ENABLED = True
        # except:
        _BOINC_ENABLED = False

        """
    Main control loop for game play.
    """
        self.display.initialize(self.state.data)
        self.numMoves = 0

        # self.display.initialize(self.state.makeObservation(1).data)
        # inform learning agents of the game start
        for i in range(len(self.agents)):
            agent = self.agents[i]
            if not agent:
                # this is a null agent, meaning it failed to load
                # the other team wins
                self._agentCrash(i, quiet=True)
                return
            if ("registerInitialState" in dir(agent)):
                self.mute()
                if self.catchExceptions:
                    try:
                        timed_func = TimeoutFunction(
                            agent.registerInitialState, int(self.rules.getMaxStartupTime(i)))
                        try:
                            start_time = time.time()
                            timed_func(self.state.deepCopy())
                            time_taken = time.time() - start_time
                            self.totalAgentTimes[i] += time_taken
                        except TimeoutFunctionException:
                            print("Agent %d ran out of time on startup!" % i)
                            self.unmute()
                            self.agentTimeout = True
                            self._agentCrash(i, quiet=True)
                            return
                    except Exception as data:
                        self.unmute()
                        self._agentCrash(i, quiet=True)
                        return
                else:
                    agent.registerInitialState(self.state.deepCopy())
                # TODO: could this exceed the total time
                self.unmute()

    def step(self, pacman_action, render):
        _BOINC_ENABLED = False
        agentIndex = self.startingIndex
        numAgents = len(self.agents)

        while not self.gameOver and agentIndex < numAgents:
            # Fetch the next agent
            agent = self.agents[agentIndex]
            move_time = 0
            skip_action = False
            # Generate an observation of the state
            if False and 'observationFunction' in dir(agent):
                pass
                # self.mute()
                # if self.catchExceptions:
                #   try:
                #     timed_func = TimeoutFunction(agent.observationFunction, int(self.rules.getMoveTimeout(agentIndex)))
                #     try:
                #       start_time = time.time()
                #       observation = timed_func(self.state.deepCopy())
                #     except TimeoutFunctionException:
                #       skip_action = True
                #     move_time += time.time() - start_time
                #     self.unmute()
                #   except Exception as data:
                #     self.unmute()
                #     self._agentCrash(agentIndex, quiet=True)
                #     return
                # else:
                #   observation = agent.observationFunction(self.state.deepCopy())
                # self.unmute()
            else:
                observation = self.state.deepCopy()

            # Solicit an action
            action = None
            self.mute()
            if False and self.catchExceptions:
                pass
                # try:
                #   timed_func = TimeoutFunction(agent.getAction, int(self.rules.getMoveTimeout(agentIndex)) - int(move_time))
                #   try:
                #     start_time = time.time()
                #     if skip_action:
                #       raise TimeoutFunctionException()
                #     action = timed_func( observation )
                #   except TimeoutFunctionException:
                #     print("Agent %d timed out on a single move!" % agentIndex)
                #     self.agentTimeout = True
                #     self.unmute()
                #     self._agentCrash(agentIndex, quiet=True)
                #     return

                #   move_time += time.time() - start_time

                #   if move_time > self.rules.getMoveWarningTime(agentIndex):
                #     self.totalAgentTimeWarnings[agentIndex] += 1
                #     print("Agent %d took too long to make a move! This is warning %d" % (agentIndex, self.totalAgentTimeWarnings[agentIndex]))
                #     if self.totalAgentTimeWarnings[agentIndex] > self.rules.getMaxTimeWarnings(agentIndex):
                #       print("Agent %d exceeded the maximum number of warnings: %d" % (agentIndex, self.totalAgentTimeWarnings[agentIndex]))
                #       self.agentTimeout = True
                #       self.unmute()
                #       self._agentCrash(agentIndex, quiet=True)

                #   self.totalAgentTimes[agentIndex] += move_time
                #   #print "Agent: %d, time: %f, total: %f" % (agentIndex, move_time, self.totalAgentTimes[agentIndex])
                #   if self.totalAgentTimes[agentIndex] > self.rules.getMaxTotalTime(agentIndex):
                #     print("Agent %d ran out of time! (time: %1.2f)" % (agentIndex, self.totalAgentTimes[agentIndex]))
                #     self.agentTimeout = True
                #     self.unmute()
                #     self._agentCrash(agentIndex, quiet=True)
                #     return
                #   self.unmute()
                # except Exception as data:
                #   self.unmute()
                #   self._agentCrash(agentIndex)
                #   return
            else:
                action = agent.getAction(
                    observation) if agentIndex is not 0 else pacman_action
            self.unmute()

            # Execute the action
            self.moveHistory.append((agentIndex, action))
            if False and self.catchExceptions:
                pass
                # try:
                #   self.state = self.state.generateSuccessor( agentIndex, action )
                # except Exception as data:
                #   self._agentCrash(agentIndex)
                #   return
            else:
                self.state = self.state.generateSuccessor(agentIndex, action)
                if agentIndex is 0:
                    pacman_successor = self.state

            # Change the display
            ###idx = agentIndex - agentIndex % 2 + 1
            ###self.display.update( self.state.makeObservation(idx).data )

            # Allow for game specific conditions (winning, losing, etc.)
            self.rules.process(self.state, self)
            # Track progress
            if agentIndex == numAgents + 1:
                self.numMoves += 1
            # Next agent
            agentIndex = (agentIndex + 1)  # % numAgents

            if _BOINC_ENABLED:
                boinc.set_fraction_done(self.getProgress())
            self.render()

        # # inform a learning agent of the game result
        # for agent in self.agents:
        #   if "final" in dir( agent ) :
        #     try:
        #       self.mute()
        #       agent.final( self.state )
        #       self.unmute()
        #     except Exception as data:
        #       if not self.catchExceptions: raise
        #       self.unmute()
        #       print("Exception",data)
        #       self._agentCrash(agent.index)
        #       return
        if self.gameOver:
            self.display.finish()

        return pacman_successor

    def render(self):
        self.display.update(self.state.data)

    def close(self):
        self.display.finish()


class ClassicGameRulesExtended(ClassicGameRules):
    def newGame(self, layout, pacmanAgent, ghostAgents, display, quiet=False, catchExceptions=False):
        from pacman import GameState
        agents = [pacmanAgent] + ghostAgents[:layout.getNumGhosts()]
        initState = GameState()
        initState.initialize(layout, len(ghostAgents))
        game = GameExtended(agents, display, self,
                            catchExceptions=catchExceptions)
        game.state = initState
        self.initialState = initState.deepCopy()
        self.quiet = quiet
        return game


class PacmanEnvAbs(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, enable_render=False, layout_name="mediumClassic", view_distance = (2, 2)):
        self.layouts = dict()
        self.layout_name=layout_name
        self.pacman = KeyboardAgent()
        self.ghosts = [RandomGhost(i+1) if i % 2 ==
                       0 else DirectionalGhost(i+1) for i in range(20)]
        frameTime = 0.03

        textDisplay.SLEEP_TIME = frameTime
        self.display_text = textDisplay.PacmanGraphics()
        self.display_graphics = graphicsDisplay.PacmanGraphics(
            1.0, frameTime=frameTime)

        self.beQuiet = True
        self.game = None
        self.view_distance = view_distance
        self.textGraphics = False
        self.reset(enable_render=enable_render, layout_name=layout_name)

    def init_game(self, layout, pacman, ghosts, display, catchExceptions=False, timeout=30):
        rules = ClassicGameRulesExtended(timeout)
        if self.beQuiet and not self.enable_render:
            # Suppress output and graphics
            gameDisplay = textDisplay.NullGraphics()
            rules.quiet = self.beQuiet
        else:
            gameDisplay = display
            rules.quiet = self.beQuiet
            # rules.quiet = False
        game = rules.newGame(layout, pacman, ghosts,
                             gameDisplay, self.beQuiet, catchExceptions)
        game.init()
        return game

    def reset(self, enable_render=False, layout_name=None):
        self.enable_render = enable_render
        self.layout_name = layout_name if layout_name is not None else self.layout_name
        if (self.game):
            self.game.close()
        # c = readCommand(["-l", self.layout_name])

        self.layouts[layout_name] = self.layouts.get(
            layout_name, getLayout(layout_name))

        if (self.textGraphics):
            display = self.display_text
        else:
            display = self.display_graphics

        self.game = self.init_game(
            self.layouts[layout_name], self.pacman, self.ghosts, display)

        return self._get_observations(), {"internal_state": self.game.state}

    # def seed(self, seed=None):
    #     self.np_random = np.random.RandomState(seed)
    #     return [seed]

    def step(self, action):
        if (not(self._check_action(action))):
            return self._get_observations(), 0, self.game.gameOver, {"internal_state": self.game.state, "win": self.game.state.data._win, "score": self.game.state.getScore()}
        pacman_successor = self.game.step(action, render=self.render)

        reward = self._get_reward(pacman_successor)
        obs = self._get_observations()
        
        return obs, reward, self.game.gameOver, {"internal_state": self.game.state, "win": self.game.state.data._win, "score": self.game.state.getScore()}

    def _get_reward(self, pacman_successor):#pacman_successor is the state after pacman moved but before the other agent moved
        if self.game.gameOver:
            reward = self.game.state.getScore(
            ) if self.game.state.data._win else self.game.state.getScore() - 1000
        else:
            reward = pacman_successor.data.scoreChange
        return reward


    def get_legal_actions(self):
        return self.game.state.getLegalActions()

    def _check_action(self, action):
        if (action not in map(lambda x: x[0], _directionsAsList)):
            raise Exception('Action not in action_space')

        return action in self.game.state.getLegalActions()

    def _get_observations(self):
        """ (x,y) are positions on a Pacman map with x horizontal,
  y vertical and the origin (0,0) in the bottom left corner."""
        food = np.array(self.game.state.data.food.data)
        walls = np.array(self.game.state.data.layout.walls.data)
        map_shape = walls.shape
        capsules = self.game.state.data.capsules
        pacman_pos = self.game.state.data.agentStates[0].configuration.pos

        gosts_pos = list(map(lambda agent: agent.configuration.pos,
                             self.game.state.data.agentStates[1:]))
        gosts_scared = list(
            map(lambda agent: agent.scaredTimer > 0, self.game.state.data.agentStates[1:]))

        """
            0: empty,
            1: wall,
            2: food,
            3: capsules,
            4: ghost,
            5: scared ghost,
            6: pacman
        """

        view_slices = ((max(pacman_pos[0]-self.view_distance[0], 0), min(pacman_pos[0]+self.view_distance[0]+1, map_shape[0])),
                       (max(pacman_pos[1]-self.view_distance[1], 0), min(pacman_pos[1]+self.view_distance[1]+1, map_shape[1])))

        def select(l):
            return l[view_slices[0][0]:view_slices[0][1], view_slices[1][0]:view_slices[1][1]]

        obs = np.vectorize(lambda v: 1 if v else 0)(select(walls))
        obs = obs + np.vectorize(lambda v: 2 if v else 0)(select(food))

        def pos_to_relative_pos(pos):
            if (pos[0] < view_slices[0][0] or view_slices[0][1] <= pos[0]
                    or pos[1] < view_slices[1][0] or view_slices[1][1] <= pos[1]):
                return None
            else:
                return pos[0]-view_slices[0][0], pos[1]-view_slices[1][0]

        for c_relative_pos in filter(lambda x: x is not None, map(pos_to_relative_pos, capsules)):
            obs[c_relative_pos[0], c_relative_pos[1]] = 3

        for i, g_relative_pos in enumerate(map(pos_to_relative_pos, gosts_pos)):
            if (g_relative_pos is not None):
                obs[int(g_relative_pos[0]), int(g_relative_pos[1])
                    ] = 5 if gosts_scared[i] else 4

        pacman_relative_pos = pos_to_relative_pos(pacman_pos)

        obs[pacman_relative_pos[0], pacman_relative_pos[1]] = 6

        obs[0, 0] = 2 if np.any(
            food[0:pacman_pos[0]+1, 0:pacman_pos[1]+1]) else 0
        obs[obs.shape[0]-1,
            0] = 2 if np.any(food[pacman_pos[0]:map_shape[0], 0:pacman_pos[1]+1])else 0

        obs[0, obs.shape[1] -
            1] = 2 if np.any(food[0:pacman_pos[0]+1, pacman_pos[1]:map_shape[0]]) else 0
        obs[obs.shape[0]-1, obs.shape[1]-1] = 2 if np.any(
            food[pacman_pos[0]:map_shape[0], pacman_pos[1]:map_shape[0]]) else 0

        # print(np.transpose(obs)[::-1, :])

        return obs

    def close(self):
        if (self.game is not None):
            self.game.close()

    def render(self, mode='human', close=False):
        obs = self._get_observations()
        if (not(close)):
            print(np.transpose(obs)[::-1, :])

    def flatten_obs(self, s):
        return tuple(s.flatten())
