from PacmanEnvAbs import PacmanEnvAbs
from game import Directions



class PacmanEnv(PacmanEnvAbs):
    def _get_reward(self, pacman_successor):#pacman_successor is the state after pacman moved but before the other agent moved
        return super(PacmanEnv, self)._get_reward(pacman_successor)
        
    def _get_observations(self):
        return super(PacmanEnv, self)._get_observations()

    def flatten_obs(self, s):
        return super(PacmanEnv, self).flatten_obs(s)

    
        