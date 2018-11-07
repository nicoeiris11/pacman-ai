from obligatorio.PacmanEnv import PacmanEnv
from obligatorio.miniMax import getAction
from obligatorio.qlearning import QLearner
import random
import math
import numpy as np


def miniMax():
    env = PacmanEnv()
    layouts = [
        # "custom1",
        # "custom2",
        # "capsule1",
        #"capsuleClassic",
        # "contestClassic",
        #"mediumClassic",
        "minimaxClassic",
        "openClassic",
        # "originalClassic",
        # "smallClassic",
        #"testClassic",
        #"trappedClassic",
        # "trickyClassic",
        "mediumGrid",
        "smallGrid"
    ]

    for l in layouts:
        for i in range(2):
            done = False
            _, info = env.reset(layout_name=l, enable_render=True)
            game_state = info["internal_state"]
            while (not(done)):
                a = getAction(game_state, 10)
                _, _, done, info = env.step(a)
                game_state = info["internal_state"]
                if (info["score"] < -1500):
                    done = True
            print(l, i, "win" if info["win"]
                  else "lose", "score", info["score"])


def qlearning(use_softmax):
    env = PacmanEnv(view_distance=(2, 2))
    layouts = [
        # "custom1",
        # "custom2",
        # "capsule1",
        # "capsuleClassic", }
        # "contestClassic",
        # "mediumClassic",
        "minimaxClassic",
        "openClassic",
        #"originalClassic",
        #"smallClassic",
        "testClassic",
        "trappedClassic",
        # "trickyClassic",
        "mediumGrid",
        "smallGrid"
    ]
    learner = QLearner.load()
    if (learner is None):
        learner = QLearner(use_softmax=use_softmax)
    for i in range(1):
        learner.learn_n(env, iterations=6000,
                        layouts=layouts, verbose=False, save_mod=500, info_mod=500, render_mod=None)#(len(layouts)-1)*20)
        learner.run(env, layouts=layouts,
                    iterations=len(layouts), render_mod=1)
    learner.save()


if __name__ == '__main__':
    miniMax()
    qlearning(True)
    #qlearning(False)
