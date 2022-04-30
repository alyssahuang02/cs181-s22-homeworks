# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import matplotlib.pyplot as plt

# uncomment this for animation
# from SwingyMonkey import SwingyMonkey

# uncomment this for no animation
from SwingyMonkeyNoAnimation import SwingyMonkey


X_BINSIZE = 200
Y_BINSIZE = 100
X_SCREEN = 1400
Y_SCREEN = 900


class Learner(object):
    """
    This agent jumps randomly.
    """

    def __init__(self, alpha=0.2, gamma=0.9, starting_epsilon=0.1):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.epoch = 0

        self.alpha = alpha
        self.gamma = gamma
        self.starting_epsilon = starting_epsilon

        # We initialize our Q-value grid that has an entry for each action and state.
        # (action, rel_x, rel_y)
        self.Q = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE))

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def discretize_state(self, state):
        """
        Discretize the position space to produce binned features.
        rel_x = the binned relative horizontal distance between the monkey and the tree
        rel_y = the binned relative vertical distance between the monkey and the tree
        """

        rel_x = int((state["tree"]["dist"]) // X_BINSIZE)
        rel_y = int((state["tree"]["top"] - state["monkey"]["top"]) // Y_BINSIZE)
        return (rel_x, rel_y)

    def action_callback(self, state):
        """
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        """
        self.epoch += 1
        # Hyperparameters
        alpha = self.alpha
        gamma = self.gamma
        # epsilon = max(self.starting_epsilon - 0.001*self.epoch, 0)
        epsilon = self.starting_epsilon

        # 1. Discretize 'state' to get your transformed 'current state' features.
        rel_x, rel_y = self.discretize_state(state)

        # 2. Perform the Q-Learning update using 'current state' and the 'last state'.
        if self.last_state:
            cur_max = np.max(self.Q[:, rel_x, rel_y])
            old_rel_x, old_rel_y = self.discretize_state(self.last_state)
            old_Q = self.Q[self.last_action, old_rel_x, old_rel_y]
            updated_Q = (1 - alpha) * old_Q + alpha * (self.last_reward + gamma * cur_max)
            self.Q[self.last_action, old_rel_x, old_rel_y] = updated_Q

        # 3. Choose the next action using an epsilon-greedy policy.
        if npr.uniform(0, 1) < epsilon:
            # Explore
            if npr.rand() < 0.5:
                action = 0
            else:
                action = 1
        else:
            # Exploit
            action = np.argmax(self.Q[:, rel_x, rel_y], axis=0)
        
        new_action = action
        new_state = state

        self.last_action = new_action
        self.last_state = new_state

        return self.last_action

    def reward_callback(self, reward):
        """This gets called so you can see what reward you get."""

        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=100):
    """
    Driver function to simulate learning by having the agent play a sequence of games.
    """
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,  # Don't play sounds.
                             text="Epoch %d" % (ii),  # Display the epoch on screen.
                             tick_length=t_len,  # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':
    # Select agent.

    alphas = np.linspace(0.1, 0.3, num=3)
    gammas = np.linspace(0.8, 1, num=3)
    epsilons = np.linspace(0.1, 0.3, num=3)
    num_trials = 10

    # for alpha in alphas:
    #     for gamma in gammas:
    #         for epsilon in epsilons:
    #             agent = Learner(alpha=alpha, gamma=gamma, starting_epsilon=epsilon)
    #             maxes = 0
    #             for trial in range(num_trials):
    #                 hist = []
    #                 run_games(agent, hist, 100, 100)
    #                 maxes += max(hist)
    #             print('alpha: ' + str(alpha) + 
    #             ', gamma: ' + str(gamma) + ', epsilon: ' + str(epsilon) + 
    #             ", avg max score: " + str(maxes/num_trials))

    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games. You can update t_len to be smaller to run it faster.
    run_games(agent, hist, 100, 100)
    print(hist)

    plt.plot(range(100), hist)
    plt.title("Scores over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig('scores_epoch.png')
    plt.show()

    # Save history. 
    np.save('hist', np.array(hist))
