"""Model configuration class"""


class Config:
    """Model configuration"""

    # BATCH_SIZE is the number of Experience sampled from the replay buffer
    BATCH_SIZE = 64
    # GAMMA is the discount factor
    GAMMA = 0.99
    # EPSILON_START is the starting value of epsilon
    EPSILON_START = 0.9
    # EPSILON_END is the final value of epsilon
    EPSILON_END = 0.05
    # EPSILON_DECAY controls the rate of exponential decay of epsilon, higher
    # means a slower decay
    EPSILON_DECAY = 20
    # TAU is the update rate of the target network
    TAU = 0.005
    # LR is the learning rate of the AdamW optimizer
    LR = 1e-4
    # the hidden layers in the DQN
    HIDDEN_SIZE = 256
    # memory capacity
    MEM_CAP = 10000
    # reset after this many steps with constant loss
    RESET_STEPS = 100
    # state vec max size
    VEC_DIM = 4096
