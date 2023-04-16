"""Model configuration class"""


class Config:
    """Model configuration"""

    # BATCH_SIZE is the number of Experience sampled from the replay buffer
    BATCH_SIZE = 32
    # GAMMA is the discount factor
    GAMMA = 0.99
    # EPSILON_START is the starting value of epsilon
    EPSILON_START = 0.9
    # EPSILON_END is the final value of epsilon
    EPSILON_END = 0.05
    # EPS_DECAY_STEPS controls the rate of exponential decay of epsilon, higher
    # means a slower decay
    EPS_DECAY_STEPS = 1000
    # optional fixed value for epsilon decay. 0 turns off the decay and uses
    # the EPSILON_END value as the fixed threshold. None enables the decay.
    EPS_DECAY = None
    # TAU is the update rate of the target network
    TAU = 0.005
    # LR is the learning rate of the AdamW optimizer
    LR = 5e-5
    # the hidden layers in the DQN
    HIDDEN_SIZE = 64
    # memory capacity
    MEM_CAP = 10000
    # reset after this many steps with constant loss
    RESET_STEPS = 100
    # state vec max size
    VEC_DIM = 256
    # max number of node features
    FEATURE_NUM = 100
    # gradient clipping value
    GRAD_CLIP = 100
