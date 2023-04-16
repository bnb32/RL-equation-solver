"""Default Model configuration"""

DefaultConfig = {

    # BATCH_SIZE is the number of Experience sampled from the replay buffer
    'batch_size': 32,
    # GAMMA is the discount factor
    'gamma': 0.99,
    # EPSILON_START is the starting value of epsilon
    'eps_start': 0.9,
    # EPSILON_END is the final value of epsilon
    'eps_end': 0.05,
    # EPS_DECAY_STEPS controls the rate of exponential decay of epsilon, higher
    # means a slower decay
    'eps_decay_steps': 1000,
    # optional fixed value for epsilon decay. 0 turns off the decay and uses
    # the EPSILON_END value as the fixed threshold. None enables the decay.
    'eps_decay': None,
    # TAU is the update rate of the target network
    'tau': 0.005,
    # LR is the learning rate of the AdamW optimizer
    'learning_rate': 5e-5,
    # the hidden layers in the DQN
    'hidden_size': 64,
    # memory capacity
    'memory_cap': 10000,
    # reset after this many steps with constant loss
    'reset_steps': 100,
    # state vec max size
    'state_dim': 256,
    # max number of node features
    'feature_num': 100,
    # gradient clipping value
    'grad_clip': 100
}
