##############################
Welcome to RL-Equation-Solver!
##############################

This software is an equation solver using Deep Q Networks implemented in PyTorch.

Installing RL-equation-solver
=============================

NOTE: The installation instruction below assume that you have python installed
on your machine and are using `conda <https://docs.conda.io/en/latest/index.html>`_
as your package/environment manager.

Option 1: Clone repo (recommended for developers)
-------------------------------------------------

1. from home dir, ``git clone git@github.com:bnb32/RL-equation-solver.git``

2. Create ``rl_solver`` environment and install package
    1) Create a conda env: ``conda create -n rl_solver``
    2) Run the command: ``conda activate rl_solver``
    3) ``cd`` into the repo cloned in 1.
    4) Prior to running ``pip`` below, make sure the branch is correct (install
       from main!)
    5) Install ``rl_solver`` and its dependencies by running:
       ``pip install .`` (or ``pip install -e .`` if running a dev branch
       or working on the source code)
