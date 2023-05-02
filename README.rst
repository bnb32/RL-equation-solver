##############################
Welcome to RL-Equation-Solver!
##############################

.. raw:: html

    <p align="center">
        <a href="https://bnb32.github.io/RL-equation-solver/"><img alt="Documentation" src="https://github.com/bnb32/RL-equation-solver/workflows/Documentation/badge.svg"></a>
        <a href="https://github.com/bnb32/RL-equation-solver/actions?query=workflow%3A%22CI%22"><img alt="CI" src="https://github.com/bnb32/RL-equation-solver/workflows/CI/badge.svg"></a>
        <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
        <a href="https://codecov.io/gh/bnb32/RL-equation-solver"><img alt="codecov" src="https://codecov.io/gh/bnb32/RL-equation-solver/branch/main/graph/badge.svg"></a>
        <a href="https://opensource.org/licenses/MIT"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-purple.svg"></a>
    </p>

This software is an equation solver using various neural networks implemented in PyTorch.

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

3. To use pytorch with cuda make sure you have cuda installed and then run the following command:

    ``pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html``