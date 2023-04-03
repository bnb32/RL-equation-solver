"""
setup.py
"""
import os
from codecs import open
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from subprocess import check_call
import shlex
from warnings import warn

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    readme = f.read()

with open(os.path.join(here, "requirements.txt")) as f:
    install_requires = f.readlines()

version_file = os.path.join(here, "rl_equation_solver", "version.py")
with open(version_file, encoding="utf-8") as f:
    version = f.read()

version = version.split('=')[-1].strip().strip('"').strip("'")


class PostDevelopCommand(develop):
    """
    Class to run post setup commands
    """

    def run(self):
        """
        Run method that tries to install pre-commit hooks
        """
        try:
            check_call(shlex.split("pre-commit install"))
        except Exception as e:
            warn("Unable to run 'pre-commit install': {}"
                 .format(e))

        develop.run(self)


setup(
    name="RL-equation-solver",
    version=version,
    description="RL equation solver using deep Q networks implemented in "
                "PyTorch",
    long_description=readme,
    author="Brandon Benton, Kevin O'Keefe",
    author_email="bnb32@cornell.edu, kevin.p.okeeffe@gmail.com",
    entry_points={"console_scripts": []},
    url="https://github.com/RL-equation-solver",
    packages=find_packages(),
    package_dir={"RL-equation-solver": "rl_equation_solver"},
    include_package_data=True,
    license="MIT License",
    zip_safe=False,
    keywords="RL-equation-solver",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    test_suite="tests",
    python_requires='>=3.7',
    install_requires=install_requires,
    extras_require={
        "dev": ["flake8", "pre-commit", "pylint"],
    },
    cmdclass={"develop": PostDevelopCommand},
)
