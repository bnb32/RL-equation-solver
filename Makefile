SHELL=/bin/bash
LINT_PATHS=./rl_equation_solver/ ./tests/ ./setup.py

pytest:
	python -m pytest --cov-config .coveragerc --cov-report html --cov-report term --cov=. -v --color=yes -m "not expensive"

pytype:
	pytype -j auto

mypy:
	mypy --install-types --non-interactive ${LINT_PATHS}

missing-annotations:
	mypy --disallow-untyped-defs --ignore-missing-imports ${LINT_PATHS}

missing docstrings:
	pylint -d R,C,W,E -e C0116 rl_equation_solver -j 4

type: pytype mypy

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://www.flake8rules.com/
	ruff ${LINT_PATHS} --config='./pyproject.toml' --show-source
	# exit-zero treats all errors as warnings.
	ruff ${LINT_PATHS} #--exit-zero

format:
	# Sort imports
	isort ${LINT_PATHS}
	# Reformat using black
	black ${LINT_PATHS}

check-codestyle:
	# Sort imports
	isort --check ${LINT_PATHS}
	# Reformat using black
	black --check ${LINT_PATHS}

commit-checks: format type lint

doc:
	cd docs && make html

spelling:
	cd docs && make spelling

clean:
	cd docs && make clean

# Build docker images
# If you do export RELEASE=True, it will also push them
docker: docker-cpu docker-gpu

docker-cpu:
	./scripts/build_docker.sh

docker-gpu:
	USE_GPU=True ./scripts/build_docker.sh

# PyPi package release
release:
	python setup.py sdist
	python setup.py bdist_wheel
	twine upload dist/*

# Test PyPi package release
test-release:
	python setup.py sdist
	python setup.py bdist_wheel
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: clean spelling doc lint format check-codestyle commit-checks
