.PHONY: init test clean format lint env-create env-remove env-recreate

# Environment name from environment.yml
ENV_NAME := mushkil-viz

# Detect OS and set the correct command to check if environment exists
ifeq ($(OS),Windows_NT)
	CONDA_CMD := conda
	CHECK_ENV := $(CONDA_CMD) env list | findstr /B /L "$(ENV_NAME)"
else
	CONDA_CMD := conda
	CHECK_ENV := $(CONDA_CMD) env list | grep -E "^$(ENV_NAME) "
endif

# Remove environment if it exists
env-remove:
	@echo "Checking if environment exists..."
	@if $(CHECK_ENV) >/dev/null 2>&1; then \
		echo "Removing existing environment $(ENV_NAME)..."; \
		$(CONDA_CMD) env remove -n $(ENV_NAME) -y; \
	else \
		echo "Environment $(ENV_NAME) does not exist."; \
	fi

# Create environment
env-create:
	@echo "Creating environment $(ENV_NAME)..."
	$(CONDA_CMD) env create -f environment.yml

# Recreate environment from scratch
env-recreate: env-remove env-create

# Initialize project (recreate environment and install pre-commit hooks)
init: env-recreate
	$(CONDA_CMD) run -n $(ENV_NAME) pre-commit install

# Update environment
update-env:
	$(CONDA_CMD) env update -n $(ENV_NAME) -f environment.yml

# Run tests
test:
	$(CONDA_CMD) run -n $(ENV_NAME) pytest tests/ -v

# Clean up Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".tox" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +

# Format code
format:
	$(CONDA_CMD) run -n $(ENV_NAME) black src/ tests/ examples/
	$(CONDA_CMD) run -n $(ENV_NAME) isort src/ tests/ examples/

# Run linting
lint:
	$(CONDA_CMD) run -n $(ENV_NAME) flake8 src/ tests/ examples/
	$(CONDA_CMD) run -n $(ENV_NAME) black --check src/ tests/ examples/
	$(CONDA_CMD) run -n $(ENV_NAME) isort --check-only src/ tests/ examples/

# Run all checks (lint and test)
check: lint test 