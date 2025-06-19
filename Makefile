.PHONY: init test clean format lint env-create env-remove env-recreate

CONDA_CMD ?= conda
# Environment name from environment.yml
ENV_NAME := mushkil

# Detect OS and set the correct command to check if environment exists
ifeq ($(OS),Windows_NT)
	CHECK_ENV := $(CONDA_CMD) env list | findstr /B /L "$(ENV_NAME)"
else
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

# Install Python dependencies with uv
install-deps:
	@echo "Installing dependencies with uv..."
	$(CONDA_CMD) run -n $(ENV_NAME) uv pip install -r requirements/requirements.txt
	$(CONDA_CMD) run -n $(ENV_NAME) uv pip install -e .

# Initialize project (recreate environment and install pre-commit hooks)
init: env-recreate install-deps
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


# Run the Streamlit app
run-app:
	@echo "Starting Streamlit app..."
	$(CONDA_CMD) run -n $(ENV_NAME) streamlit run src/mushkil_viz/streamlit/app.py --server.port=8501
