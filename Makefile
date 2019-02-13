.PHONY: clean requirements create_environment test_environment delete_environment show-help help-prefix

PROJECT_NAME = bus_number
PYTHON_INTERPRETER = python3
VIRTUALENV = conda

## Install or update Python Dependencies
requirements: test_environment environment.lock

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

environment.lock: environment.yml
ifeq (conda, $(VIRTUALENV))
	$(CONDA_EXE) env update -n $(PROJECT_NAME) -f $<
	$(CONDA_EXE) env export -n $(PROJECT_NAME) -f $@
else
	$(error Unsupported Environment `$(VIRTUALENV)`. Use conda)
endif

## Set up python interpreter environment
create_environment:
ifeq (conda,$(VIRTUALENV))
	@echo ">>> Detected conda, creating conda environment."
ifneq ("X$(wildcard ./environment.lock)","X")
	$(CONDA_EXE) env create --name $(PROJECT_NAME) -f environment.lock
else
	@echo ">>> Creating lockfile from $(CONDA_EXE) environment specification."
	$(CONDA_EXE) env create --name $(PROJECT_NAME) -f environment.yml
	$(CONDA_EXE) env export --name $(PROJECT_NAME) -f environment.lock
endif
	@echo ">>> New conda env created. Activate with: 'conda activate $(PROJECT_NAME)'"
else
	@pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

delete_environment:
ifeq (conda,$(VIRTUALENV))
	@echo "Deleting conda environment."
	$(CONDA_EXE) env remove -n $(PROJECT_NAME)
endif


## Test python environment is set-up correctly
test_environment:
ifeq (conda,$(VIRTUALENV))
ifneq (${CONDA_DEFAULT_ENV}, $(PROJECT_NAME))
	$(error Must activate `$(PROJECT_NAME)` environment before proceeding)
endif
endif
	$(PYTHON_INTERPRETER) test_environment.py


.DEFAULT_GOAL := show-help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>

HELP_VARS := PROJECT_NAME

print-%  : ; @echo $* = $($*)

help-prefix:
	@echo "To get started:"
	@echo "  >>> $$(tput bold)make create_environment$$(tput sgr0)"
	@echo "  >>> $$(tput bold)conda activate $(PROJECT_NAME)$$(tput sgr0)"
	@echo
	@echo "$$(tput bold)Project Variables:$$(tput sgr0)"

show-help: help-prefix $(addprefix print-, $(HELP_VARS))
	@echo
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
