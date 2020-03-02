
.PHONY: test ex.%

help :
	# TARGETS (e.g. trigger with 'make <target>'
	#
	#	test             Run all python tests
	#                        Tests are found under test/
	#
	#	ex.<exp>	 Run the experiment <exp>.
	#			 Experiments are found under
	#                           pylib/experiments
	#
	#                        Example:
	#                           make ex.classifier1


test :
	python3 -m unittest discover -s test

ex.% :
	PYTHONPATH=./pylib python3 -m experiments $*


# Run an experiment




