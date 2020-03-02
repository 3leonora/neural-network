'''
experiments - entry point for running various experiments

usage:

python -m experiments SUBEXPERIMENT <options>
'''

import sys
import importlib

def main(args):

    if len(args) is 0:
        print(
"""
Usage: XXX EXPERMINENTNAME [options ...]

where XXX is something like "PYTHONPATH=pylib python3 -m experiments" or
some helperscript like "runex".

The EXPERIMENTNAME must be a python module in pylib/experiments.

""")
        sys.exit(0)

    module = importlib.import_module(f"..{args[0]}", "experiments.x")

    module.run_experiment(args[1:])

main(sys.argv[1:])
