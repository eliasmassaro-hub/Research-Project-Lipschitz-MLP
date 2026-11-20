#!/bin/bash
# Whole experiment: training (runs already trained are skipped), metrics, figures.
set -e
python main_lipmlp.py
python evaluate.py
python figures.py
