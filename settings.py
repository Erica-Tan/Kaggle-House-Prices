import os

# raw data
DATA_DIR = "data"
# processed data
PROCESSED_DIR = "processed"
# models or logs
OUTPUT_DIR = "output"
# graphs
FIGURE_DIR = "figures"

CV_FOLDS = 3
SEED = 42

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)