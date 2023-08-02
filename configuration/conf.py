# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 13:25:38 2022

@author: yshapira
"""

import os

TIME_COEFF = 1
COST_COEFF = 5
BASE_PATH = "E:/yshapira/opl/project_28_01_22"
CONF_PATH = 'e:\yshapira\py\conf.py'

# cplex files paths
ARC_BASED_MODEL_FILE = os.path.join(BASE_PATH, "model_arc_based.mod")
PATH_BASED_MODEL_FILE = os.path.join(BASE_PATH, "model_path_based.mod")
PATH_BASED_MODEL_BINARY_FILE = os.path.join(BASE_PATH, "model_path_based_binary.mod")


# multiprocessing conf
MP_MAP_PATH = os.path.join(BASE_PATH, 'mp_map.pkl')
MP_SCRIPT_PATH = 'e:\yshapira\py\multiprocessing_generate_paths.py'
# THE BELOW LINE NEED TO BE THE LAST LINE!!!
MP_NUM_WORKERS = 8