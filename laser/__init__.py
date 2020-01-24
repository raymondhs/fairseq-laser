import os, sys
# hack to prevent ModuleNotFoundError with multiprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from . import laser_lstm
from . import laser_task
