import logging
import os
from pathlib import Path
class Log(object):
    def __init__(self,module,filename):
        self.logger = logging.getLogger(module)
        self.logger.setLevel(level=logging.INFO)
        default = Path(__file__).resolve().parents[1] / 'log'
        log_dir = os.environ.get(
            'ASTRA_LOG_DIR',
            os.environ.get('CDRUG_LOG_DIR', str(default)),
        )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        handler = logging.FileHandler(os.path.join(log_dir, filename+'.log'))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def add(self,text):
        self.logger.info(text)
