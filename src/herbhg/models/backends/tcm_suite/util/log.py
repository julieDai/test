import logging
from pathlib import Path

from .paths import resolve_path


class Log(object):
    def __init__(self,module,filename,log_dir=None):
        self.logger = logging.getLogger(module)
        self.logger.setLevel(level=logging.INFO)
        directory = resolve_path(log_dir or './log/')
        directory.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(directory / (filename+'.log'))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def add(self,text):
        self.logger.info(text)
