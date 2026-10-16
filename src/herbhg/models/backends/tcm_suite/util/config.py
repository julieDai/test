from pathlib import Path
import shlex

from .paths import resolve_path


class ModelConf(object):
    def __init__(self,fileName):
        self.config = {}
        self.path = resolve_path(fileName, Path.cwd())
        self.config_dir = self.path.parent
        self.readConfiguration(self.path)

    def __getitem__(self, item):
        if not self.contains(item):
            raise KeyError('parameter '+item+' is invalid')
        return self.config[item]

    def contains(self,key):
        return key in self.config

    def set(self, key, value):
        """Set or override a configuration value."""

        self.config[key] = str(value)

    def resolve_path(self, value):
        """Resolve *value* relative to the loaded configuration file."""

        return resolve_path(value, self.config_dir)

    def readConfiguration(self,file):
        if not Path(file).is_file():
            raise FileNotFoundError('config file is not found: %s' % file)
        with Path(file).open(encoding='utf-8') as f:
            for ind,line in enumerate(f):
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    try:
                        key,value=stripped.split('=', 1)
                        self.config[key.strip()]=value.strip()
                    except ValueError:
                        raise ValueError(
                            'config file is not in the correct format: %s, line %d'
                            % (file, ind + 1)
                        )

class OptionConf(object):
    def __init__(self,content):
        self.line = shlex.split(content.strip())
        self.options = {}
        self.mainOption = False
        if not self.line:
            return
        if self.line[0] == 'on':
            self.mainOption = True
        elif self.line[0] == 'off':
            self.mainOption = False
        for i,item in enumerate(self.line):
            if (item.startswith('-') or item.startswith('--')) and  not item[1:].isdigit():
                ind = i+1
                for j,sub in enumerate(self.line[ind:]):
                    if (sub.startswith('-') or sub.startswith('--')) and  not sub[1:].isdigit():
                        ind = j
                        break
                    if j == len(self.line[ind:])-1:
                        ind=j+1
                        break
                try:
                    self.options[item] = ' '.join(self.line[i+1:i+1+ind])
                except IndexError:
                    self.options[item] = 1

    def __getitem__(self, item):
        if not self.contains(item):
            raise KeyError('parameter '+item+' is invalid')
        return self.options[item]

    def keys(self):
        return self.options.keys()

    def isMainOn(self):
        return self.mainOption

    def contains(self,key):
        return key in self.options
