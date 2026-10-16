import os.path
import shlex


class ModelConf(object):
    def __init__(self,fileName):
        self.config = {}
        self.readConfiguration(fileName)

    def __getitem__(self, item):
        if not self.contains(item):
            print('parameter '+item+' is invalid!')
            exit(-1)
        return self.config[item]

    def contains(self,key):
        return key in self.config

    def readConfiguration(self,file):
        if not os.path.exists(file):
            print('config file is not found!')
            raise IOError
        with open(file) as f:
            for ind,line in enumerate(f):
                if line.strip()!='':
                    try:
                        key,value=line.strip().split('=', 1)
                        self.config[key]=value
                    except ValueError:
                        print('config file is not in the correct format! Error Line:%d'%(ind))

class OptionConf(object):
    def __init__(self,content):
        # shlex keeps quoted paths containing spaces as one value. Existing
        # configuration syntax (for example ``-columns 0 1 2``) is unchanged.
        self.line = shlex.split(content.strip())
        self.options = {}
        self.mainOption = False
        if not self.line:
            return
        if self.line[0] == 'on':
            self.mainOption = True
        elif self.line[0] == 'off':
            self.mainOption = False
        for i, item in enumerate(self.line):
            if not self._is_option(item):
                continue
            end = i + 1
            while end < len(self.line) and not self._is_option(self.line[end]):
                end += 1
            value = ' '.join(self.line[i + 1:end])
            self.options[item] = value if value else 1

    @staticmethod
    def _is_option(item):
        if not item.startswith('-'):
            return False
        try:
            float(item)
        except ValueError:
            return True
        return False

    def __getitem__(self, item):
        if not self.contains(item):
            print('parameter '+item+' is invalid!')
            exit(-1)
        return self.options[item]

    def keys(self):
        return self.options.keys()

    def isMainOn(self):
        return self.mainOption

    def contains(self,key):
        return key in self.options
