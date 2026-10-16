import os.path
from os import makedirs,remove
from pathlib import Path
from re import compile,findall,split
from .config import OptionConf
import random
class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def writeFile(dir,file,content,op = 'w'):
        output_dir = Path(dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / file).open(op, encoding='utf-8') as f:
            f.writelines(content)

    @staticmethod
    def deleteFile(filePath):
        if os.path.exists(filePath):
            remove(filePath)

    @staticmethod
    def loadDataSet(conf, file, bTest=False):
        trainingData = []
        testData = []
        ratingConfig = OptionConf(conf['ratings.setup'])
        if not bTest:
            print('loading training data...')
        else:
            print('loading test data...')
        with open(file) as f:
            ratings = f.readlines()
        # ignore the headline
        if ratingConfig.contains('-header'):
            ratings = ratings[1:]
        # order of the columns
        order = ratingConfig['-columns'].strip().split()
        delim = ' |,|\t'
        if ratingConfig.contains('-delim'):
            delim=ratingConfig['-delim']
        for lineNo, line in enumerate(ratings):
            hda = split(delim,line.strip())
            if not bTest and len(order) < 2:
                print('The rating file is not in a correct format. Error: Line num %d' % lineNo)
                exit(-1)
            try:
                # 调试信息
                if lineNo < 3:  # 只打印前3行
                    print(f"Line {lineNo}: hda={hda}, len(hda)={len(hda)}, order={order}, len(order)={len(order)}")

                herbId = hda[int(order[0])]
                diseaseId = hda[int(order[1])]
                if len(order)<3:
                    rating = 1 #default value
                else:
                    rating  = hda[int(order[2])]
            except ValueError:
                print('Error! Have you added the option -header to the rating.setup?')
                exit(-1)
            except IndexError as e:
                print(f'IndexError at line {lineNo}: hda={hda}, order={order}')
                print(f'len(hda)={len(hda)}, len(order)={len(order)}')
                raise
            if bTest:
                testData.append([herbId, diseaseId, float(rating)])
            else:
                trainingData.append([herbId, diseaseId, float(rating)])

        # 加载负样本
        if conf.contains('negative.file'):
            negative_file = Path(conf['negative.file']).expanduser().resolve()
        elif conf.contains('data.dir'):
            negative_file = Path(conf['data.dir']).expanduser().resolve() / 'negativehdlist.txt'
        else:
            negative_file = Path(file).expanduser().resolve().parent / 'negativehdlist.txt'
        if negative_file.exists():
            print(f'Loading negative samples from {negative_file}...')
            with negative_file.open(encoding='utf-8') as f:
                ratings = f.readlines()
            # ignore the headline
            if ratingConfig.contains('-header'):
                ratings = ratings[1:]
            # 使用所有负样本（不随机采样）
            for lineNo, line in enumerate(ratings):
                hda = split(delim,line.strip())
                if not bTest and len(order) < 2:
                    print('The rating file is not in a correct format. Error: Line num %d' % lineNo)
                    exit(-1)
                try:
                    herbId = hda[int(order[0])]
                    diseaseId = hda[int(order[1])]
                    if len(order)<3:
                        rating = 0 #default value for negative
                    else:
                        rating  = hda[int(order[2])]
                except ValueError:
                    print('Error! Have you added the option -header to the rating.setup?')
                    exit(-1)
                except IndexError as e:
                    print(f'IndexError at line {lineNo}: hda={hda}, order={order}')
                    continue
                if bTest:
                    testData.append([herbId, diseaseId, float(rating)])
                else:
                    trainingData.append([herbId, diseaseId, float(rating)])
            print(f'Loaded {len(ratings)} negative samples')
        else:
            print(f'Warning: negative sample file not found: {negative_file}')

        if bTest:
            return testData
        else:
            return trainingData

    @staticmethod
    def loadHerbList(filepath):
        herbList = []
        with open(filepath) as f:
            for line in f:
                herbList.append(line.strip().split()[0])
        return herbList
