import os
import random
from pathlib import Path
from re import split

from .config import OptionConf
from .paths import first_environment_value, resolve_config_path, resolve_path


class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def writeFile(dir,file,content,op = 'w'):
        directory = Path(dir)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / file).open(op, encoding='utf-8') as f:
            f.writelines(content)

    @staticmethod
    def deleteFile(filePath):
        path = Path(filePath)
        if path.exists():
            path.unlink()

    @staticmethod
    def loadDataSet(conf, file, bTest=False):
        trainingData = []
        testData = []
        ratingConfig = OptionConf(conf['ratings.setup'])
        if not bTest:
            print('loading training data...')
        else:
            print('loading test data...')
        # Respect the file requested by the caller (for example an explicit
        # test set) while resolving relative values against the config file.
        data_file = (
            conf.resolve_path(file)
            if hasattr(conf, 'resolve_path')
            else resolve_path(file)
        )
        if not data_file.is_file():
            raise FileNotFoundError('association file is not found: %s' % data_file)
        with data_file.open(encoding='utf-8') as f:
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
                herbId = hda[int(order[0])]
                diseaseId = hda[int(order[1])]
                if len(order)<3:
                    rating = 1 #default value
                else:
                    rating  = hda[int(order[2])]
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    'invalid association row in %s at line %d' % (data_file, lineNo + 1)
                ) from exc
            if bTest:
                testData.append([herbId, diseaseId, float(rating)])
            else:
                trainingData.append([herbId, diseaseId, float(rating)])

        negative_file = resolve_config_path(
            conf,
            'negative.path',
            data_file.parent / 'negativehdlist.txt',
            ('ASTRA_NEGATIVE_FILE', 'CDRUG_NEGATIVE_FILE'),
        )
        if not negative_file.is_file():
            raise FileNotFoundError('negative association file is not found: %s' % negative_file)
        with negative_file.open(encoding='utf-8') as f:
            ratings = f.readlines()
        # ignore the headline
        if ratingConfig.contains('-header'):
            ratings = ratings[1:]
        # order of the columns
        order = ratingConfig['-columns'].strip().split()
        delim = ' |,|\t'
        if ratingConfig.contains('-delim'):
            delim=ratingConfig['-delim']
        requested_count = first_environment_value(
            ('ASTRA_NEGATIVE_COUNT', 'CDRUG_NEGATIVE_COUNT')
        )
        if requested_count is None and conf.contains('negative.count'):
            requested_count = conf['negative.count']
        if requested_count in (None, '', 'match-positive'):
            negative_count = len(testData) if bTest else len(trainingData)
        else:
            negative_count = int(requested_count)
        if negative_count < 0:
            raise ValueError('negative sample count must be non-negative')
        if negative_count > len(ratings):
            raise ValueError(
                'requested %d negatives, but %s contains only %d rows'
                % (negative_count, negative_file, len(ratings))
            )

        # A caller-provided frozen file whose size already matches the target
        # is consumed in its stored order.  Otherwise retain the legacy
        # sampling-without-replacement behaviour with an optional seed.
        frozen_negative = first_environment_value(
            ('ASTRA_NEGATIVE_FILE', 'CDRUG_NEGATIVE_FILE')
        )
        if len(ratings) == negative_count and frozen_negative:
            new_ratings = ratings
        else:
            seed = first_environment_value(('ASTRA_SEED', 'CDRUG_SEED'))
            rng = random.Random(int(seed)) if seed is not None else random
            new_ratings = rng.sample(ratings, negative_count)

        for lineNo, line in enumerate(new_ratings):
            hda = split(delim,line.strip())
            if not bTest and len(order) < 2:
                print('The rating file is not in a correct format. Error: Line num %d' % lineNo)
                exit(-1)
            try:
                herbId = hda[int(order[0])]
                diseaseId = hda[int(order[1])]
                if len(order)<3:
                    rating = 0 #default value
                else:
                    rating  = hda[int(order[2])]
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    'invalid negative row in %s at line %d' % (negative_file, lineNo + 1)
                ) from exc
            if bTest:
                testData.append([herbId, diseaseId, float(rating)])
            else:
                trainingData.append([herbId, diseaseId, float(rating)])

        if bTest:
            return testData
        else:
            return trainingData

    @staticmethod
    def loadHerbList(filepath):
        herbList = []
        with Path(filepath).open(encoding='utf-8') as f:
            for line in f:
                herbList.append(line.strip().split()[0])
        return herbList
