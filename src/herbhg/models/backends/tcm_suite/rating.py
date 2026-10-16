import numpy as np
from util.config import OptionConf
from util.paths import first_environment_value, resolve_config_path
import random
from collections import defaultdict
import pandas as pd
import os
class Rating(object):
    'data access control'
    def __init__(self,config,trainingSet, testSet):
        self.config = config
        self.evalSettings = OptionConf(self.config['evaluation.setup'])
        self.herb = {} #map herb names to id
        self.disease = {} #map disease names to id
        self.compound={}
        self.protein={}
        self.id2herb = {}
        self.id2disease = {}
        self.id2compound={}
        self.id2protein={}
        self.herbMeans = {} #mean values of herbs's ratings
        self.diseaseMeans = {} #mean values of diseases's ratings
        self.globalMean = 0
        self.trainSet_h = defaultdict(dict)
        self.trainSet_d = defaultdict(dict)
        self.testSet_h = defaultdict(dict) #test set in the form of [herb][disease]=rating
        self.testSet_d = defaultdict(dict) #test set in the form of [disease][herb]=rating
        self.rScale = [] #rating scale
        hc_path = resolve_config_path(
            config, 'hc.path', './dataset/H_C.txt', ('ASTRA_HC_FILE', 'CDRUG_HC_FILE')
        )
        cp_path = resolve_config_path(
            config, 'cp.path', './dataset/C_P.txt', ('ASTRA_CP_FILE', 'CDRUG_CP_FILE')
        )
        pd_path = resolve_config_path(
            config, 'pd.path', './dataset/P_D.txt', ('ASTRA_PD_FILE', 'CDRUG_PD_FILE')
        )
        if first_environment_value(('ASTRA_NODE_HC_FILE', 'CDRUG_NODE_HC_FILE')):
            hc_node_path = resolve_config_path(
                config,
                'node.hc.path',
                hc_path,
                ('ASTRA_NODE_HC_FILE', 'CDRUG_NODE_HC_FILE'),
            )
        elif first_environment_value(('ASTRA_HC_FILE', 'CDRUG_HC_FILE')):
            # Preserve the legacy fallback: a structural relation override is
            # also the node-universe source unless a dedicated node file is set.
            hc_node_path = hc_path
        else:
            hc_node_path = resolve_config_path(config, 'node.hc.path', hc_path)
        if first_environment_value(('ASTRA_NODE_PD_FILE', 'CDRUG_NODE_PD_FILE')):
            pd_node_path = resolve_config_path(
                config,
                'node.pd.path',
                pd_path,
                ('ASTRA_NODE_PD_FILE', 'CDRUG_NODE_PD_FILE'),
            )
        elif first_environment_value(('ASTRA_PD_FILE', 'CDRUG_PD_FILE')):
            pd_node_path = pd_path
        else:
            pd_node_path = resolve_config_path(config, 'node.pd.path', pd_path)
        self.relation_paths = {
            'H_C': hc_path,
            'C_P': cp_path,
            'P_D': pd_path,
            'H_C_nodes': hc_node_path,
            'P_D_nodes': pd_node_path,
        }
        for relation, path in self.relation_paths.items():
            if not path.is_file():
                raise FileNotFoundError('%s relation file is not found: %s' % (relation, path))

        HC=pd.read_csv(hc_path,sep='\t',header=None,dtype={0:str,1:str})
        CP=pd.read_csv(cp_path,sep='\t',header=None,dtype={0:str,1:str})
        PD=pd.read_csv(pd_path,sep='\t',header=None,dtype={0:str,1:str})
        HC_nodes = pd.read_csv(hc_node_path,sep='\t',header=None,dtype={0:str,1:str})
        PD_nodes = pd.read_csv(pd_node_path,sep='\t',header=None,dtype={0:str,1:str})
        hlist=HC_nodes[0].unique()
        clist=HC_nodes[1].unique()
        plist=CP[1].unique()
        dlist=PD_nodes[1].unique()
        self.Hlist=hlist
        self.Dlist=dlist
        self.Clist=clist
        self.Plist=plist
        self.trainingData = trainingSet[:]
        self.testData = testSet[:]
        # HC=HC.values.tolist()
        self.hcassociation = HC.values.tolist()
        self.cpassociation = CP.values.tolist()
        self.pdassociation = PD.values.tolist()
        #print(HC)
        print('trainingdata type:',type(self.trainingData))
        #print(testSet)
        self.__generateSet()
        self.__computediseaseMean()
        self.__computeherbMean()
        self.__globalAverage()
        if self.evalSettings.contains('-cold'):
            #evaluation on cold-start herbs
            self.__cold_start_test()


    def __generateSet(self):
        scale = set()
        #if validation is conducted, we sample the training data at a given probability to form the validation set,
        #and then replacing the test data with the validation data to tune parameters.
        if self.evalSettings.contains('-val'):
            random.shuffle(self.trainingData)
            separation = int(self.elemCount()*float(self.evalSettings['-val']))
            self.testData = self.trainingData[:separation]
            self.trainingData = self.trainingData[separation:]

        for i,herb in enumerate(self.Hlist):
            herbName=str(herb)
            if herbName not in self.herb:
                self.herb[herbName] = len(self.herb)
                self.id2herb[self.herb[herbName]] = herbName
        # order the disease
        for i,disease in enumerate(self.Dlist):
            diseaseName=str(disease)
            if diseaseName not in self.disease:
                self.disease[diseaseName] = len(self.disease)
                self.id2disease[self.disease[diseaseName]] = diseaseName
        # order the compound
        for i, compound in enumerate(self.Clist):
            compoundName = str(compound)
            if compoundName not in self.compound:
                self.compound[compoundName] = len(self.compound)
                self.id2compound[self.compound[compoundName]] = compoundName
        # order the protein
        for i, protein in enumerate(self.Plist):
            proteinName = str(protein)
            if proteinName not in self.protein:
                self.protein[proteinName] = len(self.protein)
                self.id2protein[self.protein[proteinName]] = proteinName

        for i,entry in enumerate(self.trainingData):
            herbName,diseaseName,rating = entry
            #print('type of herbname',type(herbName))
            # makes the rating within the range [0, 1].
            #rating = normalize(float(rating), self.rScale[-1], self.rScale[0])
            #self.trainingData[i][2] = rating
            # order the herb
                # herbList.append
            self.trainSet_h[herbName][diseaseName] = rating
            self.trainSet_d[diseaseName][herbName] = rating
            scale.add(float(rating))
        self.rScale = list(scale)
        self.rScale.sort()
        for entry in self.testData:
            if self.evalSettings.contains('-predict'):
                self.testSet_h[entry]={}
            else:
                herbName, diseaseName, rating = entry
                self.testSet_h[herbName][diseaseName] = rating
                self.testSet_d[diseaseName][herbName] = rating


    def __globalAverage(self):
        total = sum(self.herbMeans.values())
        if total==0:
            self.globalMean = 0
        else:
            self.globalMean = total/len(self.herbMeans)

    def __computeherbMean(self):
        for u in self.herb:
            self.herbMeans[u] = sum(self.trainSet_h[u].values())/(len(self.trainSet_h[u])+1)

    def __computediseaseMean(self):
        for c in self.disease:
            self.diseaseMeans[c] = sum(self.trainSet_d[c].values())/(len(self.trainSet_d[c])+1)

    def getherbId(self,u):
        if u in self.herb:
            return self.herb[u]

    def getdiseaseId(self,i):
        if i in self.disease:
            return self.disease[i]

    def trainingSize(self):
        return (len(self.herb),len(self.disease),len(self.trainingData))

    def cpSize(self):
        return (len(self.compound),len(self.protein))

    def testSize(self):
        return (len(self.testSet_h),len(self.testSet_d),len(self.testData))

    def contains(self,u,i):
        'whether herb u rated disease i'
        if u in self.herb and i in self.trainSet_u[u]:
            return True
        else:
            return False

    def containsherb(self,u):
        'whether herb is in training set'
        if u in self.herb:
            return True
        else:
            return False

    def containsdisease(self,i):
        'whether disease is in training set'
        if i in self.disease:
            return True
        else:
            return False

    def herbRated(self,u):
        return list(self.trainSet_u[u].keys()),list(self.trainSet_u[u].values())

    def diseaseRated(self,i):
        return list(self.trainSet_i[i].keys()),list(self.trainSet_i[i].values())

    def row(self,u):
        k,v = self.herbRated(u)
        vec = np.zeros(len(self.disease))
        #print vec
        for pair in zip(k,v):
            iid = self.disease[pair[0]]
            vec[iid]=pair[1]
        return vec

    def col(self,i):
        k,v = self.diseaseRated(i)
        vec = np.zeros(len(self.herb))
        #print vec
        for pair in zip(k,v):
            uid = self.herb[pair[0]]
            vec[uid]=pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.herb),len(self.disease)))
        for u in self.herb:
            k, v = self.herbRated(u)
            vec = np.zeros(len(self.disease))
            # print vec
            for pair in zip(k, v):
                iid = self.disease[pair[0]]
                vec[iid] = pair[1]
            m[self.herb[u]]=vec
        return m
    # def row(self,u):
    #     return self.trainingMatrix.row(self.getherbId(u))
    #
    # def col(self,c):
    #     return self.trainingMatrix.col(self.getdiseaseId(c))

    def sRow(self,u):
        return self.trainSet_u[u]

    def sCol(self,c):
        return self.trainSet_i[c]

    def rating(self,u,c):
        if self.contains(u,c):
            return self.trainSet_u[u][c]
        return -1

    def ratingScale(self):
        return (self.rScale[0],self.rScale[1])

    def elemCount(self):
        return len(self.trainingData)
