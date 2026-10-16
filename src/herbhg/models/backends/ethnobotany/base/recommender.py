import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from rating import Rating
from util.io import FileIO
from util.config import OptionConf
from util.log import Log
from os.path import abspath
from time import strftime,localtime,time
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import random
import os
from pathlib import Path


def _reproduction_output_dir():
    """Return the configured result directory, independent of caller cwd."""
    default = Path(__file__).resolve().parents[1] / 'results'
    return os.environ.get(
        'ASTRA_OUTPUT_DIR',
        os.environ.get('CDRUG_OUTPUT_DIR', str(default)),
    )
class Recommender(object):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        self.config = conf
        self.data = None
        self.isSaveModel = False
        self.ranking = None
        self.isLoadModel = False
        self.output = None
        self.isOutput = True
        self.data = Rating(self.config, trainingSet, testSet)
        #print(self.data.herb)
        #print(self.data.disease)
        #print(len(self.data.disease))
        self.foldInfo = fold
        self.evalSettings = OptionConf(self.config['evaluation.setup'])
        self.measure = []
        self.recOutput = []
        self.num_herbs, self.num_diseases, self.train_size = self.data.trainingSize()
        self.num_compounds,self.num_proteins=self.data.cpSize()

    def initializing_log(self):
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.log = Log(self.modelName, self.modelName + self.foldInfo + ' ' + currentTime)
        #save configuration
        self.log.add('### model configuration ###')
        for k in self.config.config:
            self.log.add(k+'='+self.config[k])

    def readConfiguration(self):
        self.modelName = self.config['model.name']
        self.output = OptionConf(self.config['output.setup'])
        self.isOutput = self.output.isMainOn()

    def printAlgorConfig(self):
        "show model's configuration"
        print('Model:',self.config['model.name'])
        print('Ratings dataset:',abspath(self.config['datapath']))
        if OptionConf(self.config['evaluation.setup']).contains('-testSet'):
            print('Test set:', abspath(OptionConf(self.config['evaluation.setup'])['-testSet']))
        #print dataset statistics
        print('Training set size: (herb count: %d, disease count %d, record count: %d)' %(self.data.trainingSize()))
        print('Test set size: (herb count: %d, disease count %d, record count: %d)' %(self.data.testSize()))
        print('='*80)
        #print specific parameters if applicable
        if self.config.contains(self.config['model.name']):
            parStr = ''
            args = OptionConf(self.config[self.config['model.name']])
            for key in args.keys():
                parStr+=key[1:]+':'+args[key]+'  '
            print('Specific parameters:',parStr)
            print('=' * 80)

    def initModel(self):
        pass

    def trainModel(self):
        'build the model (for model-based Models )'
        pass

    def trainModel_tf(self):
        'training model on tensorflow'
        pass

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    #for rating prediction
    def predictForRating(self, u, i):
        pass

    #for disease prediction
    def predictForRanking(self,u):
        pass

    def checkRatingBoundary(self,prediction):
        pass
    def evalRatings(self):
        pass

    def softmax(self,x):
    #    orig_shape=x.shape
        if len(x.shape)>1:
            tmp=np.max(x,axis=1)
            x-=tmp.reshape((x.shape[0],1))
            x=np.exp(x)
            tmp=np.sum(x,axis=1)
            x/=tmp.reshape((x.shape[0],1))
        else:
            tmp=np.max(x)
            x-=tmp
            x=np.exp(x)
            tmp=np.sum(x)
            x/=tmp
        return x

    def evalRanking(self):
        print('recommender evalRanking-------------------------------------------------------')

        candidates = self.predictForRanking()
        print(candidates)
        #print(candidates.shape)
        print(type(candidates))
        #candidates=(candidates-np.min(candidates))/(np.max(candidates)-np.min(candidates))
        candidates=1/(1+np.exp(-candidates))
        print(candidates)
        # test_premat=[]
        # test_realmat=[]
        herb_mat = []
        lable_mat = []
        for i, herb in enumerate(self.data.testSet_h):
            diseaselist=self.data.testSet_h[herb].keys()
            for disease in diseaselist:
                herb_mat.append(candidates[self.data.herb[herb]][self.data.disease[disease]])
                lable_mat.append(self.data.testSet_h[herb][disease])


        #选取和测试样本相同数量的负样本
        #negativelist=pd.read_csv('./dataset/negativehdlist.txt',sep='\t',header=None)
        #randomlist=random.sample(range(0,len(negativelist)),len(self.data.testData))
        #negativel=negativelist.iloc[randomlist]
        #for line in negativel.values:
        #    # print(line[0])
        #    # print(line[1])
        #    herb_mat.append(candidates[self.data.herb[str(line[0])]][self.data.disease[str(line[1])]])
        #    lable_mat.append(0)
        #for i, user in enumerate(self.data.testSet_h):
        #    itemlist=self.data.testSet_h[user].keys()
        #    for item in itemlist:
        #        herb_mat.append(candidates[self.data.herb[user]][self.data.disease[item]])
        #        lable_mat.append(1)
        #        #lable_mat.append(self.data.testSet_u[user][item])
        #    # test_premat.append(np.array(user_mat))
        #    # test_realmat.append(np.array(lable_mat))
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        data0={'lable':lable_mat,'predict':herb_mat}
        dataframe0=pd.DataFrame(data0)
        cv_dir = os.path.join(_reproduction_output_dir(), 'cv')
        os.makedirs(cv_dir, exist_ok=True)
        prediction_path = os.path.join(cv_dir, currentTime + self.foldInfo + '.txt')
        dataframe0.to_csv(prediction_path, columns=['lable','predict'],index=False,header=None)

        # ===== 计算评估指标 =====
        print("\n计算评估指标...")

        # 转换为numpy数组
        y_true = np.array(lable_mat)
        y_score = np.array(herb_mat)

        # 1. AUC (Area Under ROC Curve)
        try:
            auc_score = roc_auc_score(y_true, y_score)
            print(f'AUC: {auc_score:.4f}')
        except Exception as e:
            print(f'AUC计算失败: {e}')
            auc_score = 0.0

        # 2. AUPR (Area Under Precision-Recall Curve)
        try:
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score)
            aupr_score = auc(recall_vals, precision_vals)
            print(f'AUPR: {aupr_score:.4f}')
        except Exception as e:
            print(f'AUPR计算失败: {e}')
            aupr_score = 0.0

        # 3. 使用最优阈值计算准确率、精确率、召回率、F1
        # 找到最优阈值（使F1最大）
        thresholds = np.linspace(0, 1, 100)
        best_f1 = 0
        best_threshold = 0.5
        best_metrics = {}

        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            try:
                f1 = f1_score(y_true, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_metrics = {
                        'accuracy': accuracy_score(y_true, y_pred),
                        'precision': precision_score(y_true, y_pred, zero_division=0),
                        'recall': recall_score(y_true, y_pred, zero_division=0),
                        'f1': f1
                    }
            except:
                continue

        print(f'\n最优阈值: {best_threshold:.4f}')
        print(f'准确率 (Accuracy): {best_metrics.get("accuracy", 0.0):.4f}')
        print(f'精确率 (Precision): {best_metrics.get("precision", 0.0):.4f}')
        print(f'召回率 (Recall): {best_metrics.get("recall", 0.0):.4f}')
        print(f'F1分数 (F1-Score): {best_metrics.get("f1", 0.0):.4f}')

        # 保存详细指标
        metrics_data = {
            'Metric': ['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Best_Threshold'],
            'Value': [
                auc_score,
                aupr_score,
                best_metrics.get('accuracy', 0.0),
                best_metrics.get('precision', 0.0),
                best_metrics.get('recall', 0.0),
                best_metrics.get('f1', 0.0),
                best_threshold
            ]
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_path = os.path.join(cv_dir, currentTime + self.foldInfo + '_metrics.txt')
        metrics_df.to_csv(metrics_path, index=False)
        print(f'\n指标已保存到: {metrics_path}')

        # 返回指标用于交叉验证汇总
        self.measure = [
            f'AUC:{auc_score:.6f}',
            f'AUPR:{aupr_score:.6f}',
            f'Accuracy:{best_metrics.get("accuracy", 0.0):.6f}',
            f'Precision:{best_metrics.get("precision", 0.0):.6f}',
            f'Recall:{best_metrics.get("recall", 0.0):.6f}',
            f'F1-Score:{best_metrics.get("f1", 0.0):.6f}'
        ]



    def execute(self):
        self.readConfiguration()
        self.initializing_log()
        if self.foldInfo == '[1]':
            self.printAlgorConfig()
        #load model from disk or build model
        if self.isLoadModel:
            print('Loading model %s...' %self.foldInfo)
            self.loadModel()
        else:
            print('Initializing model %s...' %self.foldInfo)
            self.initModel()
            print('Building Model %s...' %self.foldInfo)
            try:
                if self.evalSettings.contains('-tf'):
                    import tensorflow
                    self.trainModel_tf()
                else:
                    self.trainModel()
            except ImportError:
                self.trainModel()
        print('Predicting %s...' %self.foldInfo)
        self.evalRanking()
        if self.isSaveModel:
            print('Saving model %s...' %self.foldInfo)
            self.saveModel()
        return self.measure
