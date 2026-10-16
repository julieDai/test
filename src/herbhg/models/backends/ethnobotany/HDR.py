#coding:utf8
import sys
import os
import importlib
# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util.config import OptionConf
from util.dataSplit import *
from multiprocessing import Process,Manager
from util.io import FileIO
from time import strftime,localtime,time
import pandas as pd
import numpy as np
try:
    import mkl
except ImportError:
    mkl = None
class HDR(object):
    def __init__(self,config):
        self.trainingData = []  # training data
        self.testData = []  # testData
        self.measure = []
        self.config =config
        self.ratingConfig = OptionConf(config['ratings.setup'])
        if self.config.contains('evaluation.setup'):
            self.evaluation = OptionConf(config['evaluation.setup'])
            self.trainingData = FileIO.loadDataSet(config, config['datapath'])

        else:
            print('Wrong configuration of evaluation!')
            exit(-1)

        print('Reading data and preprocessing...')

    def _model_class(self):
        """Load the configured local model without using exec/eval."""
        model_name = self.config['model.name']
        if not model_name.isidentifier():
            raise ValueError('Invalid model name: %s' % model_name)
        module = importlib.import_module(model_name)
        try:
            return getattr(module, model_name)
        except AttributeError as exc:
            raise ImportError(
                'Model class %s was not found in %s.py' % (model_name, model_name)
            ) from exc

    def execute(self):
        model_class = self._model_class()
        if self.evaluation.contains('-cv'):
            k = int(self.evaluation['-cv'])
            if k < 2 or k > 10: #limit to 2-10 fold cross validation
                print("k for cross-validation should not be greater than 10 or less than 2")
                exit(-1)
            if mkl is not None:
                mkl.set_num_threads(max(1,mkl.get_max_threads()//k))
            #create the manager for communication among multiple processes
            manager = Manager()
            mDict = manager.dict()
            i = 1
            tasks = []
            for train,test in DataSplit.crossValidation(self.trainingData,k):
                fold = '['+str(i)+']'
               #create the process
                p = Process(target=run,args=(mDict,model_class(self.config,train,test,fold),i))
                tasks.append(p)
                i+=1
            #start the processes
            for p in tasks:
                p.start()
                if not self.evaluation.contains('-p'):
                    p.join()
            #wait until all processes are completed
            if self.evaluation.contains('-p'):
                for p in tasks:
                    p.join()
            #compute the average error of k-fold cross validation
            self.measure = [dict(mDict)[i] for i in range(1,k+1)]

            # Aggregate fold-level metrics as mean and standard deviation.
            print("\n" + "="*80)
            print("交叉验证结果汇总")
            print("="*80)

            # 提取所有指标
            metric_names = []
            metric_values = {i: [] for i in range(1, k+1)}

            for fold_idx in range(k):
                for metric_str in self.measure[fold_idx]:
                    metric_name, metric_value = metric_str.split(':')
                    if fold_idx == 0:
                        metric_names.append(metric_name)
                    metric_values[fold_idx+1].append(float(metric_value))

            # 计算统计信息
            results_summary = []
            results_detailed = []

            print(f"\n各Fold详细结果:")
            print("-"*80)
            for metric_idx, metric_name in enumerate(metric_names):
                values = [metric_values[fold_idx+1][metric_idx] for fold_idx in range(k)]
                mean_val = np.mean(values)
                std_val = np.std(values)

                # 打印每个fold的结果
                fold_results = " | ".join([f"Fold{i+1}: {values[i]:.4f}" for i in range(k)])
                print(f"{metric_name:12s}: {fold_results}")

                # 保存汇总结果（保留 4 位小数）
                results_summary.append(f'{metric_name}:{mean_val:.4f}±{std_val:.4f}\n')

                # 保存详细结果（包含每个fold）
                results_detailed.append({
                    'Metric': metric_name,
                    'Mean': mean_val,
                    'Std': std_val,
                    **{f'Fold{i+1}': values[i] for i in range(k)}
                })

            print("\n" + "-"*80)
            print(f"平均结果 (Mean ± Std):")
            print("-"*80)
            for metric_idx, metric_name in enumerate(metric_names):
                values = [metric_values[fold_idx+1][metric_idx] for fold_idx in range(k)]
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"{metric_name:12s}: {mean_val:.4f} ± {std_val:.4f}")

            #output result
            currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
            outDir = OptionConf(self.config['output.setup'])['-dir']
            os.makedirs(outDir, exist_ok=True)
            fileName = self.config['model.name'] +'@'+currentTime+'-'+str(k)+'-fold-cv' + '.txt'

            # 保存汇总结果（简洁版）
            FileIO.writeFile(outDir,fileName,results_summary)

            # 保存详细结果（包含每个fold）
            detailed_df = pd.DataFrame(results_detailed)
            detailed_file = self.config['model.name'] +'@'+currentTime+'-'+str(k)+'-fold-cv-detailed.csv'
            detailed_df.to_csv(os.path.join(outDir, detailed_file), index=False)

            # Save the compact metrics table used by downstream reporting.
            user_columns = ['Recall', 'Accuracy', 'F1-Score', 'AUC', 'AUPR']

            # Keep a stable column order for machine-readable results.
            user_df_data = []

            # 添加每个fold的结果
            for fold_idx in range(k):
                row = {}
                for user_col in user_columns:
                    # 找到对应指标的值
                    metric_idx = metric_names.index(user_col)
                    row[user_col] = metric_values[fold_idx+1][metric_idx]
                user_df_data.append(row)

            # 计算平均值±标准差
            mean_std_row = {}
            for user_col in user_columns:
                metric_idx = metric_names.index(user_col)
                values = [metric_values[fold_idx+1][metric_idx] for fold_idx in range(k)]
                mean_val = np.mean(values)
                std_val = np.std(values)
                mean_std_row[user_col] = f'{mean_val:.4f}±{std_val:.4f}'
            user_df_data.append(mean_std_row)

            # 创建DataFrame并保存
            user_df = pd.DataFrame(user_df_data)
            user_file = self.config['model.name'] +'@'+currentTime+'-'+str(k)+'-fold-cv-result.csv'

            # 重命名列为中文
            column_names_cn = {
                'Recall': 'recall',
                'Accuracy': '准确率',
                'F1-Score': 'F1分数',
                'AUC': 'AUC',
                'AUPR': 'AUPR'
            }
            user_df.columns = [column_names_cn[col] for col in user_df.columns]

            # 添加标签列（Fold1-Fold5 + 平均±方差）
            labels = [f'Fold{i+1}' for i in range(k)] + ['Mean±SD']
            user_df.insert(0, ' ', labels)

            user_df.to_csv(os.path.join(outDir, user_file), index=False, encoding='utf-8-sig')

            print(f'\n结果已保存:')
            print(f'  汇总结果: {os.path.join(outDir, fileName)}')
            print(f'  详细结果: {os.path.join(outDir, detailed_file)}')
            print(f'  Metrics table: {os.path.join(outDir, user_file)}')
            print('='*80)
            print('The result of %d-fold cross validation:\n%s' %(k,''.join(results_summary)))
        else:
            model_class(self.config,self.trainingData,self.testData).execute()


def run(measure,algor,order):
    measure[order] = algor.execute()
