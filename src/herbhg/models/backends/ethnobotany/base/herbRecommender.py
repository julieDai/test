from base.diseaseRecommender import diseaseRecommender
from ethnobotany_feature_paths import (
    load_protein_missing_mask,
    resolve_feature_paths,
    standardize_and_clip_feature_matrix,
)
from random import shuffle,randint,choice
import pandas as pd
import numpy as np
from pathlib import Path
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class herbRecommender(diseaseRecommender):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        super(herbRecommender, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(herbRecommender, self).readConfiguration()
        self.batch_size = int(self.config['batch_size'])

    def printAlgorConfig(self):
        super(herbRecommender, self).printAlgorConfig()

    def initModel(self):
        super(herbRecommender, self).initModel()
        self.u_idx = tf.placeholder(tf.int32, name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, name="v_idx")
        self.r = tf.placeholder(tf.float32, name="rating")

        # 加载化学成分和蛋白质嵌入
        # USE_NEW_EMBEDDINGS=True: 使用KPGT+ESM2提取的嵌入
        # USE_NEW_EMBEDDINGS=False: 使用原始嵌入
        USE_NEW_EMBEDDINGS = True
        feature_paths = resolve_feature_paths(self.config)
        # Retained for the fair runner's fail-closed path-binding audit.
        self.loaded_feature_paths = {
            entity: str(path) for entity, path in feature_paths.items()
        }
        if USE_NEW_EMBEDDINGS:
            a = np.loadtxt(feature_paths['compound'])
            b = np.loadtxt(feature_paths['protein'])
            protein_missing_mask, protein_mask_path = load_protein_missing_mask(
                self.config, expected_rows=b.shape[0]
            )
            self.loaded_feature_masks = (
                {'protein': str(protein_mask_path)}
                if protein_mask_path is not None
                else {}
            )
            # 归一化到与原始嵌入相同的数值范围，避免梯度爆炸
            a = standardize_and_clip_feature_matrix(a)
            b = standardize_and_clip_feature_matrix(
                b, missing_mask=protein_missing_mask
            )
            # Retain a NumPy audit copy before TensorFlow variable construction.
            self.loaded_protein_embedding_matrix = np.asarray(b).copy()
            self.loaded_feature_matrices = {
                'compound': np.asarray(a).copy(),
                'protein': np.asarray(b).copy(),
            }
            print(f"化合物嵌入范围: [{a.min():.4f}, {a.max():.4f}]")
            print(f"蛋白质嵌入范围: [{b.min():.4f}, {b.max():.4f}]")
        else:
            data_dir = Path(self.config['data.dir']).expanduser().resolve()
            a = np.loadtxt(data_dir / 'compound_embedding-64.txt')
            b = np.loadtxt(data_dir / 'protein_low.txt')
            self.loaded_feature_masks = {}
            self.loaded_protein_embedding_matrix = np.asarray(b).copy()
            self.loaded_feature_matrices = {
                'compound': np.asarray(a).copy(),
                'protein': np.asarray(b).copy(),
            }
        self.protein_embeddings=tf.Variable(tf.convert_to_tensor(b, dtype=tf.float32),name='V')
        self.compound_embeddings = tf.Variable(tf.convert_to_tensor(a, dtype=tf.float32), name='U')

        # 加载草药和疾病的文本嵌入（新增）
        print("加载文本嵌入...")
        try:
            herb_text_emb = np.load(feature_paths['herb'])
            disease_text_emb = np.load(feature_paths['disease'])
            print(f"草药文本嵌入形状: {herb_text_emb.shape}")
            print(f"疾病文本嵌入形状: {disease_text_emb.shape}")

            # 数据检查：检测NaN和Inf
            if np.isnan(herb_text_emb).any() or np.isinf(herb_text_emb).any():
                print("错误：草药文本嵌入包含NaN或Inf值！")
                raise ValueError("草药文本嵌入数据异常")
            if np.isnan(disease_text_emb).any() or np.isinf(disease_text_emb).any():
                print("错误：疾病文本嵌入包含NaN或Inf值！")
                raise ValueError("疾病文本嵌入数据异常")

            # 数据统计
            print(f"草药嵌入范围: [{herb_text_emb.min():.4f}, {herb_text_emb.max():.4f}]")
            print(f"疾病嵌入范围: [{disease_text_emb.min():.4f}, {disease_text_emb.max():.4f}]")

            # 适度裁剪文本嵌入
            herb_text_emb = np.clip(herb_text_emb, -10, 10)
            disease_text_emb = np.clip(disease_text_emb, -10, 10)
            self.loaded_feature_matrices.update({
                'herb': np.asarray(herb_text_emb).copy(),
                'disease': np.asarray(disease_text_emb).copy(),
            })
            print(f"裁剪后草药嵌入范围: [{herb_text_emb.min():.4f}, {herb_text_emb.max():.4f}]")
            print(f"裁剪后疾病嵌入范围: [{disease_text_emb.min():.4f}, {disease_text_emb.max():.4f}]")

            # 将文本嵌入转换为TensorFlow变量
            self.herb_text_embeddings = tf.Variable(
                tf.convert_to_tensor(herb_text_emb, dtype=tf.float32),
                name='herb_text_emb'
            )
            self.disease_text_embeddings = tf.Variable(
                tf.convert_to_tensor(disease_text_emb, dtype=tf.float32),
                name='disease_text_emb'
            )
            self.use_text_embeddings = True
            print("文本嵌入加载成功！")
        except FileNotFoundError:
            allow_missing = os.environ.get(
                'ASTRA_ALLOW_MISSING_SEMANTIC', ''
            ).strip().lower() in {'1', 'true', 'yes', 'on'}
            if not allow_missing:
                raise
            print("警告：已显式允许缺少文本嵌入，将使用结构分支")
            self.use_text_embeddings = False

        self.batch_compound_emb = tf.nn.embedding_lookup(self.compound_embeddings, self.u_idx)
        self.batcsh_pos_protein_emb = tf.nn.embedding_lookup(self.protein_embeddings, self.v_idx)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def next_batch_pairwise(self):
        shuffle(self.data.trainingData)
        batch_id = 0
        while batch_id < self.train_size:
            if batch_id + self.batch_size <= self.train_size:
                herbs = [self.data.trainingData[idx][0] for idx in range(batch_id, self.batch_size + batch_id)]
                diseases = [self.data.trainingData[idx][1] for idx in range(batch_id, self.batch_size + batch_id)]
                rating = [self.data.trainingData[idx][2] for idx in range(batch_id, self.batch_size + batch_id)]
                batch_id += self.batch_size
            else:
                herbs = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
                diseases = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
                rating = [self.data.trainingData[idx][2] for idx in range(batch_id, self.train_size)]
                batch_id = self.train_size

            u_idx, i_idx, j_idx = [], [], []
            disease_list = list(self.data.disease.keys())
            for i, herb in enumerate(herbs):
                i_idx.append(self.data.disease[diseases[i]])
                u_idx.append(self.data.herb[herb])
                j_idx.append(rating[i])
            yield u_idx, i_idx, j_idx

    def predictForRanking(self,u):
        pass
