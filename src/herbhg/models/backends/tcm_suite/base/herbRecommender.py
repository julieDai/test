from base.diseaseRecommender import diseaseRecommender
from util.paths import resolve_config_path
from random import shuffle,randint,choice
import os
import pandas as pd
import numpy as np
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
        if USE_NEW_EMBEDDINGS:
            compound_embedding_path = resolve_config_path(
                self.config,
                'compound.embedding',
                './dataset/compound_embedding_kpgt.txt',
                ('ASTRA_COMPOUND_EMBEDDINGS', 'CDRUG_COMPOUND_EMBEDDINGS'),
            )
            protein_embedding_path = resolve_config_path(
                self.config,
                'protein.embedding',
                './dataset/protein_embedding_esm2.txt',
                ('ASTRA_PROTEIN_EMBEDDINGS', 'CDRUG_PROTEIN_EMBEDDINGS'),
            )
            a = np.loadtxt(compound_embedding_path)
            b = np.loadtxt(protein_embedding_path)
            # 归一化到与原始嵌入相同的数值范围，避免梯度爆炸
            a = (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-8)
            b = (b - b.mean(axis=0)) / (b.std(axis=0) + 1e-8)
        else:
            compound_embedding_path = resolve_config_path(
                self.config, 'compound.embedding.legacy', './dataset/compound_embedding-64.txt'
            )
            protein_embedding_path = resolve_config_path(
                self.config, 'protein.embedding.legacy', './dataset/protein_low.txt'
            )
            a = np.loadtxt(compound_embedding_path)
            b = np.loadtxt(protein_embedding_path)
        # Retain the resolved paths and post-transform NumPy arrays so the
        # strict benchmark runner can prove that the legacy TensorFlow model
        # consumed the same canonical-order feature files it audited.
        self.loaded_feature_paths = {
            'compound': str(compound_embedding_path),
            'protein': str(protein_embedding_path),
        }
        self.loaded_feature_masks = {}
        self.loaded_feature_matrices = {
            'compound': np.asarray(a).copy(),
            'protein': np.asarray(b).copy(),
        }
        self.protein_embeddings=tf.Variable(tf.convert_to_tensor(b, dtype=tf.float32),name='V')
        self.compound_embeddings = tf.Variable(tf.convert_to_tensor(a, dtype=tf.float32), name='U')

        # 加载草药和疾病的文本嵌入（新增）
        print("加载文本嵌入...")
        try:
            herb_text_path = resolve_config_path(
                self.config,
                'herb.text.embedding',
                '../artifacts/tcm_suite/herb_semantic_embeddings_64d.npy',
                ('ASTRA_HERB_TEXT_EMBEDDINGS', 'CDRUG_HERB_TEXT_EMBEDDINGS'),
            )
            disease_text_path = resolve_config_path(
                self.config,
                'disease.text.embedding',
                '../artifacts/tcm_suite/disease_semantic_embeddings_64d.npy',
                ('ASTRA_DISEASE_TEXT_EMBEDDINGS', 'CDRUG_DISEASE_TEXT_EMBEDDINGS'),
            )
            print(f"草药文本嵌入文件: {herb_text_path}")
            print(f"疾病文本嵌入文件: {disease_text_path}")
            herb_text_emb = np.load(herb_text_path)
            disease_text_emb = np.load(disease_text_path)
            self.loaded_feature_paths.update({
                'herb': str(herb_text_path),
                'disease': str(disease_text_path),
            })
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

            self.loaded_feature_matrices.update({
                'herb': np.asarray(herb_text_emb).copy(),
                'disease': np.asarray(disease_text_emb).copy(),
            })

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
