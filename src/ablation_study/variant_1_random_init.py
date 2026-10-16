"""
变体1：随机初始化生物实体嵌入
- 使用随机正态分布初始化化学成分和蛋白质嵌入（替代KPGT和ESM2）
- 保留SciBERT文本嵌入用于草药和疾病
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from ASTRA import ASTRA
from ablation_study.paths import resolve_embedding_paths

class Variant1RandomInit(ASTRA):
    def __init__(self, conf, trainingSet=None, testSet=None, fold='[1]'):
        self._variant1_random_init = True
        # 初始化消融实验标志（使用默认值）
        self.use_mean_pooling = False
        self.use_disease_hypergraph = True
        self.use_residual = True
        self.use_last_layer_only = False
        super(Variant1RandomInit, self).__init__(conf, trainingSet, testSet, fold)

    def _load_embeddings_variant1(self):
        """使用随机正态分布初始化化学成分和蛋白质嵌入"""
        print("变体1：使用随机初始化替代KPGT和ESM2预训练嵌入")

        # 随机初始化化学成分嵌入（替代KPGT）
        compound_emb = np.random.normal(0, 0.1, (self.num_compounds, self.emb_size)).astype(np.float32)
        print(f"随机初始化化学成分嵌入: {compound_emb.shape}")

        # 随机初始化蛋白质嵌入（替代ESM2）
        protein_emb = np.random.normal(0, 0.1, (self.num_proteins, self.emb_size)).astype(np.float32)
        print(f"随机初始化蛋白质嵌入: {protein_emb.shape}")

        # 保留SciBERT文本嵌入（草药和疾病）
        try:
            embedding_paths = resolve_embedding_paths(self.config)
            herb_text_emb = np.load(embedding_paths.herb_semantic)
            disease_text_emb = np.load(embedding_paths.disease_semantic)
            print(f"加载草药文本嵌入: {herb_text_emb.shape}")
            print(f"加载疾病文本嵌入: {disease_text_emb.shape}")

            self.herb_text_embeddings = tf.Variable(herb_text_emb, trainable=False, name='herb_text_emb')
            self.disease_text_embeddings = tf.Variable(disease_text_emb, trainable=False, name='disease_text_emb')
            self.use_text_embeddings = True
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Variant 1 requires semantic embeddings: {exc.filename}"
            ) from exc

        # 创建TensorFlow变量
        self.compound_embeddings = tf.Variable(compound_emb, trainable=True, name='compound_emb')
        self.protein_embeddings = tf.Variable(protein_emb, trainable=True, name='protein_emb')

    def initModel(self):
        """完全重写initModel方法"""
        # 调用父类的readConfiguration
        self.readConfiguration()

        # 创建占位符
        self.u_idx = tf.compat.v1.placeholder(tf.int32, name="u_idx")
        self.v_idx = tf.compat.v1.placeholder(tf.int32, name="v_idx")
        self.r = tf.compat.v1.placeholder(tf.float32, name="rating")

        # 使用变体1的嵌入加载方式
        self._load_embeddings_variant1()

        # 创建会话
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)

        # 构建HGHDA模型
        self._build_hghda_model()

    def _build_hghda_model(self):
        """构建HGHDA模型（复制自HGHDA.initModel的147-544行）"""
        # Build adjacency matrix
        A = self.buildAdjacencyMatrix()
        cp = self.buildcpAdjacencyMatrix()
        pd = self.buildpdAdjacencyMatrix()
        hc = self.buildhcAdjacencyMatrix()

        # 仅使用关联的
        H_c = hc.transpose()
        D_hc_v = H_c.sum(axis=1).reshape(1, -1)
        D_hc_e = H_c.sum(axis=0).reshape(1, -1)
        temp1 = (H_c.multiply(1.0 / D_hc_e)).transpose()
        temp2 = (H_c.transpose().multiply(1.0 / D_hc_v)).transpose()
        edge = temp1
        node = temp2
        A_e = edge.tocoo()
        A_n = node.tocoo()
        edge_indices = np.mat([A_e.row, A_e.col]).transpose()
        node_indices = np.mat([A_n.row, A_n.col]).transpose()
        H_e = tf.SparseTensor(edge_indices, A_e.data.astype(np.float32), A_e.shape)
        H_n = tf.SparseTensor(node_indices, A_n.data.astype(np.float32), A_n.shape)

        P_d = pd
        D_P_v = P_d.sum(axis=1).reshape(1, -1)
        D_P_e = P_d.sum(axis=0).reshape(1, -1)
        temp1 = (P_d.multiply(1.0 / D_P_e)).transpose()
        temp2 = (P_d.transpose().multiply(1.0 / D_P_v)).transpose()
        pd_edge = temp1
        pd_node = temp2
        A_pde = pd_edge.tocoo()
        A_pdn = pd_node.tocoo()
        pdedge_indices = np.mat([A_pde.row, A_pde.col]).transpose()
        pdnode_indices = np.mat([A_pdn.row, A_pdn.col]).transpose()
        P_e = tf.SparseTensor(pdedge_indices, A_pde.data.astype(np.float32), A_pde.shape)
        P_n = tf.SparseTensor(pdnode_indices, A_pdn.data.astype(np.float32), A_pdn.shape)

        # 自适应团扩展初始化
        if self.use_clique_expansion:
            print("初始化自适应团扩展模块...")
            herb_to_compounds, disease_to_proteins = self.build_hyperedge_node_mapping()
            print(f"草药数量: {len(herb_to_compounds)}, 疾病数量: {len(disease_to_proteins)}")

            self.compound_pairwise_indices = self.build_pairwise_edge_indices(
                herb_to_compounds, self.num_compounds
            )
            self.compound_edge_to_hyperedge = self.edge_to_hyperedge.copy()

            self.protein_pairwise_indices = self.build_pairwise_edge_indices(
                disease_to_proteins, self.num_proteins
            )
            self.protein_edge_to_hyperedge = self.edge_to_hyperedge.copy()

            self.compound_pairwise_indices_tf = tf.constant(self.compound_pairwise_indices, dtype=tf.int64)
            self.protein_pairwise_indices_tf = tf.constant(self.protein_pairwise_indices, dtype=tf.int64)
            self.compound_edge_to_hyperedge_tf = tf.constant(self.compound_edge_to_hyperedge, dtype=tf.int32)
            self.protein_edge_to_hyperedge_tf = tf.constant(self.protein_edge_to_hyperedge, dtype=tf.int32)

        # Build network
        self.isTraining = tf.compat.v1.placeholder(tf.int32)
        self.isTraining = tf.cast(self.isTraining, tf.bool)
        initializer = tf.keras.initializers.glorot_normal()
        self.n_layer = 5
        self.weights = {}
        for i in range(self.n_layer):
            self.weights['layer_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='JU_%d' % (i + 1))
            self.weights['layer_1_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='JU_1_%d' % (i + 1))
            self.weights['layer_2_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='JU_2_%d' % (i + 1))
            self.weights['layer_att_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='layer_bias_%d' %(i+1))
        for i in range(2):
            self.weights['gating%d' % (i + 1)] = tf.Variable(initializer([self.emb_size, self.emb_size]),
                                                             name='g_W_%d_1' % (i + 1))
            self.weights['gating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.emb_size]),
                                                                  name='g_W_b_%d_1' % (i + 1))

        # 自适应团扩展的注意力权重和融合权重
        if self.use_clique_expansion:
            self.weights['clique_att_c'] = tf.Variable(
                initializer([self.emb_size, self.emb_size]),
                name='clique_att_compound'
            )
            self.weights['clique_att_p'] = tf.Variable(
                initializer([self.emb_size, self.emb_size]),
                name='clique_att_protein'
            )
            self.weights['fusion_c'] = tf.Variable(
                tf.constant(self.clique_fusion_init),
                name='fusion_weight_compound'
            )
            self.weights['fusion_p'] = tf.Variable(
                tf.constant(self.clique_fusion_init),
                name='fusion_weight_protein'
            )
            print("自适应团扩展权重初始化完成")

        def self_gating(em, channel):
            return tf.multiply(em, tf.nn.sigmoid(tf.matmul(em, self.weights['gating%d' % channel]) + self.weights['gating_bias%d' % channel]))

        compound_embeddings = self_gating(self.compound_embeddings, 1)
        all_compound_embeddings = [compound_embeddings]

        protein_embeddings = self.protein_embeddings
        all_protein_embeddings = [protein_embeddings]

        # 初始化草药和疾病节点的文本嵌入
        if self.use_text_embeddings:
            print("使用文本嵌入初始化草药和疾病节点...")
            herb_embeddings = tf.nn.l2_normalize(self.herb_text_embeddings, axis=1)
            disease_embeddings = tf.nn.l2_normalize(self.disease_text_embeddings, axis=1)
            all_herb_embeddings = [herb_embeddings]
            all_disease_embeddings = [disease_embeddings]
        else:
            print("警告：未使用文本嵌入，草药和疾病节点将通过超图卷积聚合")
            all_herb_embeddings = []
            all_disease_embeddings = []

        all_hc_embeddings = []
        all_pd_embeddings = []

        # GNN层
        for i in range(self.n_layer):
            # 通道1：中药-成分超图卷积
            new_hc_edge = tf.compat.v1.sparse_tensor_dense_matmul(H_e, compound_embeddings)
            new_compound_embeddings = tf.compat.v1.sparse_tensor_dense_matmul(H_n, new_hc_edge)

            # 通道2：疾病-靶点超图卷积
            new_pd_edge = tf.compat.v1.sparse_tensor_dense_matmul(P_e, protein_embeddings)
            new_protein_embeddings = tf.compat.v1.sparse_tensor_dense_matmul(P_n, new_pd_edge)

            # 自适应团扩展 - 成对边聚合
            if self.use_clique_expansion and len(self.compound_pairwise_indices) > 0:
                emb_i_c = tf.gather(compound_embeddings, self.compound_pairwise_indices_tf[:, 0])
                emb_j_c = tf.gather(compound_embeddings, self.compound_pairwise_indices_tf[:, 1])

                att_i_c = tf.matmul(emb_i_c, self.weights['clique_att_c'])
                att_j_c = tf.matmul(emb_j_c, self.weights['clique_att_c'])

                e_c = tf.reduce_sum(att_i_c * att_j_c, axis=1)
                e_c = tf.nn.leaky_relu(e_c)

                source_nodes_c = self.compound_pairwise_indices_tf[:, 0]
                num_nodes_c = self.num_compounds

                e_max_c = tf.math.unsorted_segment_max(e_c, source_nodes_c, num_segments=num_nodes_c)
                e_max_c = tf.where(tf.math.is_inf(e_max_c), tf.zeros_like(e_max_c), e_max_c)

                e_shifted_c = e_c - tf.gather(e_max_c, source_nodes_c)
                e_exp_c = tf.exp(e_shifted_c)

                e_sum_c = tf.math.unsorted_segment_sum(e_exp_c, source_nodes_c, num_segments=num_nodes_c)

                pairwise_weights_c = e_exp_c / (tf.gather(e_sum_c, source_nodes_c) + 1e-8)

                A_pairwise_c = tf.SparseTensor(
                    indices=self.compound_pairwise_indices_tf,
                    values=pairwise_weights_c,
                    dense_shape=[self.num_compounds, self.num_compounds]
                )

                pairwise_emb_c = tf.compat.v1.sparse_tensor_dense_matmul(A_pairwise_c, compound_embeddings)

                alpha_c = tf.nn.sigmoid(self.weights['fusion_c'])
                new_compound_embeddings = alpha_c * new_compound_embeddings + (1 - alpha_c) * pairwise_emb_c

            if self.use_clique_expansion and len(self.protein_pairwise_indices) > 0:
                emb_i_p = tf.gather(protein_embeddings, self.protein_pairwise_indices_tf[:, 0])
                emb_j_p = tf.gather(protein_embeddings, self.protein_pairwise_indices_tf[:, 1])

                att_i_p = tf.matmul(emb_i_p, self.weights['clique_att_p'])
                att_j_p = tf.matmul(emb_j_p, self.weights['clique_att_p'])

                e_p = tf.reduce_sum(att_i_p * att_j_p, axis=1)
                e_p = tf.nn.leaky_relu(e_p)

                source_nodes_p = self.protein_pairwise_indices_tf[:, 0]
                num_nodes_p = self.num_proteins

                e_max_p = tf.math.unsorted_segment_max(e_p, source_nodes_p, num_segments=num_nodes_p)
                e_max_p = tf.where(tf.math.is_inf(e_max_p), tf.zeros_like(e_max_p), e_max_p)

                e_shifted_p = e_p - tf.gather(e_max_p, source_nodes_p)
                e_exp_p = tf.exp(e_shifted_p)

                e_sum_p = tf.math.unsorted_segment_sum(e_exp_p, source_nodes_p, num_segments=num_nodes_p)

                pairwise_weights_p = e_exp_p / (tf.gather(e_sum_p, source_nodes_p) + 1e-8)

                A_pairwise_p = tf.SparseTensor(
                    indices=self.protein_pairwise_indices_tf,
                    values=pairwise_weights_p,
                    dense_shape=[self.num_proteins, self.num_proteins]
                )

                pairwise_emb_p = tf.compat.v1.sparse_tensor_dense_matmul(A_pairwise_p, protein_embeddings)

                alpha_p = tf.nn.sigmoid(self.weights['fusion_p'])
                new_protein_embeddings = alpha_p * new_protein_embeddings + (1 - alpha_p) * pairwise_emb_p

            # 更新化学成分和蛋白质嵌入
            compound_embeddings = tf.nn.leaky_relu(tf.matmul(new_compound_embeddings, self.weights['layer_%d' %(i+1)]) + compound_embeddings)
            protein_embeddings = tf.nn.leaky_relu(tf.matmul(new_protein_embeddings, self.weights['layer_1_%d' %(i+1)]) + protein_embeddings)

            compound_embeddings = tf.math.l2_normalize(compound_embeddings, axis=1)
            protein_embeddings = tf.math.l2_normalize(protein_embeddings, axis=1)
            new_hc_edge = tf.math.l2_normalize(new_hc_edge, axis=1)
            new_pd_edge = tf.math.l2_normalize(new_pd_edge, axis=1)

            # 更新草药和疾病节点嵌入
            if self.use_text_embeddings:
                if i == 0:
                    print(f"new_hc_edge shape: {new_hc_edge.shape}")
                    print(f"herb_embeddings shape: {herb_embeddings.shape}")
                    print(f"new_pd_edge shape: {new_pd_edge.shape}")
                    print(f"disease_embeddings shape: {disease_embeddings.shape}")

                herb_embeddings = tf.nn.leaky_relu(
                    tf.matmul(new_hc_edge, self.weights['layer_2_%d' %(i+1)]) + herb_embeddings
                )
                herb_embeddings = tf.math.l2_normalize(herb_embeddings, axis=1)

                disease_embeddings = tf.nn.leaky_relu(
                    tf.matmul(new_pd_edge, self.weights['layer_2_%d' %(i+1)]) + disease_embeddings
                )
                disease_embeddings = tf.math.l2_normalize(disease_embeddings, axis=1)

                all_herb_embeddings += [herb_embeddings]
                all_disease_embeddings += [disease_embeddings]

            all_compound_embeddings += [compound_embeddings]
            all_protein_embeddings += [protein_embeddings]
            all_hc_embeddings += [new_hc_edge]
            all_pd_embeddings += [new_pd_edge]

        compound_embeddings = tf.reduce_sum(all_compound_embeddings, axis=0)
        protein_embeddings = tf.reduce_sum(all_protein_embeddings, axis=0)
        hc_edge = tf.reduce_sum(all_hc_embeddings, axis=0)
        pd_edge = tf.reduce_sum(all_pd_embeddings, axis=0)

        # 聚合草药和疾病的多层嵌入
        if self.use_text_embeddings:
            herb_final = tf.reduce_sum(all_herb_embeddings, axis=0)
            disease_final = tf.reduce_sum(all_disease_embeddings, axis=0)
            print("使用文本增强的草药和疾病嵌入")
            self.final_hcedge = herb_final
            self.final_pdedge = disease_final
        else:
            self.final_hcedge = hc_edge
            self.final_pdedge = pd_edge

        self.neg_idx = tf.compat.v1.placeholder(tf.float32, name="neg_holder")
        self.neg_disease_embedding = tf.convert_to_tensor(self.neg_idx, dtype=tf.float32)

        self.final_iembedding = protein_embeddings
        self.final_uembedding = compound_embeddings

        self.u_embedding = tf.nn.embedding_lookup(self.final_hcedge, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.final_pdedge, self.v_idx)
