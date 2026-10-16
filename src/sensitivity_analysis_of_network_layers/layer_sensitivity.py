"""
网络层数灵敏度分析
测试不同GNN层数(l)对HGHDA性能的影响
l的取值范围：{1, 2, 3, 4, 5, 6, 7, 8}
评估指标：AUROC, AUPRC, F1-score, 计算时间
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from HGHDA import HGHDA

class HGHDALayerSensitivity(HGHDA):
    """
    HGHDA的网络层数变体
    通过修改n_layer参数来测试不同层数的影响

    关键：必须在initModel()中的第303行之前设置n_layer
    """
    def __init__(self, conf, trainingSet=None, testSet=None, fold='[1]', n_layers=5):
        self.custom_n_layers = n_layers
        # 在父类初始化之前就设置n_layer，这样可以被父类使用
        self.n_layer = n_layers
        print(f"[层数敏感性分析] 初始化时设置GNN层数为: {n_layers}")
        super(HGHDALayerSensitivity, self).__init__(conf, trainingSet, testSet, fold)

    def initModel(self):
        """完全重写initModel，使用自定义层数而不是硬编码的5"""
        super(HGHDA, self).initModel()
        #Build adjacency matrix
        A = self.buildAdjacencyMatrix()
        cp=self.buildcpAdjacencyMatrix()
        pd=self.buildpdAdjacencyMatrix()
        hc=self.buildhcAdjacencyMatrix()

#####仅使用关联的
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
        pd_edge=temp1
        pd_node=temp2
        A_pde=pd_edge.tocoo()
        A_pdn=pd_node.tocoo()
        pdedge_indices = np.mat([A_pde.row, A_pde.col]).transpose()
        pdnode_indices = np.mat([A_pdn.row, A_pdn.col]).transpose()
        P_e = tf.SparseTensor(pdedge_indices, A_pde.data.astype(np.float32), A_pde.shape)
        P_n = tf.SparseTensor(pdnode_indices, A_pdn.data.astype(np.float32), A_pdn.shape)

        # ===== 新增：自适应团扩展初始化 =====
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
            print("自适应团扩展权重初始化完成")
        # ===== 团扩展初始化结束 =====

        self.u_idx = tf.placeholder(tf.int32, name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, name="v_idx")
        self.isTraining = tf.placeholder(tf.int32, name="training_flag")

        # ===== 关键修改：使用自定义层数而不是硬编码的5 =====
        self.n_layer = self.custom_n_layers
        print(f"[层数敏感性分析] 设置GNN层数为: {self.n_layer}")

        # 使用与原始HGHDA相同的初始化器
        initializer = tf.keras.initializers.glorot_normal()
        self.weights = dict()
        for i in range(self.n_layer):
            self.weights['layer_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='layer_bias_%d' %(i+1))
            self.weights['layer_1_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='layer_bias_1_%d' %(i+1))
            self.weights['layer_2_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='layer_bias_2_%d' %(i+1))
            self.weights['layer_att_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='layer_bias_%d' %(i+1))
        for i in range(2):
            self.weights['gating%d' % (i + 1)] = tf.Variable(initializer([self.emb_size, self.emb_size]),
                                                             name='g_W_%d_1' % (i + 1))
            self.weights['gating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.emb_size]),
                                                                  name='g_W_b_%d_1' % (i + 1))

        # ===== 新增：自适应团扩展的注意力权重和融合权重 =====
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
        # ===== 团扩展权重初始化结束 =====

        def self_gating(em,channel):
            return tf.multiply(em,tf.nn.sigmoid(tf.matmul(em,self.weights['gating%d' % channel])+self.weights['gating_bias%d' %channel]))

        compound_embeddings = self_gating(self.compound_embeddings,1)
        all_compound_embeddings = [compound_embeddings]

        protein_embeddings = self.protein_embeddings
        all_protein_embeddings = [protein_embeddings]

        # ===== 新增：初始化草药和疾病节点的文本嵌入 =====
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
        # ===== 文本嵌入初始化结束 =====

        all_hc_embeddings=[]
        all_pd_embeddings=[]

        # 使用GPU优化的稀疏矩阵乘法，避免转换为密集矩阵
        for i in range(self.n_layer):
            # 通道1：中药-成分超图卷积（使用稀疏矩阵乘法）
            new_hc_edge = tf.sparse_tensor_dense_matmul(H_e, compound_embeddings)
            new_compound_embeddings = tf.sparse_tensor_dense_matmul(H_n, new_hc_edge)

            # 通道2：疾病-靶点超图卷积（使用稀疏矩阵乘法）
            new_pd_edge = tf.sparse_tensor_dense_matmul(P_e, protein_embeddings)
            new_protein_embeddings = tf.sparse_tensor_dense_matmul(P_n, new_pd_edge)

            # ===== 新增：自适应团扩展 - 成对边聚合 =====
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
                e_max_c = tf.where(tf.is_inf(e_max_c), tf.zeros_like(e_max_c), e_max_c)

                e_shifted_c = e_c - tf.gather(e_max_c, source_nodes_c)
                e_exp_c = tf.exp(e_shifted_c)

                e_sum_c = tf.math.unsorted_segment_sum(e_exp_c, source_nodes_c, num_segments=num_nodes_c)

                pairwise_weights_c = e_exp_c / (tf.gather(e_sum_c, source_nodes_c) + 1e-8)

                A_pairwise_c = tf.SparseTensor(
                    indices=self.compound_pairwise_indices_tf,
                    values=pairwise_weights_c,
                    dense_shape=[self.num_compounds, self.num_compounds]
                )

                pairwise_emb_c = tf.sparse_tensor_dense_matmul(A_pairwise_c, compound_embeddings)

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
                e_max_p = tf.where(tf.is_inf(e_max_p), tf.zeros_like(e_max_p), e_max_p)

                e_shifted_p = e_p - tf.gather(e_max_p, source_nodes_p)
                e_exp_p = tf.exp(e_shifted_p)

                e_sum_p = tf.math.unsorted_segment_sum(e_exp_p, source_nodes_p, num_segments=num_nodes_p)

                pairwise_weights_p = e_exp_p / (tf.gather(e_sum_p, source_nodes_p) + 1e-8)

                A_pairwise_p = tf.SparseTensor(
                    indices=self.protein_pairwise_indices_tf,
                    values=pairwise_weights_p,
                    dense_shape=[self.num_proteins, self.num_proteins]
                )

                pairwise_emb_p = tf.sparse_tensor_dense_matmul(A_pairwise_p, protein_embeddings)

                alpha_p = tf.nn.sigmoid(self.weights['fusion_p'])
                new_protein_embeddings = alpha_p * new_protein_embeddings + (1 - alpha_p) * pairwise_emb_p
            # ===== 团扩展聚合结束 =====

            # 更新化学成分和蛋白质嵌入
            compound_embeddings =tf.nn.leaky_relu(tf.matmul(new_compound_embeddings,self.weights['layer_%d' %(i+1)])+ compound_embeddings)
            protein_embeddings = tf.nn.leaky_relu(tf.matmul(new_protein_embeddings,self.weights['layer_1_%d' %(i+1)])+ protein_embeddings)

            compound_embeddings = tf.math.l2_normalize(compound_embeddings,axis=1)
            protein_embeddings = tf.math.l2_normalize(protein_embeddings,axis=1)
            new_hc_edge=tf.math.l2_normalize(new_hc_edge,axis=1)
            new_pd_edge=tf.math.l2_normalize(new_pd_edge,axis=1)

            # ===== 新增：更新草药和疾病节点嵌入 =====
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
            # ===== 草药和疾病节点更新结束 =====

            all_compound_embeddings+=[compound_embeddings]
            all_protein_embeddings+=[protein_embeddings]
            all_hc_embeddings+=[new_hc_edge]
            all_pd_embeddings+=[new_pd_edge]

        compound_embeddings = tf.reduce_sum(all_compound_embeddings,axis=0)
        protein_embeddings = tf.reduce_sum(all_protein_embeddings, axis=0)
        hc_edge=tf.reduce_sum(all_hc_embeddings,axis=0)
        pd_edge=tf.reduce_sum(all_pd_embeddings,axis=0)

        # ===== 新增：聚合草药和疾病的多层嵌入 =====
        if self.use_text_embeddings:
            herb_final = tf.reduce_sum(all_herb_embeddings, axis=0)
            disease_final = tf.reduce_sum(all_disease_embeddings, axis=0)
            print("使用文本增强的草药和疾病嵌入")
            self.final_hcedge = herb_final
            self.final_pdedge = disease_final
        else:
            self.final_hcedge = hc_edge
            self.final_pdedge = pd_edge
        # ===== 文本嵌入聚合结束 =====

        self.neg_idx = tf.placeholder(tf.float32, name="neg_holder")
        self.neg_disease_embedding = tf.convert_to_tensor(self.neg_idx,dtype=tf.float32)

        self.final_iembedding = protein_embeddings
        self.final_uembedding = compound_embeddings

        self.u_embedding = tf.nn.embedding_lookup(self.final_hcedge, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.final_pdedge, self.v_idx)

    def readConfiguration(self):
        """重写readConfiguration，确保n_layer不被覆盖"""
        # 保存自定义层数
        saved_n_layer = self.custom_n_layers

        # 调用父类的readConfiguration
        super(HGHDALayerSensitivity, self).readConfiguration()

        # 恢复自定义层数（防止被父类覆盖）
        self.n_layer = saved_n_layer
        print(f"[层数敏感性分析] readConfiguration完成，确认层数: {self.n_layer}")
