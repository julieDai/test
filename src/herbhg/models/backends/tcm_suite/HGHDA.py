#coding:utf8
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from base.herbRecommender import herbRecommender
from scipy.sparse import coo_matrix,hstack
#import tensorflow as tf
import numpy as np
from math import sqrt
from time import strftime,localtime,time
import os
import random
from astra.data_validation import group_indexed_relation


def _safe_inverse_degrees(degrees):
    """Return inverse degrees while mapping isolated entries to zero."""
    degree_values = np.asarray(degrees, dtype=np.float32)
    inverse = np.zeros_like(degree_values)
    np.divide(1.0, degree_values, out=inverse, where=degree_values != 0)
    return inverse


class HGHDA(herbRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]',text_fusion_alpha=1.0):
        super(HGHDA, self).__init__(conf,trainingSet,testSet,fold)
        # 自适应团扩展配置
        self.use_clique_expansion = True  # 默认启用团扩展
        self.clique_fusion_init = 0.5  # 融合权重初始值
        # 文本增强分支融合系数。1.0 精确保持原模型，0.0 仅使用结构分支。
        self.text_fusion_alpha = float(text_fusion_alpha)
        if not 0.0 <= self.text_fusion_alpha <= 1.0:
            raise ValueError("text_fusion_alpha must be in [0, 1]")

    def buildAdjacencyMatrix(self):
        row, col, entries = [], [], []
        i=0
        for pair in self.data.trainingData:
            # symmetric matrix
            if int(pair[2])!=0:
                row += [self.data.herb[pair[0]]]
                col += [self.data.disease[pair[1]]]
                entries += [1]
                i+=1
        print('i======i',i)
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_herbs,self.num_diseases),dtype=np.float32)
        return u_i_adj

    def buildhcAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.hcassociation:
            # symmetric matrix
            #x=random.randint(0,1008)
            #y=random.randint(0,1192)
            #row += [x]
            #col += [y]
            row += [self.data.herb[pair[0]]]
            col += [self.data.compound[pair[1]]]
            entries += [1]
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_herbs,self.num_compounds),dtype=np.float32)
        return u_i_adj

    def buildcpAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.cpassociation:
            # symmetric matrix
            row += [self.data.compound[pair[0]]]
            col += [self.data.protein[pair[1]]]
            entries += [1]
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_compounds,self.num_proteins),dtype=np.float32)
        return u_i_adj
    def buildpdAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.pdassociation:
            # symmetric matrix
            #x=random.randint(0,7257)
            #y=random.randint(0,11070)
            #row += [x]
            #col += [y]
            row += [self.data.protein[pair[0]]]
            col += [self.data.disease[pair[1]]]
            entries += [1]
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_proteins,self.num_diseases),dtype=np.float32)
        return u_i_adj

    def build_hyperedge_node_mapping(self):
        """构建每个超边包含的节点列表（用于自适应团扩展）"""
        # 草药 → 化学组分列表
        herb_to_compounds = group_indexed_relation(
            self.data.hcassociation,
            group_column=0,
            member_column=1,
            member_to_index=self.data.compound,
            relation_name='H--C',
        )

        # 疾病 → 靶蛋白列表
        disease_to_proteins = group_indexed_relation(
            self.data.pdassociation,
            group_column=1,
            member_column=0,
            member_to_index=self.data.protein,
            relation_name='P--D',
        )

        return herb_to_compounds, disease_to_proteins

    def build_pairwise_edge_indices(self, hyperedge_to_nodes, num_nodes):
        """为超边内节点生成成对边索引（自适应团扩展）"""
        pairwise_rows = []
        pairwise_cols = []
        # 记录每条边所属的超边ID（用于归一化）
        edge_to_hyperedge = []

        # 为了避免边数量过大，限制每个超边最多处理一定数量的节点对
        max_pairs_per_edge = 100  # 每个超边最多处理100个节点对
        max_total_edges = 200000  # 全局最大边数限制（4090显卡可支持更大规模）

        hyperedge_id = 0
        for nodes in hyperedge_to_nodes.values():
            n_nodes = len(nodes)
            if n_nodes < 2:
                hyperedge_id += 1
                continue

            # 如果已达到全局边数限制，停止添加
            if len(pairwise_rows) >= max_total_edges:
                break

            # 生成节点对
            pair_count = 0
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    if pair_count >= max_pairs_per_edge or len(pairwise_rows) >= max_total_edges:
                        break
                    node_i, node_j = nodes[i], nodes[j]
                    # 双向边
                    pairwise_rows.extend([node_i, node_j])
                    pairwise_cols.extend([node_j, node_i])
                    # 记录边所属的超边（双向边属于同一超边）
                    edge_to_hyperedge.extend([hyperedge_id, hyperedge_id])
                    pair_count += 1
                if pair_count >= max_pairs_per_edge:
                    break

            hyperedge_id += 1

        # 创建索引数组（使用int64，TensorFlow SparseTensor要求）
        indices = np.array([pairwise_rows, pairwise_cols], dtype=np.int64).T
        # 保存边到超边的映射（用于后续归一化）
        self.edge_to_hyperedge = np.array(edge_to_hyperedge, dtype=np.int32)
        print(f'成对边数量: {len(indices)} (双向), 涉及超边数量: {hyperedge_id}')
        return indices
    def buildJointAdjacency(self):
        indices = [[self.data.herb[item[0]], self.data.item[item[1]]] for item in self.data.trainingData]
        values = [float(item[2]) / sqrt(len(self.data.trainSet_u[item[0]])) / sqrt(len(self.data.trainSet_i[item[1]]))
                  for item in self.data.trainingData]
        norm_adj = tf.SparseTensor(indices=indices, values=values,
                                   dense_shape=[self.num_herbs, self.num_diseases])
        return norm_adj

    def initModel(self):
        super(HGHDA, self).initModel()
        #Build adjacency matrix
        A = self.buildAdjacencyMatrix()
        #A=A.dot(A.transpose().dot(A))
        cp=self.buildcpAdjacencyMatrix()
        pd=self.buildpdAdjacencyMatrix()
        hc=self.buildhcAdjacencyMatrix()
        #print('hc-------------type-------------------')

        #print(type(hc))
        #print('len hlist',len(self.data.Hlist))
        #print('len clist',len(self.data.Clist))
        #print('len plist',len(self.data.Plist))
        #print('len dlist',len(self.data.Dlist))

        #hp=hc.dot(cp)
        #cd=cp.dot(pd)


#####仅使用关联的
        H_c = hc.transpose()
        D_hc_v = H_c.sum(axis=1).reshape(1, -1)
        D_hc_e = H_c.sum(axis=0).reshape(1, -1)
        temp1 = (H_c.multiply(_safe_inverse_degrees(D_hc_e))).transpose()
        temp2 = (H_c.transpose().multiply(_safe_inverse_degrees(D_hc_v))).transpose()
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
        temp1 = (P_d.multiply(_safe_inverse_degrees(D_P_e))).transpose()
        temp2 = (P_d.transpose().multiply(_safe_inverse_degrees(D_P_v))).transpose()
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
            # 构建超边-节点映射
            herb_to_compounds, disease_to_proteins = self.build_hyperedge_node_mapping()
            print(f"草药数量: {len(herb_to_compounds)}, 疾病数量: {len(disease_to_proteins)}")

            # 生成成对边索引（化学组分）
            self.compound_pairwise_indices = self.build_pairwise_edge_indices(
                herb_to_compounds, self.num_compounds
            )
            self.compound_edge_to_hyperedge = self.edge_to_hyperedge.copy()

            # 生成成对边索引（靶蛋白）
            self.protein_pairwise_indices = self.build_pairwise_edge_indices(
                disease_to_proteins, self.num_proteins
            )
            self.protein_edge_to_hyperedge = self.edge_to_hyperedge.copy()

            # 将索引转换为TensorFlow常量（SparseTensor要求int64）
            self.compound_pairwise_indices_tf = tf.constant(self.compound_pairwise_indices, dtype=tf.int64)
            self.protein_pairwise_indices_tf = tf.constant(self.protein_pairwise_indices, dtype=tf.int64)
            self.compound_edge_to_hyperedge_tf = tf.constant(self.compound_edge_to_hyperedge, dtype=tf.int32)
            self.protein_edge_to_hyperedge_tf = tf.constant(self.protein_edge_to_hyperedge, dtype=tf.int32)
        # ===== 团扩展初始化结束 =====



#####使用相似度矩阵的
#        self.compound_embeddings=coo_matrix(self.compound_embeddings,dtype=np.float32)
#        self.target_embeddings=coo_matrix(self.target_embeddings,dtype=np.float32)
#        print(type(self.compound_embeddings))
#        H_c = hc
#        temp1 = H_c.dot(self.compound_embeddings)
#        temp2 = temp1.dot(H_c.transpose())
#        temp2.tocoo()
#        D_hc=temp2.sum(axis=1).reshape(1,-1)
#        print(type(D_hc))
#        A_h = temp2.multiply(1.0/D_hc)
#        A_h = A_h.tocoo()
#        indices = np.mat([A_h.row, A_h.col]).transpose()
#        H_c = tf.SparseTensor(indices, A_h.data.astype(np.float32), A_h.shape)

#        P_d = pd.transpose()
#        temp1 = P_d.dot(self.target_embeddings)
#        temp2 = temp1.dot(P_d.transpose())
#        temp2.tocoo()
#        D_pd=temp2.sum(axis=1).reshape(1,-1)
#        A_pd = temp2.multiply(1.0/D_pd)
#        A_pd = A_pd.tocoo()
#        indices = np.mat([A_pd.row, A_pd.col]).transpose()
#        P_d = tf.SparseTensor(indices, A_pd.data.astype(np.float32), A_pd.shape)


###使用特征向量的
#        self.compound_embeddings=coo_matrix(self.compound_embeddings,dtype=np.float32)
#        self.target_embeddings=coo_matrix(self.target_embeddings,dtype=np.float32)
#        print(type(self.compound_embeddings))
#        H_c = hc
#        temp1 = H_c.dot(self.compound_embeddings)
#        temp2 = temp1.dot(temp1.transpose())
#        temp2.tocoo()
#        D_hc=temp2.sum(axis=1).reshape(1,-1)
#        print(type(D_hc))
#        A_h = temp2.multiply(1.0/D_hc)
#        A_h = A_h.tocoo()
#        indices = np.mat([A_h.row, A_h.col]).transpose()
#        H_c = tf.SparseTensor(indices, A_h.data.astype(np.float32), A_h.shape)

#        P_d = pd.transpose()
#        temp1 = P_d.dot(self.target_embeddings)
#        temp2 = temp1.dot(temp1.transpose())
#        temp2.tocoo()
#        D_pd=temp2.sum(axis=1).reshape(1,-1)
#        A_pd = temp2.multiply(1.0/D_pd)
#        A_pd = A_pd.tocoo()
#        indices = np.mat([A_pd.row, A_pd.col]).transpose()
#        P_d = tf.SparseTensor(indices, A_pd.data.astype(np.float32), A_pd.shape)




        #Build incidence matrix
        #H_u = hstack([A,A.dot(A.transpose().dot(A))])
#        H_u = A
#        D_u_v = H_u.sum(axis=1).reshape(1,-1)
#        D_u_e = H_u.sum(axis=0).reshape(1,-1)
#        temp1 = (H_u.transpose().multiply(np.sqrt(1.0/D_u_v))).transpose()
#        temp2 = temp1.transpose()
#        A_u = temp1.multiply(1.0/D_u_e).dot(temp2)
#        A_u = A_u.tocoo()
#        indices = np.mat([A_u.row, A_u.col]).transpose()
#        H_u = tf.SparseTensor(indices, A_u.data.astype(np.float32), A_u.shape)

#        H_i = A.transpose()
#        D_i_v = H_i.sum(axis=1).reshape(1,-1)
#        D_i_e = H_i.sum(axis=0).reshape(1,-1)
#        temp1 = (H_i.transpose().multiply(np.sqrt(1.0 / D_i_v))).transpose()
#        temp2 = temp1.transpose()
#        A_i = temp1.multiply(1.0 / D_i_e).dot(temp2)
#        A_i = A_i.tocoo()
#        indices = np.mat([A_i.row, A_i.col]).transpose()
#        H_i = tf.SparseTensor(indices, A_i.data.astype(np.float32), A_i.shape)

        #Build network
        self.isTraining = tf.placeholder(tf.int32)
        self.isTraining = tf.cast(self.isTraining, tf.bool)
        #initializer = tf.contrib.layers.xavier_initializer()
        initializer =tf.keras.initializers.glorot_normal()
        self.n_layer = 5
        self.weights={}
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

        # ===== 新增：自适应团扩展的注意力权重和融合权重 =====
        if self.use_clique_expansion:
            # 注意力权重矩阵（用于计算成对边权重）
            self.weights['clique_att_c'] = tf.Variable(
                initializer([self.emb_size, self.emb_size]),
                name='clique_att_compound'
            )
            self.weights['clique_att_p'] = tf.Variable(
                initializer([self.emb_size, self.emb_size]),
                name='clique_att_protein'
            )
            # 融合权重（控制超图卷积和成对边聚合的比例）
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
            # 草药节点初始嵌入：使用文本嵌入并归一化
            herb_embeddings = tf.nn.l2_normalize(self.herb_text_embeddings, axis=1)
            # 疾病节点初始嵌入：使用文本嵌入并归一化
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
                # 计算化学组分之间的成对边权重（注意力机制）
                emb_i_c = tf.gather(compound_embeddings, self.compound_pairwise_indices_tf[:, 0])
                emb_j_c = tf.gather(compound_embeddings, self.compound_pairwise_indices_tf[:, 1])

                # 注意力权重计算：使用可学习的权重矩阵
                att_i_c = tf.matmul(emb_i_c, self.weights['clique_att_c'])
                att_j_c = tf.matmul(emb_j_c, self.weights['clique_att_c'])

                # 边权重 = 注意力点积 + LeakyReLU（类似GAT）
                e_c = tf.reduce_sum(att_i_c * att_j_c, axis=1)
                e_c = tf.nn.leaky_relu(e_c)

                # ===== 按节点softmax归一化（手动实现，不依赖排序）=====
                # 对每个源节点，计算其所有出边的softmax
                # softmax(e_ij) = exp(e_ij) / sum_k(exp(e_ik))
                source_nodes_c = self.compound_pairwise_indices_tf[:, 0]
                num_nodes_c = self.num_compounds

                # 计算每个节点的最大e值（用于数值稳定性）
                e_max_c = tf.math.unsorted_segment_max(e_c, source_nodes_c, num_segments=num_nodes_c)
                e_max_c = tf.where(tf.is_inf(e_max_c), tf.zeros_like(e_max_c), e_max_c)  # 处理无出边的节点

                # 减去最大值并计算exp
                e_shifted_c = e_c - tf.gather(e_max_c, source_nodes_c)
                e_exp_c = tf.exp(e_shifted_c)

                # 计算每个节点的exp和
                e_sum_c = tf.math.unsorted_segment_sum(e_exp_c, source_nodes_c, num_segments=num_nodes_c)

                # 计算softmax权重
                pairwise_weights_c = e_exp_c / (tf.gather(e_sum_c, source_nodes_c) + 1e-8)
                # ===== 归一化结束 =====

                # 构建成对边邻接矩阵（稀疏）
                A_pairwise_c = tf.SparseTensor(
                    indices=self.compound_pairwise_indices_tf,
                    values=pairwise_weights_c,
                    dense_shape=[self.num_compounds, self.num_compounds]
                )

                # 成对边聚合
                pairwise_emb_c = tf.sparse_tensor_dense_matmul(A_pairwise_c, compound_embeddings)

                # 加权融合：超图卷积 + 成对边聚合
                alpha_c = tf.nn.sigmoid(self.weights['fusion_c'])  # 融合权重通过sigmoid限制在[0,1]
                new_compound_embeddings = alpha_c * new_compound_embeddings + (1 - alpha_c) * pairwise_emb_c

            if self.use_clique_expansion and len(self.protein_pairwise_indices) > 0:
                # 计算靶蛋白之间的成对边权重（注意力机制）
                emb_i_p = tf.gather(protein_embeddings, self.protein_pairwise_indices_tf[:, 0])
                emb_j_p = tf.gather(protein_embeddings, self.protein_pairwise_indices_tf[:, 1])

                # 注意力权重计算
                att_i_p = tf.matmul(emb_i_p, self.weights['clique_att_p'])
                att_j_p = tf.matmul(emb_j_p, self.weights['clique_att_p'])

                # 边权重
                e_p = tf.reduce_sum(att_i_p * att_j_p, axis=1)
                e_p = tf.nn.leaky_relu(e_p)

                # ===== 按节点softmax归一化（手动实现）=====
                source_nodes_p = self.protein_pairwise_indices_tf[:, 0]
                num_nodes_p = self.num_proteins

                e_max_p = tf.math.unsorted_segment_max(e_p, source_nodes_p, num_segments=num_nodes_p)
                e_max_p = tf.where(tf.is_inf(e_max_p), tf.zeros_like(e_max_p), e_max_p)

                e_shifted_p = e_p - tf.gather(e_max_p, source_nodes_p)
                e_exp_p = tf.exp(e_shifted_p)

                e_sum_p = tf.math.unsorted_segment_sum(e_exp_p, source_nodes_p, num_segments=num_nodes_p)

                pairwise_weights_p = e_exp_p / (tf.gather(e_sum_p, source_nodes_p) + 1e-8)
                # ===== 归一化结束 =====

                # 构建成对边邻接矩阵
                A_pairwise_p = tf.SparseTensor(
                    indices=self.protein_pairwise_indices_tf,
                    values=pairwise_weights_p,
                    dense_shape=[self.num_proteins, self.num_proteins]
                )

                # 成对边聚合
                pairwise_emb_p = tf.sparse_tensor_dense_matmul(A_pairwise_p, protein_embeddings)

                # 加权融合
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
                # 打印维度信息（仅第一层）
                if i == 0:
                    print(f"new_hc_edge shape: {new_hc_edge.shape}")
                    print(f"herb_embeddings shape: {herb_embeddings.shape}")
                    print(f"new_pd_edge shape: {new_pd_edge.shape}")
                    print(f"disease_embeddings shape: {disease_embeddings.shape}")

                # 草药节点通过超图卷积聚合化学成分信息
                # new_hc_edge 实际上就是草药节点的表示（聚合了其成分）
                # 将文本信息与图结构信息融合
                herb_embeddings = tf.nn.leaky_relu(
                    tf.matmul(new_hc_edge, self.weights['layer_2_%d' %(i+1)]) + herb_embeddings
                )
                herb_embeddings = tf.math.l2_normalize(herb_embeddings, axis=1)

                # 疾病节点通过超图卷积聚合靶点信息
                # new_pd_edge 实际上就是疾病节点的表示（聚合了其靶点）
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
            print("使用文本增强的草药和疾病嵌入，融合系数 alpha=%.2f" % self.text_fusion_alpha)
            # 在纯结构分支与原始文本增强分支之间做受控融合。
            # alpha=1.0 与修改前完全相同；alpha=0.0 不让文本分支进入预测表示。
            alpha = tf.constant(self.text_fusion_alpha, dtype=tf.float32)
            self.final_hcedge = (1.0 - alpha) * hc_edge + alpha * herb_final
            self.final_pdedge = (1.0 - alpha) * pd_edge + alpha * disease_final
        else:
            # 原有逻辑：使用超图边嵌入
            self.final_hcedge = hc_edge
            self.final_pdedge = pd_edge
        # ===== 文本嵌入聚合结束 =====


        self.neg_idx = tf.placeholder(tf.float32, name="neg_holder")

        self.neg_disease_embedding = tf.convert_to_tensor(self.neg_idx,dtype=tf.float32)
        #self.neg_disease_embedding = tf.nn.embedding_lookup(tf.convert_to_tensor(A.toarray(),dtype=tf.float32), self.u_idx)

        self.final_iembedding = protein_embeddings
        self.final_uembedding = compound_embeddings

        self.u_embedding = tf.nn.embedding_lookup(self.final_hcedge, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.final_pdedge, self.v_idx)
        #self.v_embedding = self.final_pdedge

    def trainModel(self):

	#quanjujiaocha
        #sigmoid_output=tf.sigmoid(tf.matmul(self.u_embedding, self.v_embedding, transpose_b=True))
        #y=tf.reduce_sum(tf.multiply(self.neg_disease_embedding,tf.log(sigmoid_output))+tf.multiply((1-self.neg_disease_embedding),tf.log(1-sigmoid_output)),1)

	#jiaochashang
        sigmoid_output=tf.sigmoid(tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding),1))
        # 添加数值稳定性：限制sigmoid输出范围，避免log(0)
        sigmoid_output = tf.clip_by_value(sigmoid_output, 1e-7, 1.0 - 1e-7)
        #print("loss compute")
        #print(sigmoid_output)
        #print(self.neg_disease_embedding)
        y=tf.reduce_sum(tf.multiply(self.neg_disease_embedding,tf.log(sigmoid_output))+tf.multiply((1-self.neg_disease_embedding),tf.log(1-sigmoid_output)),0)


        reg_loss = 0
        for key in self.weights:
            reg_loss += 0.001*tf.nn.l2_loss(self.weights[key])
        reg_loss += self.regU * (tf.nn.l2_loss(self.final_hcedge) + tf.nn.l2_loss(self.final_pdedge))

        loss = -tf.reduce_sum(y)+reg_loss
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                herb_idx, i_idx, j_idx = batch
                _, l = self.sess.run([train, loss],
                                feed_dict={self.u_idx: herb_idx, self.neg_idx: j_idx, self.v_idx: i_idx,self.isTraining:1})
                # Per-batch printing can fill a long-running PTY and block GPU
                # training. Keep enough progress information for monitoring.
                if n % 100 == 0:
                    print('training:', epoch + 1, 'batch', n, 'loss:', l, flush=True)
        self.u,self.i,self.weight=self.sess.run([self.final_hcedge,self.final_pdedge,self.weights],feed_dict={self.isTraining:0})
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        output_dir = os.environ.get(
            'ASTRA_OUTPUT_DIR',
            os.environ.get('CDRUG_OUTPUT_DIR', './results'),
        )
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(os.path.join(output_dir, 'herbedgeDHCN_herb_embedding'+currentTime+'.txt'),self.u)
        np.savetxt(os.path.join(output_dir, 'diseaseedgeDHCN_disease_embedding'+currentTime+'.txt'),self.i)
    def predictForRanking(self):
        print('ASTRA prediction ------------------------------------------------------------------------')
        return self.u.dot(self.i.transpose())
