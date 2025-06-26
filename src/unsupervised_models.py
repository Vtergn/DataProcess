# src/unsupervised_models.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn # 新增
import torch.optim as optim # 新增
from torch.utils.data import TensorDataset, DataLoader # 新增
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy import stats

from src import config
from src.custom_models.unsupervised.autoencoder3d import (
    Autoencoder3D, load_autoencoder_model, extract_features_with_autoencoder
) # 导入 Autoencoder3D 类本身

# --- 辅助函数：获取聚类评估指标 (保持不变) ---
def _evaluate_clustering(features, labels):
    """
    计算并返回聚类的 Silhouette、Calinski-Harabasz 和 Davies-Bouldin 分数。
    
    Args:
        features (np.ndarray): 用于聚类的数据（通常是降维后的特征）。
        labels (np.ndarray): 聚类算法生成的标签。
    
    Returns:
        tuple: (Silhouette Score, Calinski-Harabasz Score, Davies-Bouldin Score)。
               如果聚类数量不满足计算条件，则返回 NaN。
    """
    n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        print("警告: 聚类数量少于2，无法计算 Silhouette, Calinski-Harabasz, Davies-Bouldin Score。返回 NaN。")
        return np.nan, np.nan, np.nan
        
    sil_score = silhouette_score(features, labels)
    ch_score = calinski_harabasz_score(features, labels)
    db_score = davies_bouldin_score(features, labels)

    return sil_score, ch_score, db_score

# --- 辅助函数：对多片段数据进行聚类投票 (保持不变) ---
def _cluster_data_with_voting(labels, patient_ids):
    """
    对于每个病人有多个数据片段的情况，通过投票机制确定最终的聚类标签。
    
    Args:
        labels (np.ndarray): 每个数据片段的聚类标签。
        patient_ids (np.ndarray): 每个数据片段对应的病人ID。
        
    Returns:
        tuple: (final_labels_for_all_segments (np.ndarray), unique_patient_labels (np.ndarray))
               第一个是每个片段的投票后标签，第二个是每个唯一病人的投票后标签。
    """
    patient_dict = {}
    
    for i, pid in enumerate(patient_ids):
        if pid not in patient_dict:
            patient_dict[pid] = []
        patient_dict[pid].append(labels[i])
    
    final_labels_for_all_segments = np.zeros_like(labels, dtype=int)
    unique_patient_labels = {}

    original_order = list(patient_ids)

    for i, pid in enumerate(original_order):
        if pid in patient_dict and len(patient_dict[pid]) > 0:
            most_common_label = stats.mode(patient_dict[pid], keepdims=False)[0]
            final_labels_for_all_segments[i] = most_common_label
            unique_patient_labels[pid] = most_common_label

    unique_patient_labels_array = np.array(list(unique_patient_labels.items()))
    return final_labels_for_all_segments, unique_patient_labels_array

# --- 新增：自编码器训练函数 ---
def _train_autoencoder(model, data_tensor, train_params, device):
    """
    训练 Autoencoder3D 模型。

    Args:
        model (Autoencoder3D): Autoencoder3D 模型实例。
        data_tensor (torch.Tensor): 用于训练的输入数据，应为 torch.float32 类型。
        train_params (dict): 训练参数字典 (epochs, learning_rate, batch_size)。
        device (torch.device): 模型训练所在的设备。

    Returns:
        Autoencoder3D: 训练后的模型实例。
    """
    epochs = train_params.get("epochs", 50)
    learning_rate = train_params.get("learning_rate", 0.001)
    batch_size = train_params.get("batch_size", 32)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 创建 DataLoader
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"开始训练 Autoencoder3D，Epochs: {epochs}, LR: {learning_rate}, Batch Size: {batch_size}")
    model.train() # 设置为训练模式
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data,) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            
            encoded, decoded = model(data)
            loss = criterion(decoded, data) # 重建损失
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1: # 每10个epoch打印一次
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    print("Autoencoder3D 训练完成。")
    model.eval() # 训练完成后设置为评估模式
    return model

# --- 核心函数：运行无监督学习分析 ---
def run_unsupervised_analysis(data_tensor, patient_ids_raw, model_name_key):
    """
    执行无监督学习的流程，包括特征提取/降维和聚类。

    Args:
        data_tensor (torch.Tensor): 原始输入数据 (例如来自 .npy 文件的 tensor)。
                                     形状应为 [N, C, D, H, W] 或 [N, C, H, W, D]。
        patient_ids_raw (np.ndarray): 与数据对应的病人ID数组。
        model_name_key (str): config.UNSUPERVISED_MODELS 中定义的模型名称键。

    Returns:
        dict: 包含无监督学习结果的字典，包括：
              - "model_name": 模型名称
              - "extracted_features": 从自编码器提取的原始特征
              - "reduced_features": 降维后的特征
              - "cluster_labels": 聚类结果标签 (针对每个数据片段)
              - "patient_cluster_labels": 每个唯一病人的投票后聚类标签
              - "evaluation_scores": 聚类评估指标字典
              - "reduced_features_df": 包含 Patient_ID, UMAP_1, UMAP_2, Cluster 的 DataFrame
    """
    if model_name_key not in config.UNSUPERVISED_MODELS:
        raise ValueError(f"模型 '{model_name_key}' 未在 config.UNSUPERVISED_MODELS 中定义。")

    model_config = config.UNSUPERVISED_MODELS[model_name_key]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- 开始运行 {model_name_key} 无监督学习流程 ---")
    print(f"当前计算设备: {device}")

    extracted_features = None
    reduced_features = None
    cluster_labels = None
    patient_cluster_labels = None
    evaluation_scores = {}
    reduced_features_df = None

    # --- 1. 特征提取 (例如 Autoencoder) ---
    if "feature_extractor" in model_config:
        extractor_config = model_config["feature_extractor"]
        extractor_type = extractor_config["type"]

        if extractor_type == "Autoencoder3D":
            latent_dim = extractor_config["latent_dim"]
            autoencoder_model = Autoencoder3D(latent_dim=latent_dim).to(device) # 初始化模型

            train_autoencoder = extractor_config.get("train_autoencoder", False)
            autoencoder_train_params = extractor_config.get("autoencoder_train_params", {})
            
            if train_autoencoder:
                print("根据配置，正在训练 Autoencoder3D...")
                autoencoder_model = _train_autoencoder(autoencoder_model, data_tensor, autoencoder_train_params, device)
                
                # 训练后保存模型
                if autoencoder_train_params.get("save_model_after_training", False):
                    save_filename = autoencoder_train_params.get("save_model_filename", f"trained_autoencoder_{model_name_key}.pth")
                    save_path = config.MODEL_OUTPUT_DIR / save_filename
                    torch.save(autoencoder_model.state_dict(), save_path)
                    print(f"训练后的 Autoencoder3D 模型已保存到: {save_path}")

            else:
                print("根据配置，正在加载预训练的 Autoencoder3D...")
                model_path = config.DATA_EXTERNAL_DIR / extractor_config["model_path"]
                autoencoder_model = load_autoencoder_model(str(model_path), latent_dim, device)
            
            # 提取特征
            extracted_features = extract_features_with_autoencoder(autoencoder_model, data_tensor, device)
            print(f"特征提取完成。形状: {extracted_features.shape}")
        else:
            print("未指定或不支持的特征提取器，将直接使用原始数据并展平作为特征。")
            extracted_features = data_tensor.cpu().numpy().reshape(len(data_tensor), -1)
    else:
        print("未指定特征提取器，将直接使用原始数据并展平作为特征。")
        extracted_features = data_tensor.cpu().numpy().reshape(len(data_tensor), -1)
        
    if extracted_features is None:
        raise ValueError("特征提取失败，extracted_features 为 None。请检查 feature_extractor 配置。")

    # --- 2. 降维 (例如 UMAP, PCA, t-SNE) ---
    if "dim_reducer" in model_config:
        reducer_config = model_config["dim_reducer"]
        reducer_type = reducer_config["type"]
        reducer_params = reducer_config.get("params", {})
        n_components = reducer_params.get("n_components", 2)

        print(f"正在使用 {reducer_type} 进行降维到 {n_components} 维...")
        if reducer_type == "UMAP":
            umap_model = umap.UMAP(n_components=n_components, random_state=config.RANDOM_STATE, **reducer_params)
            reduced_features = umap_model.fit_transform(extracted_features)
        elif reducer_type == "PCA":
            pca_model = PCA(n_components=n_components, random_state=config.RANDOM_STATE, **reducer_params)
            reduced_features = pca_model.fit_transform(extracted_features)
        elif reducer_type == "TSNE":
            tsne_model = TSNE(n_components=n_components, random_state=config.RANDOM_STATE, **reducer_params)
            reduced_features = tsne_model.fit_transform(extracted_features)
        else:
            print(f"警告: 不支持的降维方法 '{reducer_type}'。将不进行降维。")
            reduced_features = extracted_features # 不降维
    else:
        print("未指定降维方法。将使用提取的原始特征进行聚类。")
        reduced_features = extracted_features # 不降维

    # --- 3. 聚类 (例如 KMeans, DBSCAN) ---
    if "clusterer" in model_config:
        clusterer_config = model_config["clusterer"]
        clusterer_type = clusterer_config["type"]
        clusterer_params = clusterer_config.get("params", {})

        print(f"正在使用 {clusterer_type} 进行聚类...")
        if clusterer_type == "KMeans":
            if clusterer_config.get("find_optimal_k", False):
                print("正在通过肘部法则寻找 KMeans 的最佳 k 值...")
                wcss = []
                max_k = clusterer_config.get("max_k_for_elbow", 10)
                for i in range(1, max_k + 1):
                    kmeans = KMeans(n_clusters=i, random_state=config.RANDOM_STATE, n_init='auto', **clusterer_params)
                    kmeans.fit(reduced_features)
                    wcss.append(kmeans.inertia_)
                
                # 返回 WCSS 列表，以便 main.py 进行可视化
                from src.visualizer import plot_elbow_method # 临时导入，避免循环依赖
                # 肘部法则图的保存将在 main.py 中完成
                
                print("WCSS values for different K:")
                for k_val, wcss_val in enumerate(wcss):
                    print(f"K={k_val+1}: WCSS={wcss_val:.2f}")
                print("请根据肘部法则图手动确定最佳 K 值，并在 config 中设置 n_clusters。")
            else:
                wcss = None # 如果不进行肘部法则分析，则不返回 WCSS
            
            n_clusters = clusterer_params.get("n_clusters", 3)
            kmeans_model = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_STATE, n_init='auto', **clusterer_params)
            cluster_labels = kmeans_model.fit_predict(reduced_features)
            print(f"KMeans 聚类完成，聚类数量: {len(np.unique(cluster_labels))}")

        elif clusterer_type == "DBSCAN":
            if clusterer_config.get("search_optimal_params", False):
                print("正在通过网格搜索寻找 DBSCAN 的最佳参数 (eps, min_samples)...")
                best_score = -np.inf
                best_labels = None
                best_eps = None
                best_min_samples = None

                eps_list = np.linspace(clusterer_config.get("eps_min", 0.1), clusterer_config.get("eps_max", 1.0), clusterer_config.get("eps_steps", 10))
                min_samples_list = range(clusterer_config.get("min_samples_min", 5), clusterer_config.get("min_samples_max", 20) + 1)
                
                for eps in eps_list:
                    for min_samples in min_samples_list:
                        dbscan_model = DBSCAN(eps=eps, min_samples=min_samples, **clusterer_params)
                        current_labels = dbscan_model.fit_predict(reduced_features)
                        
                        n_clusters = len(np.unique(current_labels)) - (1 if -1 in current_labels else 0)
                        if n_clusters >= 2:
                            sil, ch, db = _evaluate_clustering(reduced_features, current_labels)
                            current_total_score = np.abs(sil) + 0.1 * np.abs(ch) + 0.1 * (1 / (db + 1e-6))
                            
                            if current_total_score > best_score:
                                best_score = current_total_score
                                best_labels = current_labels
                                best_eps = eps
                                best_min_samples = min_samples
                        else:
                            pass
                
                if best_labels is not None:
                    cluster_labels = best_labels + 1
                    print(f"DBSCAN 最佳参数: eps={best_eps:.2f}, min_samples={best_min_samples}")
                    print(f"最佳 DBSCAN 聚类数量: {len(np.unique(cluster_labels)) - (1 if 0 in cluster_labels else 0)}")
                    # 存储最佳参数，以便记录到性能报告
                    model_config["clusterer"]["best_params"] = {"eps": best_eps, "min_samples": best_min_samples}
                else:
                    print("DBSCAN 未找到有效的聚类参数组合。使用默认参数。")
                    dbscan_model = DBSCAN(**clusterer_params)
                    cluster_labels = dbscan_model.fit_predict(reduced_features) + 1
            else:
                dbscan_model = DBSCAN(**clusterer_params)
                cluster_labels = dbscan_model.fit_predict(reduced_features) + 1

            print(f"DBSCAN 聚类完成。初始聚类数量: {len(np.unique(cluster_labels)) - (1 if 0 in cluster_labels else 0)}")

        else:
            print(f"警告: 不支持的聚类方法 '{clusterer_type}'。将不进行聚类。")
            cluster_labels = np.zeros(len(reduced_features), dtype=int)

    else:
        print("未指定聚类方法。")
        cluster_labels = np.zeros(len(reduced_features), dtype=int)

    # --- 4. 聚类评估 ---
    if cluster_labels is not None and len(np.unique(cluster_labels)) > 1:
        sil, ch, db = _evaluate_clustering(reduced_features, cluster_labels)
        evaluation_scores = {
            "silhouette_score": sil,
            "calinski_harabasz_score": ch,
            "davies_bouldin_score": db
        }
        print("\n聚类评估结果:")
        for metric, score in evaluation_scores.items():
            print(f"  {metric}: {score:.4f}")
    else:
        print("无法进行聚类评估，聚类数量不足或没有聚类结果。")

    # --- 5. 聚合病人级别的聚类结果 (投票) ---
    final_segment_labels, unique_patient_labels_array = _cluster_data_with_voting(cluster_labels, patient_ids_raw)
    patient_cluster_labels = unique_patient_labels_array

    if reduced_features.ndim == 1:
        reduced_features_plot = reduced_features.reshape(-1, 1)
    else:
        reduced_features_plot = reduced_features

    df_columns = ["Patient_ID"] + [f"Component_{i+1}" for i in range(reduced_features_plot.shape[1])] + ["Cluster"]
    
    temp_data = np.hstack((patient_ids_raw.reshape(-1, 1), reduced_features_plot, final_segment_labels.reshape(-1, 1)))
    reduced_features_df = pd.DataFrame(temp_data, columns=df_columns)
    
    reduced_features_df['Cluster'] = reduced_features_df['Cluster'].astype(int)
    reduced_features_df['Patient_ID'] = reduced_features_df['Patient_ID'].astype(str)

    print(f"无监督学习流程完成。")

    return {
        "model_name": model_name_key,
        "extracted_features": extracted_features,
        "reduced_features": reduced_features,
        "cluster_labels": cluster_labels,
        "patient_cluster_labels": patient_cluster_labels,
        "evaluation_scores": evaluation_scores,
        "reduced_features_df": reduced_features_df,
        "wcss_values": wcss if 'wcss' in locals() else None # 返回 WCSS 列表以供 Elbow Method 可视化
    }