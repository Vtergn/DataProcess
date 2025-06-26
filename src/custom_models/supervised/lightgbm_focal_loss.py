# src/custom_models/supervised/lightgbm_focal_loss.py

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
import warnings
from sklearn.datasets import make_classification # 用于生成假数据
from sklearn.model_selection import train_test_split
import torch # 确保导入 torch，用于判断 device 类型

warnings.filterwarnings("ignore")

def focal_binary_object(alpha, gamma):
    """
    LightGBM 的自定义 Focal Loss 目标函数。
    它返回梯度 (grad) 和 Hessian (hess) 以供 LightGBM 优化。
    """
    def focal_loss_object(y_pred, dtrain): # LightGBM 传入的是原始分数，但接口习惯用 y_pred
        y_true = dtrain.get_label()
        # 将原始预测分数转换为概率
        y_pred_prob = 1.0 / (1.0 + np.exp(-y_pred)) # 内部使用 y_pred_prob 避免混淆
        
        # 计算Focal Loss的各项因子
        pt = np.where(y_true == 1, y_pred_prob, 1 - y_pred_prob)
        alpha_factor = np.where(y_true == 1, alpha, 1 - alpha)
        focal_weight = alpha_factor * np.power(1 - pt, gamma)
        
        grad = -(focal_weight * (y_true - y_pred_prob))
        hess = focal_weight * y_pred_prob * (1 - y_pred_prob)
        
        return grad, hess
    return focal_loss_object

# --- 自定义 LightGBM Focal Loss 评估指标 ---
def focal_metric_lgb(alpha, gamma):
    """
    LightGBM 的自定义 Focal Loss 评估指标。
    用于在训练过程中监控 Focal Loss 的值。
    """
    def focal_loss_metric(y_pred, dtrain): # LightGBM 传入的是原始分数，但接口习惯用 y_pred
        y_true = dtrain.get_label()
        # 将原始预测分数转换为概率
        y_pred_prob = 1.0 / (1.0 + np.exp(-y_pred)) # 内部使用 y_pred_prob 避免混淆
        
        # 计算Focal Loss
        pt = np.where(y_true == 1, y_pred_prob, 1 - y_pred_prob)
        alpha_factor = np.where(y_true == 1, alpha, 1 - alpha)
        focal_weight = alpha_factor * np.power(1 - pt, gamma)
        # 避免 log(0) 错误，对概率进行裁剪
        y_pred_clipped = np.clip(y_pred_prob, 1e-9, 1 - 1e-9)
        loss = -focal_weight * (y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        
        return "focal_loss", np.mean(loss), False
    return focal_loss_metric


def train_lightgbm_focal_loss(X_train, y_train, X_test, y_test, model_config, random_state, device, tuning_metric="f1_score", minority_class=1):
    """
    tuning_metric: 可选 f1_score/recall/precision/accuracy/roc_auc
    minority_class: 自动识别的少数类标签
    """
    # 从 model_config 中提取参数，并提供默认值
    focal_loss_params = model_config.get("focal_loss_params", {})
    lgb_base_params = model_config.get("params", {})
    
    # 获取config.py 中的参数列表
    alpha_list = focal_loss_params.get("alpha_list") 
    gamma_list = focal_loss_params.get("gamma_list")
    thresholds = focal_loss_params.get("thresholds")
    
    best_metric_val = -1.0 # 针对指标的最佳值
    best_model = None            # 训练好的最佳模型
    best_params_overall = {}     # 包含所有最佳参数的字典
    best_y_pred_prob = None      # 最佳模型在测试集上的预测概率
    best_threshold = 0.5         # 最佳分类阈值
    best_classification_report = {} # 最佳模型的分类报告

    # 将数据转换为 LightGBM 内部的 Dataset 格式
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

    print("开始 LightGBM Focal Loss 参数 (alpha, gamma, threshold) 调优...")

    # 遍历所有 Focal Loss 参数组合
    for alpha in alpha_list:
        for gamma in gamma_list:
            
            current_lgb_params = lgb_base_params.copy()
            
            # 合并您测试代码中硬编码的 LightGBM 参数，确保它们优先
            # 这样可以保持与您本地运行成功的代码行为一致
            current_lgb_params.update({
                'objective': 'binary', # 如果您的 model_config.params 中有 objective，这里会被覆盖
                'metric': 'None',      # 禁用默认评估指标
                'num_leaves': 31,
                'scale_pos_weight': 5, # 使用您测试代码中的硬编码值
                'verbosity': -1        # 设置 verbosity，与您测试代码保持一致
                 
            })
            
            # 设置随机种子
            current_lgb_params['random_state'] = random_state

            # >>> 根据 device 参数配置 LightGBM 的 GPU 使用 <<<
            if device.type == 'cuda':
                # print(f"尝试使用 GPU ({device}) 训练 LightGBM...")
                current_lgb_params['device'] = 'gpu' # 或 'gpu'
                # 如果有多个 GPU，可以指定 device.index
                if device.index is not None:
                    current_lgb_params['gpu_device_id'] = device.index
                # 对于某些LightGBM版本，可能需要设置'gpu_use_dp': True
                # current_lgb_params['gpu_use_dp'] = True 
            else:
                print("使用 CPU 训练 LightGBM...")
                current_lgb_params['device'] = 'cpu' # 明确设置为 cpu


            # 训练 LightGBM 模型
            try:
                # 获取 num_boost_round，如果 config.py 中有 n_estimators，则使用它，否则默认 100
                num_boost_round = current_lgb_params.get('n_estimators', 100) # 默认 100

                current_model = lgb.train(
                    current_lgb_params, # 所有的核心参数通过 params 传入
                    lgb_train,
                    num_boost_round=num_boost_round,
                    valid_sets=[lgb_eval],
                    fobj=focal_binary_object(alpha, gamma), # 自定义目标函数作为独立参数传入
                    feval=focal_metric_lgb(alpha, gamma),   # 自定义评估指标作为独立参数传入
                    verbose_eval=False # 不打印每次迭代的评估结果，作为独立参数传入
                )
            except Exception as e:
                print(f"LightGBM 训练失败 (alpha={alpha}, gamma={gamma}): {e}")
                import traceback
                traceback.print_exc()
                continue # 跳过当前组合

            # 获取当前模型在测试集上的预测概率
            # LightGBM 的 .predict() 默认返回原始分数，这里需要转换为概率
            current_y_pred_prob_raw = current_model.predict(X_test, raw_score=True)
            current_y_pred_prob = 1.0 / (1.0 + np.exp(-current_y_pred_prob_raw)) # Sigmoid 转换到概率

            # 在不同的分类阈值下评估模型性能 (这里以 F1-score for Class 1 作为优化目标)
            for thres in thresholds:
                current_y_pred_labels = (current_y_pred_prob >= thres).astype(int)
                
                # 针对 minority_class 计算指标
                if minority_class in y_test.unique() and minority_class in np.unique(current_y_pred_labels):
                    if tuning_metric == "f1_score":
                        metric_val = f1_score(y_test, current_y_pred_labels, pos_label=minority_class, average='binary', zero_division=0)
                    elif tuning_metric == "recall":
                        from sklearn.metrics import recall_score
                        metric_val = recall_score(y_test, current_y_pred_labels, pos_label=minority_class, average='binary', zero_division=0)
                    elif tuning_metric == "precision":
                        from sklearn.metrics import precision_score
                        metric_val = precision_score(y_test, current_y_pred_labels, pos_label=minority_class, average='binary', zero_division=0)
                    elif tuning_metric == "accuracy":
                        from sklearn.metrics import accuracy_score
                        metric_val = accuracy_score(y_test, current_y_pred_labels)
                    else:
                        metric_val = f1_score(y_test, current_y_pred_labels, pos_label=minority_class, average='binary', zero_division=0)
                else:
                    metric_val = 0.0

                # 更新最佳模型和参数
                if metric_val > best_metric_val:
                    best_metric_val = metric_val
                    best_model = current_model # 保存最佳模型对象
                    best_y_pred_prob = current_y_pred_prob # 保存最佳预测概率
                    best_threshold = thres # 保存最佳阈值

                    # 记录最佳参数，包括 Focal Loss 的 alpha, gamma 和最佳分类阈值
                    # 注意：这里我们保留了 params 中设置的 'verbosity'，因为您的测试代码中也保留了
                    best_params_overall = {
                        'alpha': alpha, 
                        'gamma': gamma, 
                        'threshold': thres, 
                        **current_lgb_params # 包含 LightGBM 的基础参数
                    }
                    # 重新生成并保存最佳分类报告
                    best_classification_report = classification_report(y_test, current_y_pred_labels, output_dict=True, zero_division=0)
                    
    # 如果没有找到有效的模型（例如，所有 F1-score 都为 0），则进行提示
    if best_model is None:
        print("警告: 未能找到有效的 LightGBM 模型或最佳 F1-score 始终为 -1.0 (可能因为类别不平衡或模型表现不佳)。")
        # 返回默认或空的结构，以便上层函数可以处理
        return {
            "best_model": None,
            "best_params": {},
            "y_pred_prob": None,
            "best_threshold": 0.5,
            "classification_report": {}
        }

    print(f"LightGBM Focal Loss 调优完成。最佳 F1-score (Class 1): {best_metric_val:.4f}, 最佳分类阈值: {best_threshold:.2f}")
    # print(f"最终使用的最佳参数: {best_params_overall}") # 如果需要，可以打印所有最佳参数

    # 返回训练结果，供 `supervised_models.py` 处理和报告
    return {
        "best_model": best_model,
        "best_params": best_params_overall, # 返回完整的最佳参数字典
        "y_pred_prob": best_y_pred_prob,
        "best_threshold": best_threshold,
        "classification_report": best_classification_report # 返回最佳分类报告
    }


if __name__ == "__main__":
    print("--- 正在进行 LightGBM Focal Loss 函数的独立调试 ---")

    # 1. 生成模拟数据
    # 为了模拟您真实的类不平衡数据，我们可以调整 weights
    X, y = make_classification(
        n_samples=400,         # 样本数量，与您约 400 行数据接近
        n_features=60,         # 特征数量，与您 60 列特征接近
        n_informative=10,      # 有信息量的特征数量
        n_redundant=0,         # 冗余特征数量
        n_repeated=0,          # 重复特征数量
        n_classes=2,           # 2分类
        n_clusters_per_class=1, # 每类一个聚类
        weights=[0.9, 0.1],    # 类别不平衡：90% 为类别 0，10% 为类别 1
        flip_y=0.01,           # 翻转标签的比例，引入少量噪声
        random_state=42        # 确保可复现性
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y, name='label')

    # 2. 分割训练集和测试集
    X_train_debug, X_test_debug, y_train_debug, y_test_debug = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y # 使用 stratify 保持类别比例
    )

    # 3. 准备 model_config (模拟 config.py 的输入)
    debug_model_config = {
        "focal_loss_params": {
            "alpha_list": [0.42, 0.5], # 调试时可以尝试多个值
            "gamma_list": [2, 5],      # 调试时可以尝试多个值
            "thresholds": [0.1, 0.2, 0.5] # 调试时可以尝试多个值
        },
        # "params": {} # 在此调试模式下，params 不会从 config 读取，而是硬编码在函数内部
    }

    debug_random_state = 42

    # 模拟 device 从 main_pipeline 传入
    # 你可以根据实际情况修改这个 device
    # debug_device = torch.device("cpu") # 示例：使用 CPU
    if torch.cuda.is_available():
        debug_device = torch.device("cuda:0") # 示例：使用第一个 GPU
        print("CUDA is available. Debugging with GPU.")
    else:
        debug_device = torch.device("cpu")
        print("CUDA not available. Debugging with CPU.")


    # 4. 调用要调试的函数
    results = train_lightgbm_focal_loss(
        X_train_debug, y_train_debug,
        X_test_debug, y_test_debug,
        debug_model_config,
        debug_random_state,
        debug_device, # 传递 device 参数
        tuning_metric="f1_score", # 指定 tuning_metric
        minority_class=1 # 指定 minority_class
    )

    # 5. 打印调试结果
    print("\n--- 调试结果 ---")
    if results["best_model"] is not None:
        print(f"最佳 Focal Loss 参数: {results['best_params']}")
        print(f"最佳分类阈值: {results['best_threshold']:.2f}")
        print("\n分类报告:")
        print(pd.DataFrame(results['classification_report']).transpose()) # 更美观地打印报告
    else:
        print("调试过程中未能成功训练模型。")

    print("\n--- 调试完成 ---")