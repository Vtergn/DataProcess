# src/supervised_models.py

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report
)
import joblib
import os
import torch # 确保已导入 torch
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings

from src import config
from src.data_loader import save_model

# 导入你自定义的 LightGBM Focal Loss 训练函数
from src.custom_models.supervised.lightgbm_focal_loss import train_lightgbm_focal_loss

# 忽略 LightGBM 的特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="lightgbm")


# --- 评估函数 (可以保持不变,或移到 util 文件,但目前留在 supervised_models 便于调用) ---
def evaluate_classification_metrics(y_true, y_pred, y_pred_proba=None):
    """
    评估分类模型的性能,并返回详细指标。
    """
    metrics = {}
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    # Classification Report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    print("\n--- 调试: classification_report 完整输出 ---")
    print(report) # <--- **这是最重要的调试信息！请复制粘贴这部分输出给我！**
    print("------------------------------------------")


    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision_weighted"] = report['weighted avg']['precision']
    metrics["recall_weighted"] = report['weighted avg']['recall']
    metrics["f1_weighted"] = report['weighted avg']['f1-score']

    # 添加每个类别的详细指标
    for label, data in report.items():
        if label in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        metrics[f"Class_{label}_precision"] = data['precision']
        metrics[f"Class_{label}_recall"] = data['recall']
        metrics[f"Class_{label}_f1-score"] = data['f1-score']
        metrics[f"Class_{label}_support"] = data['support']

    # ROC AUC 只对二分类问题且有概率预测时计算
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        except ValueError as e:
            print(f"警告: 计算 ROC AUC 失败: {e}. 可能是只有单类存在。")
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan

    return metrics


# --- 主运行函数 `run_supervised_analysis` (重大修改) ---
def run_supervised_analysis(X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series,
                            model_name: str, use_k_fold_cv: bool = False,
                            device: torch.device = torch.device('cpu'),
                            label_name: str = None,
                            tuning_metric: str = "f1_score",
                            minority_class: int = 1):
    """
    运行指定模型的有监督学习分析,包括数据预处理、超参数调优和模型评估。
    新增label_name参数,用于多标签训练时区分输出。
    tuning_metric: 可选 f1_score/recall/precision/accuracy/roc_auc
    minority_class: 自动识别的少数类标签
    """
    model_config = config.SUPERVISED_MODELS.get(model_name)
    if not model_config:
        raise ValueError(f"配置中未找到模型 '{model_name}'。")

    model_type = model_config["type"]
    model_class = model_config.get("model_class")
    base_params = model_config.get("params", {})
    param_grid = model_config.get("param_grid", {})

    # 根据模式调整输出信息
    mode_text = "当前折叠" if use_k_fold_cv else "整体训练"
    print(f"--- 开始有监督分析: {model_name} ({mode_text}) ---")
    print(f"--- 运行在设备: {device} ---") # 打印当前运行设备

    # --- 新增：根据 label_name 过滤 NaN 标签的行 ---
    if label_name:
        # 检查训练集标签
        initial_train_samples = len(y_train)
        nan_train_indices = y_train.index[y_train.isna()]
        if not nan_train_indices.empty:
            X_train = X_train.drop(nan_train_indices)
            y_train = y_train.drop(nan_train_indices)
            print(f"--- 已从训练集移除 {len(nan_train_indices)} 个 '{label_name}' 标签为 NaN 的样本。")
            print(f"--- 训练集剩余样本数: {len(y_train)}")

        # 检查测试集标签
        initial_test_samples = len(y_test)
        nan_test_indices = y_test.index[y_test.isna()]
        if not nan_test_indices.empty:
            X_test = X_test.drop(nan_test_indices)
            y_test = y_test.drop(nan_test_indices)
            print(f"--- 已从测试集移除 {len(nan_test_indices)} 个 '{label_name}' 标签为 NaN 的样本。")
            print(f"--- 测试集剩余样本数: {len(y_test)}")
    else:
        print("--- 未指定 label_name，跳过 NaN 标签过滤。请确保你的标签没有缺失值。 ---")

    # 对训练和测试集进行副本操作,确保不修改原始数据 (在NaN过滤后进行副本，以避免对原始X_train/y_train的外部引用造成影响)
    X_train_processed = X_train.copy()
    y_train_processed = y_train.copy()
    X_test_processed = X_test.copy()
    y_test_processed = y_test.copy()


    # --- 特征标准化 (基于树的模型通常不需要) ---
    if model_name not in ["RandomForestClassifier", "XGBClassifier", "LightGBM_FocalLoss", "LGBMClassifier"]:
        print("--- 正在进行特征标准化 ---")
        scaler = StandardScaler()
        X_train_processed = pd.DataFrame(scaler.fit_transform(X_train_processed),
                                         columns=X_train.columns, index=X_train.index)
        X_test_processed = pd.DataFrame(scaler.transform(X_test_processed),
                                         columns=X_test.columns, index=X_test.index)
    else:
        print("--- 基于树的模型,跳过特征标准化 ---")

    # --- 针对 LightGBM_FocalLoss 的特殊处理 (调用外部自定义函数) ---
    if model_name == "LightGBM_FocalLoss":
        print(f"--- 正在使用外部自定义训练函数 for {model_name} ---")
        results_from_custom_trainer = train_lightgbm_focal_loss(
            X_train_processed,
            y_train_processed,
            X_test_processed,
            y_test_processed,
            model_config,
            config.RANDOM_STATE,
            device,
            tuning_metric,
            minority_class
        )

        best_model = results_from_custom_trainer["best_model"]
        best_params = results_from_custom_trainer["best_params"]
        y_pred_proba = results_from_custom_trainer["y_pred_prob"]
        best_threshold = results_from_custom_trainer["best_threshold"]

        y_pred = (y_pred_proba >= best_threshold).astype(int)

        metrics = evaluate_classification_metrics(y_test_processed, y_pred, y_pred_proba)

        feature_importances_fold = None
        if hasattr(best_model, 'feature_importance'):
            feature_importances_fold = best_model.feature_importance(importance_type='gain')

        print(f"\n{model_name} ({mode_text}) 最佳参数: {best_params}, 最佳阈值: {best_threshold:.2f}")

        # --- 新增: 打印特征重要性 ---
        if feature_importances_fold is not None and len(feature_importances_fold) > 0:
            feature_importance_series = pd.Series(feature_importances_fold, index=X_train.columns)
            print(f"\n--- 特征重要性 ({mode_text}) ---")
            print(feature_importance_series.sort_values(ascending=False).head(10)) # 打印前10个最重要的特征
            print("------------------------------")

        # --- 在这里使用 label_name 来保存模型 ---
        # 构造带有 label_name 的模型文件名
        model_filename = f"{model_name}"
        if label_name:
            model_filename += f"_{label_name}"
        # 如果是交叉验证模式，可能还需要加上折数或其他标识符
        # if use_k_fold_cv:
        #     model_filename += f"_fold{current_fold_idx}" # 假设你有一个 current_fold_idx 变量

        # 调用 save_model 函数来保存模型
        try:
            save_model(best_model, model_filename, path=config.MODELS_DIR)
            print(f"模型已保存为: {model_filename}.pkl 在 {config.MODELS_DIR} 目录下。")
        except Exception as e:
            print(f"警告: 保存模型 {model_filename} 失败: {e}")


        return {
            "model_name": model_name,
            "best_model": best_model,
            "best_params": best_params,
            "performance_metrics": metrics,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
            "X_test": X_test_processed,
            "y_test": y_test_processed,
            "feature_importances_fold": feature_importances_fold
        }

    # --- 其他模型的通用 GridSearchCV 流程 ---
    else:
        # 根据模式调整输出信息
        if use_k_fold_cv:
            print(f"--- 正在使用 GridSearchCV 进行超参数调优 (在当前训练集上进行 {config.N_SPLITS_GRIDSEARCH} 折交叉验证) ---")
        else:
            print(f"--- 正在使用 GridSearchCV 进行超参数调优 ---")

        # 根据模型名称和设备类型初始化模型
        if model_class:
            model = model_class(**base_params)
        else:
            if model_name == "RandomForestClassifier":
                model = RandomForestClassifier(**base_params)
            elif model_name == "LogisticRegression":
                model = LogisticRegression(solver='liblinear', **base_params)
            elif model_name == "SVC":
                model = SVC(probability=True, **base_params)
            elif model_name == "KNeighborsClassifier":
                model = KNeighborsClassifier(**base_params)
            elif model_name == "XGBClassifier":
                # 根据 device 设置 XGBoost 的 tree_method
                if device.type == 'cuda':
                    print(f"--- XGBoost 将使用 GPU ({device}) 进行训练 ---")
                    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                                              use_label_encoder=False, tree_method='gpu_hist',
                                              gpu_id=device.index if device.index is not None else 0, # 如果device.index为None,默认为0
                                              **base_params)
                else:
                    print(f"--- XGBoost 将使用 CPU ({device}) 进行训练 ---")
                    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                                              use_label_encoder=False, tree_method='hist', # CPU 默认
                                              **base_params)
            elif model_name == "LGBMClassifier":
                # 根据 device 设置 LightGBM 的 device
                if device.type == 'cuda':
                    print(f"--- LightGBM 将使用 GPU ({device}) 进行训练 ---")
                    # LightGBM 可以通过 'gpu' 参数启用GPU,通常会自动选择可用的GPU
                    # 如果需要指定特定GPU,可以使用 gpu_device_id
                    model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss',
                                               device='gpu', #gpu_device_id=device.index if device.index is not None else 0,
                                               **base_params)
                else:
                    print(f"--- LightGBM 将使用 CPU ({device}) 进行训练 ---")
                    model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss',
                                               device='cpu', # CPU 默认
                                               **base_params)
            else:
                raise ValueError(f"不支持或未定义的模型名称: {model_name}。请检查 SUPERVISED_MODELS 配置。")

        # 动态设置 scoring
        if tuning_metric in ["f1_score", "recall", "precision"]:
            scoring = f"{tuning_metric}_{minority_class}"
        else:
            scoring = tuning_metric if tuning_metric else model_config.get("scoring", "f1_weighted")

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=config.N_SPLITS_GRIDSEARCH,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train_processed, y_train_processed)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print(f"\n最佳参数 ({mode_text}): {best_params}")
        print(f"最佳交叉验证分数 ({scoring}) ({mode_text}): {best_score:.4f}")

        y_pred = best_model.predict(X_test_processed)
        y_pred_proba = None
        if hasattr(best_model, 'predict_proba'):
            # 对于二分类问题,predict_proba 返回 (n_samples, 2) 数组,我们需要正类的概率
            if y_test_processed.nunique() == 2:
                y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]
            else: # 多分类问题,需要为每个类别提供概率
                 y_pred_proba = best_model.predict_proba(X_test_processed)

        metrics = evaluate_classification_metrics(y_test_processed, y_pred, y_pred_proba)

        feature_importances_fold = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importances_fold = best_model.feature_importances_
        elif hasattr(best_model, 'coef_') and model_name == "LogisticRegression":
            feature_importances_fold = np.abs(best_model.coef_[0])

        # --- 打印特征重要性 ---
        if feature_importances_fold is not None and len(feature_importances_fold) > 0:
            feature_importance_series = pd.Series(feature_importances_fold, index=X_train.columns)
            print(f"\n--- 特征重要性 ({mode_text}) ---")
            print(feature_importance_series.sort_values(ascending=False).head(10)) # 打印前10个最重要的特征
            print("------------------------------")

        # --- 在这里使用 label_name 来保存模型 ---
        # 构造带有 label_name 的模型文件名
        model_filename = f"{model_name}"
        if label_name:
            model_filename += f"_{label_name}"
        # 如果是交叉验证模式，可能还需要加上折数或其他标识符
        # if use_k_fold_cv:
        #     model_filename += f"_fold{current_fold_idx}" # 假设你有一个 current_fold_idx 变量

        # 调用 save_model 函数来保存模型
        try:
            save_model(best_model, model_filename, path=config.MODELS_DIR)
            print(f"模型已保存为: {model_filename}.pkl 在 {config.MODELS_DIR} 目录下。")
        except Exception as e:
            print(f"警告: 保存模型 {model_filename} 失败: {e}")

        return {
            "model_name": model_name,
            "best_model": best_model,
            "best_params": best_params,
            "performance_metrics": metrics,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
            "X_test": X_test_processed,
            "y_test": y_test_processed,
            "feature_importances_fold": feature_importances_fold
        }