# src/pipelines/supervised_pipeline.py

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import joblib
import os
import shap
import torch
from pathlib import Path
from openpyxl import load_workbook

# 确保所有可能用到的模型类都已导入
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src import config
# 确保 run_supervised_analysis 可以接收 device 参数
from src.supervised_models import run_supervised_analysis, evaluate_classification_metrics
from src.visualizer import plot_and_save_figure, plot_confusion_matrix, plot_feature_importance, \
                            plot_shap_summary, plot_roc_pr_curve

def run_supervised_pipeline(
    X_supervised_df: pd.DataFrame,
    y_supervised_processed: pd.Series,
    supervised_models_to_run: list,
    experiment_name: str,
    clean_file_name: str,
    clean_sheet_name: str,
    use_k_fold_cv: bool = False,
    n_splits_supervised: int = 5,
    test_size_overall: float = 0.2,
    random_state: int = 42,
    device: torch.device = torch.device("cpu"),
    label_name: str = None,
    supervised_tuning_metric: str = None
):
    """
    执行完整的有监督学习管道，根据参数决定 K 折交叉验证或整体训练。
    新增label_name参数，用于多标签训练时区分输出。

    Args:
        X_supervised_df (pd.DataFrame): 完整的特征数据集。
        y_supervised_processed (pd.Series): 完整的标签数据集。
        supervised_models_to_run (list): 要运行的模型名称列表。
        experiment_name (str): 当前实验的名称。
        clean_file_name (str): 清理后的数据文件名，用于命名。
        clean_sheet_name (str): 清理后的数据工作表名，用于命名。
        use_k_fold_cv (bool): 是否使用 K 折交叉验证。
        n_splits_supervised (int): K 折交叉验证的折数。
        test_size_overall (float): 整体训练/测试划分的测试集比例。
        random_state (int): 随机种子。
        device (torch.device): 指定运行计算的设备 (CPU 或 CUDA)。
        label_name (str): 标签名称，用于多标签训练时区分输出。
        supervised_tuning_metric (str): 监督学习调参指标。

    Returns:
        tuple: (all_performance_results, all_figure_paths)
    """
    print(f"\n--- 正在运行有监督学习管道 ---")
    if label_name is not None:
        print(f"--- 当前标签: {label_name} ---")

    # --- 开始添加 NaN 过滤逻辑 ---
    # 找到当前标签 y_supervised_processed 中为 NaN 的行索引
    nan_indices_in_y = y_supervised_processed.index[y_supervised_processed.isna()]

    # 打印发现的 NaN 数量
    if not nan_indices_in_y.empty:
        print(f"--- 标签 '{label_name}' 发现 {len(nan_indices_in_y)} 个 NaN 值，正在移除对应样本。")
        # 从 X 和 y 中移除这些行
        X_filtered = X_supervised_df.drop(nan_indices_in_y)
        y_filtered = y_supervised_processed.drop(nan_indices_in_y)
        print(f"--- 移除 NaN 后，数据集样本数: {len(y_filtered)}")
    else:
        X_filtered = X_supervised_df
        y_filtered = y_supervised_processed
        print(f"--- 标签 '{label_name}' 没有发现 NaN 值。")

    # 如果过滤后数据为空，则跳过整个管道
    if len(y_filtered) == 0:
        print(f"警告: 标签 '{label_name}' 过滤 NaN 后没有剩余样本，跳过该标签的训练。")
        return [], [] # 返回空结果

    # 确保标签是数值类型，并且如果使用分层抽样，确保至少有两个类别
    if not np.issubdtype(y_filtered.dtype, np.number):
        print(f"警告: 标签 '{label_name}' 不是数值类型，尝试转换为数值。")
        # 尝试转换为数值，非数值将变为 NaN，但我们已经处理了原始 NaN
        y_filtered = pd.to_numeric(y_filtered, errors='coerce')
        # 再次检查转换后是否引入了新的 NaN，并再次过滤
        nan_indices_after_convert = y_filtered.index[y_filtered.isna()]
        if not nan_indices_after_convert.empty:
            print(f"--- 标签 '{label_name}' 转换后发现 {len(nan_indices_after_convert)} 个 NaN 值，正在移除对应样本。")
            X_filtered = X_filtered.drop(nan_indices_after_convert)
            y_filtered = y_filtered.drop(nan_indices_after_convert)
            if len(y_filtered) == 0:
                print(f"警告: 标签 '{label_name}' 转换并过滤 NaN 后没有剩余样本，跳过该标签的训练。")
                return [], [] # 返回空结果

    # 如果进行分层抽样，确保标签至少有两个类别
    stratify_param = None
    if y_filtered.nunique() >= 2: # 只有当y有至少2个唯一值时才考虑分层
        stratify_param = y_filtered
    else:
        print(f"警告: 标签 '{label_name}' 仅有 {y_filtered.nunique()} 个唯一类别，无法进行分层抽样。将使用非分层抽样。")

    # --- 结束添加 NaN 过滤逻辑 ---

    all_performance_results = []
    all_figure_paths = []

    # --- 关键修正：确保 'splits' 在两个分支中都被定义 ---
    splits = [] # 在外部初始化 splits 变量

    for model_name in supervised_models_to_run:
        print(f"\n--- 针对模型: {model_name} ---")
        # 自动识别少数类
        class_counts = y_filtered.value_counts()
        if len(class_counts) < 2:
            print(f"警告: 标签 '{label_name}' 类别数不足2，跳过该模型。")
            continue
        minority_class = class_counts.idxmin()
        print(f"自动识别到少数类为: {minority_class}")
        # 传递调参指标
        tuning_metric = supervised_tuning_metric or "f1_score"

        # 初始化 OOF 预测数组，基于过滤后的数据集大小
        oof_y_pred = np.full(len(y_filtered), np.nan, dtype=float)
        oof_y_pred_proba = np.full(len(y_filtered), np.nan, dtype=float)

        all_fold_shap_values = []
        all_fold_X_test_for_shap = []
        fold_expected_values_list = []
        fold_feature_importances = []

        if use_k_fold_cv:
            print(f"--- 正在进行 {n_splits_supervised} 折交叉验证 ---")
            # 使用过滤后的数据进行 StratifiedKFold
            skf = StratifiedKFold(n_splits=n_splits_supervised, shuffle=True, random_state=random_state)
            splits = skf.split(X_filtered, y_filtered)
            num_iterations = n_splits_supervised
        else:
            print("--- 正在使用整体训练集训练，并使用测试集评估 ---")
            try:
                # 使用过滤后的数据进行 train_test_split
                X_train_overall, X_test_overall, y_train_overall, y_test_overall = train_test_split(
                    X_filtered, y_filtered,
                    test_size=test_size_overall,
                    random_state=random_state,
                    stratify=stratify_param # 这里 stratify_param 已经是过滤并检查后的
                )
                num_iterations = 1
                # 为单次训练/测试分割创建 'splits' 列表
                # 注意：这里我们使用内部索引，因为 train_test_split 已经返回了子集
                train_idx_overall = X_train_overall.index.tolist()
                test_idx_overall = X_test_overall.index.tolist()
                
                # splits 现在是一个列表，包含一个元组，代表一个"折叠"
                # 这个元组中的索引是原始 X_filtered/y_filtered 的 .loc 索引
                splits = [(
                    y_filtered.index.get_indexer(train_idx_overall), # 获取在y_filtered中的整数位置
                    y_filtered.index.get_indexer(test_idx_overall)   # 获取在y_filtered中的整数位置
                )]
                
            except ValueError as e:
                print(f"错误: 标签 '{label_name}' 无法进行 train_test_split: {e}")
                print(f"可能原因: 分层抽样 (stratify=True) 时，某个类别在测试集中比例过小或缺失。")
                print(f"尝试将 stratify_param 设置为 None，或者调整 test_size_overall。")
                all_performance_results.append({"model_name": model_name, "error": str(e)})
                continue # 跳过当前模型，处理下一个

        for fold, (train_idx_inner, test_idx_inner) in enumerate(splits):
            if use_k_fold_cv:
                # 使用 .iloc 和 .index 来确保正确地从原始过滤后的 DataFrame/Series 中提取数据
                X_train_fold, X_test_fold = X_filtered.iloc[train_idx_inner], X_filtered.iloc[test_idx_inner]
                y_train_fold, y_test_fold = y_filtered.iloc[train_idx_inner], y_filtered.iloc[test_idx_inner]
                print(f"--- 模型 {model_name} - 折叠 {fold + 1}/{num_iterations} ---")
                # current_test_global_indices = X_filtered.iloc[test_idx_inner].index.tolist() # 获取原始数据帧中的索引
            else:
                X_train_fold, X_test_fold = X_train_overall, X_test_overall
                y_train_fold, y_test_fold = y_train_overall, y_test_overall
                print(f"--- 模型 {model_name} - 整体训练与评估 ---")
                # current_test_global_indices = X_test_overall.index.tolist() # 获取原始数据帧中的索引

            try:
                # 确保数据类型正确 (虽然上面已经过滤，但这里做一次防御性检查)
                if not isinstance(X_train_fold, pd.DataFrame):
                    X_train_fold = pd.DataFrame(X_train_fold, columns=X_filtered.columns)
                if not isinstance(X_test_fold, pd.DataFrame):
                    X_test_fold = pd.DataFrame(X_test_fold, columns=X_filtered.columns)
                if not isinstance(y_train_fold, pd.Series):
                    y_train_fold = pd.Series(y_train_fold, index=X_train_fold.index) # 保持索引一致
                if not isinstance(y_test_fold, pd.Series):
                    y_test_fold = pd.Series(y_test_fold, index=X_test_fold.index) # 保持索引一致

                result = run_supervised_analysis(
                    X_train_fold,
                    y_train_fold,
                    X_test_fold,
                    y_test_fold,
                    model_name,
                    use_k_fold_cv=use_k_fold_cv,
                    device=device,
                    label_name=label_name,
                    tuning_metric=tuning_metric,
                    minority_class=minority_class
                )

                fold_performance = {
                    "model_name": model_name,
                    "fold": fold + 1 if use_k_fold_cv else "Overall_Test", # 标记是哪个折叠或整体测试
                    "accuracy": result["performance_metrics"].get("accuracy", np.nan),
                    "precision_weighted": result["performance_metrics"].get("precision_weighted", np.nan),
                    "recall_weighted": result["performance_metrics"].get("recall_weighted", np.nan),
                    "f1_weighted": result["performance_metrics"].get("f1_weighted", np.nan),
                    "roc_auc": result["performance_metrics"].get("roc_auc", np.nan),
                    "Class_0_precision": result["performance_metrics"].get("Class_0_precision", np.nan),
                    "Class_0_recall": result["performance_metrics"].get("Class_0_recall", np.nan),
                    "Class_0_f1-score": result["performance_metrics"].get("Class_0_f1-score", np.nan),
                    "Class_0_support": result["performance_metrics"].get("Class_0_support", np.nan),
                    "Class_1_precision": result["performance_metrics"].get("Class_1_precision", np.nan),
                    "Class_1_recall": result["performance_metrics"].get("Class_1_recall", np.nan),
                    "Class_1_f1-score": result["performance_metrics"].get("Class_1_f1-score", np.nan),
                    "Class_1_support": result["performance_metrics"].get("Class_1_support", np.nan),
                    "Best Parameters": str(result["best_params"]),
                }
                all_performance_results.append(fold_performance)

                # 填充 OOF 预测。现在 oof_y_pred 是针对所有数据的，在整体模式下只填充测试集部分。
                # 这里的 `test_idx_inner` 已经是 X_filtered/y_filtered 中的整数位置索引
                oof_y_pred[test_idx_inner] = result["y_pred"]
                oof_y_pred_proba[test_idx_inner] = result["y_pred_proba"]

                # 收集特征重要性
                if result["feature_importances_fold"] is not None:
                    if isinstance(X_train_fold, pd.DataFrame):
                        feature_names = X_train_fold.columns
                    else:
                        feature_names = X_filtered.columns # 如果不是DataFrame，使用过滤后的完整列名

                    if len(result["feature_importances_fold"]) == len(feature_names):
                        fold_feature_importances.append(
                            pd.Series(result["feature_importances_fold"], index=feature_names)
                        )
                    else:
                        print(f"警告: 模型 {model_name} 在折叠 {fold + 1 if use_k_fold_cv else '整体'} 的特征重要性数量与特征名称数量不匹配，跳过收集。")

                # --- SHAP 值收集部分 ---
                if config.GENERATE_SHAP_PLOTS and model_name in config.SHAP_SUPPORTED_MODELS:
                    try:
                        explainer_model = result["best_model"]
                        shap_values_current = None
                        expected_value_current = None
                        mean_abs_shap = None
                        feature_names_for_shap = list(X_test_fold.columns)
                        if isinstance(explainer_model, (
                            RandomForestClassifier,
                            xgb.XGBClassifier,
                            lgb.LGBMClassifier
                        )):
                            if hasattr(explainer_model, 'predict_proba'):
                                explainer = shap.TreeExplainer(explainer_model)
                                shap_values_current = explainer.shap_values(X_test_fold)
                                expected_value_current = explainer.expected_value
                                if isinstance(shap_values_current, list) and len(shap_values_current) == 2:
                                    shap_values_current = shap_values_current[1]
                                    if isinstance(expected_value_current, list) and len(expected_value_current) == 2:
                                        expected_value_current = expected_value_current[1]
                        elif model_name == "LogisticRegression":
                            if hasattr(explainer_model, 'coef_'):
                                explainer = shap.LinearExplainer(explainer_model, X_test_fold)
                                shap_values_current = explainer.shap_values(X_test_fold)
                                expected_value_current = explainer.expected_value
                                if isinstance(shap_values_current, list) and len(shap_values_current) == 2:
                                    shap_values_current = shap_values_current[1]
                                    if isinstance(expected_value_current, list) and len(expected_value_current) == 2:
                                        expected_value_current = expected_value_current[1]
                        elif model_name == "LightGBM_FocalLoss":
                            if hasattr(explainer_model, 'predict'):
                                explainer = shap.TreeExplainer(explainer_model)
                                shap_values_current = explainer.shap_values(X_test_fold)
                                expected_value_current = explainer.expected_value
                                if isinstance(shap_values_current, list) and len(shap_values_current) == 2:
                                    shap_values_current = shap_values_current[1]
                                    if isinstance(expected_value_current, list) and len(expected_value_current) == 2:
                                        expected_value_current = expected_value_current[1]
                        if shap_values_current is not None and expected_value_current is not None:
                            all_fold_shap_values.append(shap_values_current)
                            all_fold_X_test_for_shap.append(X_test_fold)
                            fold_expected_values_list.append(expected_value_current)
                            # 新增：将shap均值和特征名写入perf
                            mean_abs_shap = np.abs(shap_values_current).mean(axis=0)
                            fold_performance['mean_abs_shap'] = mean_abs_shap
                            fold_performance['feature_names'] = feature_names_for_shap
                        else:
                            print(f"警告: 模型 {model_name} ({'当前折叠' if use_k_fold_cv else '整体模式'}) SHAP 值或期望值未能成功计算，跳过收集。")

                    except Exception as shap_e:
                        print(f"SHAP 值计算失败 (模型 {model_name}, {'当前折叠' if use_k_fold_cv else '整体模式'}): {shap_e}")
                        import traceback
                        traceback.print_exc()

            except Exception as e:
                print(f"运行 {model_name} 在 {'折叠 ' + str(fold+1) if use_k_fold_cv else '整体模式'} 失败: {e}")
                import traceback
                traceback.print_exc()

            # 在非 K 折模式下，我们只循环一次
            if not use_k_fold_cv:
                break

        # --- 计算并保存当前模型的**整体性能 (OOF 或整体测试集)** ---
        valid_oof_indices = ~np.isnan(oof_y_pred)

        if valid_oof_indices.sum() > 0 and len(np.unique(y_filtered[valid_oof_indices])) == 2:
            print(f"\n--- {model_name} 整体 Out-of-Fold (OOF) 性能 ---" if use_k_fold_cv else f"\n--- {model_name} 整体测试集性能 ---")

            y_true_for_overall_eval = y_filtered[valid_oof_indices]
            y_pred_for_overall_eval = oof_y_pred[valid_oof_indices].astype(int)
            y_pred_proba_for_overall_eval = oof_y_pred_proba[valid_oof_indices]


            oof_metrics = evaluate_classification_metrics(
                y_true_for_overall_eval,
                y_pred_for_overall_eval,
                y_pred_proba_for_overall_eval
            )

            oof_performance_summary = {
                "model_name": model_name,
                "fold": "OOF_Overall" if use_k_fold_cv else "Overall_Test_Set",
                "accuracy": oof_metrics.get("accuracy", np.nan),
                "precision_weighted": oof_metrics.get("precision_weighted", np.nan),
                "recall_weighted": oof_metrics.get("recall_weighted", np.nan),
                "f1_weighted": oof_metrics.get("f1_weighted", np.nan),
                "roc_auc": oof_metrics.get("roc_auc", np.nan),
                "Class_0_precision": oof_metrics.get("Class_0_precision", np.nan),
                "Class_0_recall": oof_metrics.get("Class_0_recall", np.nan),
                "Class_0_f1-score": oof_metrics.get("Class_0_f1-score", np.nan),
                "Class_0_support": oof_metrics.get("Class_0_support", np.nan),
                "Class_1_precision": oof_metrics.get("Class_1_precision", np.nan),
                "Class_1_recall": oof_metrics.get("Class_1_recall", np.nan),
                "Class_1_f1-score": oof_metrics.get("Class_1_f1-score", np.nan),
                "Class_1_support": oof_metrics.get("Class_1_support", np.nan),
            }
            all_performance_results.append(oof_performance_summary)
        else:
            print(f"警告: {model_name} 没有足够的有效 {'OOF' if use_k_fold_cv else '测试集'} 数据或标签类别不符合要求来计算总体性能。")

        current_exp_sub_dir = f"supervised_models/{model_name}"

        # --- 绘制并保存 ROC 和 PR 曲线 (使用 OOF/整体测试集预测) ---
        if valid_oof_indices.sum() > 0 and len(np.unique(y_filtered[valid_oof_indices])) == 2:
            print(f"\n--- 绘制 {model_name} ROC 和 PR 曲线 ({'OOF' if use_k_fold_cv else '整体测试集'}) ---")
            roc_title = f"{model_name} ROC 曲线 ({'OOF' if use_k_fold_cv else '整体测试集'})\n(数据: {clean_file_name.replace('_', '.')}, Sheet: {clean_sheet_name.replace('_', ' ')})"
            pr_title = f"{model_name} Precision-Recall 曲线 ({'OOF' if use_k_fold_cv else '整体测试集'})\n(数据: {clean_file_name.replace('_', '.')}, Sheet: {clean_sheet_name.replace('_', ' ')})"

            roc_path = plot_and_save_figure(
                plot_roc_pr_curve,
                f"{label_name}_{model_name}_{'oof' if use_k_fold_cv else 'overall'}_roc_curve.png",
                experiment_name,
                current_exp_sub_dir,
                y_true=y_true_for_overall_eval,
                y_pred_proba=y_pred_proba_for_overall_eval,
                plot_type='ROC',
                title=roc_title
            )
            pr_path = plot_and_save_figure(
                plot_roc_pr_curve,
                f"{label_name}_{model_name}_{'oof' if use_k_fold_cv else 'overall'}_pr_curve.png",
                experiment_name,
                current_exp_sub_dir,
                y_true=y_true_for_overall_eval,
                y_pred_proba=y_pred_proba_for_overall_eval,
                plot_type='PR',
                title=pr_title
            )
            if roc_path: all_figure_paths.append(str(roc_path))
            if pr_path: all_figure_paths.append(str(pr_path))
        else:
            print(f"警告: 无法为 {model_name} 绘制 ROC/PR 曲线，因为 {'OOF' if use_k_fold_cv else '整体测试集'} 预测概率或真实标签不符合要求。")

        # --- 绘制并保存混淆矩阵 ---
        if valid_oof_indices.sum() > 0 and len(np.unique(y_filtered[valid_oof_indices])) == 2:
            print(f"\n--- 绘制 {model_name} 混淆矩阵 ({'OOF' if use_k_fold_cv else '整体测试集'}) ---")
            cm_title = f"{model_name} 混淆矩阵 ({'OOF' if use_k_fold_cv else '整体测试集'})\n(数据: {clean_file_name.replace('_', '.')}, Sheet: {clean_sheet_name.replace('_', ' ')})"

            cm_path = plot_and_save_figure(
                plot_confusion_matrix,
                f"{label_name}_{model_name}_{'oof' if use_k_fold_cv else 'overall'}_confusion_matrix.png",
                experiment_name,
                current_exp_sub_dir,
                y_true=y_true_for_overall_eval,
                y_pred=y_pred_for_overall_eval.astype(int),
                labels=[0, 1],
                title=cm_title
            )
            if cm_path: all_figure_paths.append(str(cm_path))
        else:
            print(f"警告: 无法为 {model_name} 绘制混淆矩阵，因为 {'OOF' if use_k_fold_cv else '整体测试集'} 预测或真实标签不符合要求。")

        # --- 绘制并保存特征重要性图 ---
        if len(fold_feature_importances) > 0:
            print(f"\n--- 绘制 {model_name} 平均特征重要性图 ---")
            try:
                if len(fold_feature_importances) == 1:
                    avg_feature_importances = fold_feature_importances[0]
                else:
                    avg_feature_importances = pd.concat(fold_feature_importances, axis=1).mean(axis=1)

                fi_path = plot_and_save_figure(
                    plot_feature_importance,
                    f"{label_name}_{model_name}_feature_importance.png",
                    experiment_name,
                    current_exp_sub_dir,
                    feature_importances=avg_feature_importances.values,
                    feature_names=avg_feature_importances.index.tolist(),
                    title=f"{model_name} 平均特征重要性 ({'K-Fold' if use_k_fold_cv else '整体训练'})\n(数据: {clean_file_name.replace('_', '.')}, Sheet: {clean_sheet_name.replace('_', ' ')})"
                )
                if fi_path: all_figure_paths.append(str(fi_path))
            except Exception as e:
                print(f"绘制 {model_name} 平均特征重要性失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"警告: 模型 {model_name} 未能收集特征重要性，无法生成特征重要性图。")


        # --- 生成 SHAP 汇总图 (如果收集到了 SHAP 值) ---
        if config.GENERATE_SHAP_PLOTS and model_name in config.SHAP_SUPPORTED_MODELS and len(all_fold_shap_values) > 0:
            print(f"\n--- 生成 {model_name} 的 SHAP 汇总图 ---")

            try:
                combined_shap_values = np.vstack(all_fold_shap_values)
                combined_X_for_shap = pd.concat(all_fold_X_test_for_shap, axis=0)
                avg_expected_value = np.mean(fold_expected_values_list) if len(fold_expected_values_list) > 0 else None

                if combined_X_for_shap.shape[0] > config.SHAP_MAX_DISPLAY:
                    sample_indices = np.random.choice(combined_X_for_shap.shape[0],
                                                      min(config.SHAP_MAX_DISPLAY, combined_X_for_shap.shape[0]),
                                                      replace=False)
                    X_shap_for_plotting = combined_X_for_shap.iloc[sample_indices]
                    shap_values_for_plotting = combined_shap_values[sample_indices]
                else:
                    X_shap_for_plotting = combined_X_for_shap
                    shap_values_for_plotting = combined_shap_values

                # SHAP bar图（排序前20）
                shap_bar_path = plot_and_save_figure(
                    plot_shap_summary,
                    f"{label_name}_{model_name}_shap_bar_summary_{'OOF' if use_k_fold_cv else 'Overall'}.png",
                    experiment_name,
                    f"{current_exp_sub_dir}/shap_plots",
                    shap_values=shap_values_for_plotting,
                    expected_value=avg_expected_value,
                    X=X_shap_for_plotting,
                    feature_names=X_filtered.columns.tolist(),
                    plot_type="bar",
                    title=f"{model_name} SHAP 平均特征重要性 (Top 20, {'OOF' if use_k_fold_cv else '整体'})\n(数据: {clean_file_name.replace('_', '.')}, Sheet: {clean_sheet_name.replace('_', ' ')})",
                    top_n=20,
                    sort=True
                )
                if shap_bar_path: all_figure_paths.append(str(shap_bar_path))

                # SHAP bar图（所有特征不排序）
                shap_bar_all_path = plot_and_save_figure(
                    plot_shap_summary,
                    f"{label_name}_{model_name}_shap_bar_summary_all_unordered_{'OOF' if use_k_fold_cv else 'Overall'}.png",
                    experiment_name,
                    f"{current_exp_sub_dir}/shap_plots",
                    shap_values=shap_values_for_plotting,
                    expected_value=avg_expected_value,
                    X=X_shap_for_plotting,
                    feature_names=X_filtered.columns.tolist(),
                    plot_type="bar",
                    title=f"{model_name} SHAP 所有特征重要性 (原始顺序)\n(数据: {clean_file_name.replace('_', '.')}, Sheet: {clean_sheet_name.replace('_', ' ')})",
                    top_n=None,
                    sort=False
                )
                if shap_bar_all_path: all_figure_paths.append(str(shap_bar_all_path))

                # SHAP dot图（原有）
                shap_dot_path = plot_and_save_figure(
                    plot_shap_summary,
                    f"{label_name}_{model_name}_shap_dot_summary_{'OOF' if use_k_fold_cv else 'Overall'}.png",
                    experiment_name,
                    f"{current_exp_sub_dir}/shap_plots",
                    shap_values=shap_values_for_plotting,
                    expected_value=avg_expected_value,
                    X=X_shap_for_plotting,
                    feature_names=X_filtered.columns.tolist(),
                    plot_type="dot",
                    title=f"{model_name} SHAP 影响 ({'OOF' if use_k_fold_cv else '整体'})\n(数据: {clean_file_name.replace('_', '.')}, Sheet: {clean_sheet_name.replace('_', ' ')})"
                )
                if shap_dot_path: all_figure_paths.append(str(shap_dot_path))

            except Exception as e:
                print(f"生成 {model_name} 的 SHAP 汇总图失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"警告: 模型 {model_name} 的 SHAP 值未成功收集或未配置生成 SHAP 图，无法生成 SHAP 汇总图。")

    # === 新增：保存性能+特征名+shap+重要性到Excel（分区块） ===
    excel_save_path = config.REPORTS_OUTPUT_DIR / f"model_performance_with_shap.xlsx"
    # 1. 性能指标表格
    perf_df = pd.DataFrame(all_performance_results)
    # 2. 详细特征报告
    detail_rows = []
    for perf in all_performance_results:
        label_fold = perf.get('label', perf.get('model_name','') + '_' + str(perf.get('fold','')))
        feature_names = perf.get('feature_names', None)
        shap_row = perf.get('mean_abs_shap', None)
        importance_row = perf.get('feature_importances_fold', None)
        # 兼容Series/ndarray
        if hasattr(importance_row, 'values'):
            importance_row = list(importance_row.values)
        if feature_names is not None and shap_row is not None and importance_row is not None:
            detail_rows.append([label_fold] + [''] * (len(feature_names)-1))
            detail_rows.append(list(feature_names))
            detail_rows.append(['SHAP'] + [f'{v:.6f}' for v in shap_row])
            detail_rows.append(['特征重要性'] + [f'{v:.6f}' for v in importance_row])
            detail_rows.append([''] * len(feature_names))
    # 3. 合并写入Excel
    with pd.ExcelWriter(excel_save_path, engine='openpyxl') as writer:
        perf_df.to_excel(writer, index=False, header=True, sheet_name='性能与特征报告')
        # 追加详细特征报告（无表头）
        startrow = len(perf_df) + 2
        for i, row in enumerate(detail_rows):
            pd.DataFrame([row]).to_excel(writer, index=False, header=False, startrow=startrow + i, sheet_name='性能与特征报告')
    print(f'性能+特征重要性+SHAP已保存到: {excel_save_path}')

    return all_performance_results, all_figure_paths