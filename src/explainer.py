# src/explainer.py

import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns

from src import config # 导入 config 模块

def generate_shap_plots(
    model,
    X_data: pd.DataFrame,
    model_name: str,
    experiment_name: str,
    fold: str = "Overall", # 可以是折叠号，也可以是 "Overall"
    class_names: list = ['Negative', 'Positive'],
    shap_values=None, # 如果已经计算好 SHAP 值，可以直接传入
    expected_value=None # 如果已经计算好 expected value，可以直接传入
):
    """
    生成并保存 SHAP 解释图(Summary Plot 和 Dependence Plots)。

    Args:
        model: 已训练的模型实例 (如果 shap_values 和 expected_value 未提供，则需要此参数)。
        X_data (pd.DataFrame): 用于生成 SHAP 值的特征数据(例如，测试集)。
        model_name (str): 模型名称。
        experiment_name (str): 当前实验名称。
        fold (str): 当前折叠的标识(例如 "fold_1", "Overall_OOF")。
        class_names (list): 类别名称列表，默认为 ['Negative', 'Positive']。
        shap_values: 预计算的 SHAP 值。
        expected_value: 预计算的 SHAP 期望值。

    Returns:
        list: 生成的 SHAP 图文件路径列表。
    """
    print(f"--- 正在为 {model_name} (Fold: {fold}) 生成 SHAP 图 ---")
    figure_paths = []

    # 确保 SHAP_OUTPUT_DIR 存在
    shap_output_dir = config.VISUALIZATIONS_OUTPUT_DIR / "shap_plots"
    shap_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 如果没有预计算的 SHAP 值，则从模型和数据计算
        if shap_values is None or expected_value is None:
            if model is None:
                print("警告: 缺少模型或预计算的 SHAP 值，无法生成 SHAP 图。")
                return []
            
            # 针对不同类型的模型使用不同的 Explainer
            if hasattr(model, 'tree_ensemble') or (hasattr(model, 'booster_') and 'lightgbm' in str(type(model))):
                # 适用于 LightGBM, XGBoost, RandomForest 等基于树的模型
                explainer = shap.TreeExplainer(model)
            elif hasattr(model, 'predict_proba'):
                # 适用于具有 predict_proba 方法的线性模型、SVM 等
                explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_data, 100)) # KernelExplainer 较慢，需要采样
            else:
                print(f"警告: SHAP 解释器不支持模型类型: {type(model)}。跳过 SHAP 图生成。")
                return []

            # 计算 SHAP 值
            # 对于二分类模型，shap_values 可能是两个数组的列表，分别对应 Class 0 和 Class 1
            # 我们通常关心正类 (Class 1) 的解释
            if hasattr(model, 'predict_proba') and len(class_names) == 2:
                # 获取 Class 1 的 SHAP 值和期望值
                shap_values_computed = explainer.shap_values(X_data)
                if isinstance(shap_values_computed, list) and len(shap_values_computed) > 1:
                    shap_values = shap_values_computed[1] # 取正类的 SHAP 值
                else: # 如果不是列表或只有一个元素，直接使用
                    shap_values = shap_values_computed
                
                expected_value_computed = explainer.expected_value
                if isinstance(expected_value_computed, list) and len(expected_value_computed) > 1:
                    expected_value = expected_value_computed[1] # 取正类的期望值
                else:
                    expected_value = expected_value_computed
            else: # 非分类模型或非二分类
                shap_values = explainer.shap_values(X_data)
                expected_value = explainer.expected_value

        # --- SHAP Summary Plot ---
        if shap_values is not None and len(X_data.columns) > 0:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_data, show=False, plot_type="dot", class_names=class_names)
            summary_plot_path = shap_output_dir / f"shap_summary_plot_{model_name}_{experiment_name}_{fold}.png"
            plt.tight_layout()
            plt.savefig(summary_plot_path, dpi=300)
            plt.close()
            figure_paths.append(str(summary_plot_path))
            print(f"SHAP Summary Plot 已保存到: {summary_plot_path}")

            # --- SHAP Bar Plot (Feature Importance) ---
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_data, show=False, plot_type="bar", class_names=class_names)
            bar_plot_path = shap_output_dir / f"shap_bar_plot_{model_name}_{experiment_name}_{fold}.png"
            plt.tight_layout()
            plt.savefig(bar_plot_path, dpi=300)
            plt.close()
            figure_paths.append(str(bar_plot_path))
            print(f"SHAP Bar Plot (Feature Importance) 已保存到: {bar_plot_path}")
            
            # --- 生成几个最重要的特征的 Dependence Plot ---
            # 获取特征重要性排名
            feature_importance_df = pd.DataFrame({
                'feature': X_data.columns,
                'importance': np.abs(shap_values).mean(axis=0)
            }).sort_values(by='importance', ascending=False)

            top_n_features = min(5, len(feature_importance_df)) # 生成前5个或所有特征的依赖图
            for i in range(top_n_features):
                feature = feature_importance_df.iloc[i]['feature']
                plt.figure(figsize=(8, 6))
                # plot_type="scatter" 是默认的，展示与另一个特征的交互
                # 如果只看单个特征的影响，可以不指定 interaction_index
                shap.dependence_plot(
                    feature,
                    shap_values,
                    X_data,
                    interaction_index="auto", # 自动选择最相关的交互特征
                    show=False,
                    x_jitter=0.5 # 添加少量随机抖动避免点重叠
                )
                dependence_plot_path = shap_output_dir / f"shap_dependence_plot_{model_name}_{experiment_name}_{fold}_{feature}.png"
                plt.tight_layout()
                plt.savefig(dependence_plot_path, dpi=300)
                plt.close()
                figure_paths.append(str(dependence_plot_path))
                print(f"SHAP Dependence Plot for {feature} 已保存到: {dependence_plot_path}")

        else:
            print(f"SHAP 值为空或 X_data 无特征，无法生成 SHAP 图。")

    except Exception as e:
        print(f"生成 SHAP 图时发生错误 for {model_name} (Fold: {fold}): {e}")

    return figure_paths