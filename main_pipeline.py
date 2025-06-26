# main_pipeline.py

from typing import Union 
import pandas as pd
import numpy as np
import os
import warnings
import torch # 确保这里导入了 torch
import json
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

from src import config
from src.data_loader import (
    load_npy_data, save_performance_data, load_processed_data,
    save_intermediate_data, load_intermediate_data
)
from src.pipelines.supervised_pipeline import run_supervised_pipeline # 确保导入了 run_supervised_pipeline
from src.unsupervised_models import run_unsupervised_analysis
from src.visualizer import (
    generate_powerpoint_report,
    set_chinese_font,
    plot_cluster_scatter, plot_elbow_method,
    display_performance_metrics 
)

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.*")
warnings.filterwarnings("ignore", category=UserWarning, module="shap")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="xgboost")

# GPU配置，根据个人情况调整
# --- 加载配置文件 ---
GPU_config_path = os.path.join(os.path.dirname(__file__), 'GPU_config.json')
try:
    with open(GPU_config_path, 'r', encoding='utf-8') as f:
        GPU_config = json.load(f)
    print(f"配置文件 {GPU_config_path} 加载成功！")
except FileNotFoundError:
    print(f"错误：未找到配置文件 {GPU_config_path}。请确保文件存在且路径正确。")
    exit(1)
except json.JSONDecodeError as e:
    print(f"错误：加载配置文件时发生 JSON 解析错误：{e}")
    exit(1)

# --- 从配置中获取参数 ---
use_gpu = GPU_config['training']['use_gpu']
gpu_id = GPU_config['training']['gpu_id']
if use_gpu and torch.cuda.is_available():
    device = torch.device(f"cuda:{gpu_id}")
    print(f"PyTorch 将使用 GPU: {torch.cuda.get_device_name(gpu_id)}")
else:
    device = torch.device("cpu")
    print("PyTorch 将使用 CPU")


def main_analysis_pipeline(
    experiment_name="default_analysis",
    processed_data_file_name: str = "data.xlsx",
    sheet_name: Union[str, int] = 0,
    target_column_supervised=config.DEFAULT_TARGET_COLUMN,
    supervised_models_to_run=None,
    unsupervised_models_to_run=None,
    skip_data_prep: bool = False,
    run_supervised: bool = True,
    run_unsupervised: bool = True,
    run_reporting: bool = True,
    use_k_fold_cv: bool = False,
    n_splits_supervised: int = 5,
    test_size_overall: float = 0.2,
    random_state: int = 42,
    supervised_tuning_metric: str = None
):
    """
    主分析流程函数。
    负责数据加载、模型训练、评估和报告生成。
    所有实验参数(K折、测试集比例、模型列表等(都在 main.py 里灵活配置。

    Args:
        experiment_name (str): 当前实验的名称,用于文件保存。
        processed_data_file_name (str): 包含有监督学习数据(和可选无监督数据(的Excel文件名。
        sheet_name (str or int): Excel文件中要读取的工作表名称或索引。
        target_column_supervised (str): 有监督学习的目标变量(标签(的列名。
        supervised_models_to_run (list or None): 要运行的有监督模型名称列表。
        unsupervised_models_to_run (list or None): 要运行的无监督模型名称列表。
        skip_data_prep (bool): 如果为True,则尝试从中间文件加载处理好的数据,跳过数据准备步骤。
        run_supervised (bool): 如果为True,则运行有监督学习流程。
        run_unsupervised (bool): 如果为True,则运行无监督学习流程。
        run_reporting (bool): 如果为True,则生成最终的PowerPoint报告。
        use_k_fold_cv (bool): 是否使用K折交叉验证(默认False,即整体训练/测试集划分(。
        n_splits_supervised (int): K折交叉验证的折数。
        test_size_overall (float): 整体训练/测试划分的测试集比例。
        random_state (int): 随机种子。
        supervised_tuning_metric (str or None): 有监督学习调优的指标，默认None。
    """
    set_chinese_font()

    print(f"\n--- 开始实验: {experiment_name} ---")
    print(f"--- 数据源: {processed_data_file_name}, 工作表: {sheet_name} ---")
    print(f"--- 训练模式: {'K折交叉验证' if use_k_fold_cv else '整体训练/测试集划分'} ---")
    if use_k_fold_cv:
        print(f"--- K折设置: {n_splits_supervised} 折 ---")
    else:
        print(f"--- 测试集比例: {test_size_overall:.1%} ---")

    all_performance_results = []
    all_figure_paths = []

    X_supervised_df = None
    y_supervised_processed = None
    feature_names_supervised = []
    raw_unsupervised_data_tensor = None
    patient_ids_unsupervised = None

    # 定义本次运行的中间文件路径,确保唯一性
    clean_file_name = processed_data_file_name.replace('.', '_').replace(os.sep, '_')
    clean_sheet_name = str(sheet_name).replace(' ', '_').replace('/', '_')

    processed_x_cache_path = config.INTERIM_DATA_DIR / f"X_supervised_{clean_file_name}_{clean_sheet_name}.parquet"
    processed_y_cache_path = config.INTERIM_DATA_DIR / f"y_supervised_{clean_file_name}_{clean_sheet_name}.pkl"
    processed_feature_names_cache_path = config.INTERIM_DATA_DIR / f"feature_names_supervised_{clean_file_name}_{clean_sheet_name}.pkl"

    raw_unsupervised_data_tensor_cache_path = config.INTERIM_DATA_DIR / f"raw_unsupervised_data_tensor_{clean_file_name}_{clean_sheet_name}.pt"
    patient_ids_unsupervised_cache_path = config.INTERIM_DATA_DIR / f"patient_ids_unsupervised_{clean_file_name}_{clean_sheet_name}.pkl"


    # --- 1. 数据加载与预处理 (可跳过) ---
    print(f"\n--- 1. 数据加载与预处理 ---")
    if skip_data_prep:
        print(f"尝试从中间文件加载数据 (来自 {processed_data_file_name}, Sheet: {sheet_name})...")
        X_supervised_df = load_intermediate_data(processed_x_cache_path)
        y_supervised_processed = load_intermediate_data(processed_y_cache_path)
        feature_names_supervised = load_intermediate_data(processed_feature_names_cache_path)
        raw_unsupervised_data_tensor = load_intermediate_data(raw_unsupervised_data_tensor_cache_path)
        patient_ids_unsupervised = load_intermediate_data(patient_ids_unsupervised_cache_path)

        if X_supervised_df is not None and y_supervised_processed is not None and \
           feature_names_supervised is not None and \
           raw_unsupervised_data_tensor is not None and patient_ids_unsupervised is not None:
            print("所有数据已成功从中间文件加载。")
            print(f"加载数据: X_supervised_df 形状: {X_supervised_df.shape}, y_supervised_processed 形状: {y_supervised_processed.shape}")
            print(f"加载数据: raw_unsupervised_data_tensor 形状: {raw_unsupervised_data_tensor.shape if raw_unsupervised_data_tensor is not None else 'None'}, patient_ids_unsupervised 形状: {patient_ids_unsupervised.shape if patient_ids_unsupervised is not None else 'None'}")
        else:
            print("中间数据加载不完整或失败,将重新执行数据准备。")
            X_supervised_df = None
            y_supervised_processed = None
            feature_names_supervised = []
            raw_unsupervised_data_tensor = None
            patient_ids_unsupervised = None
            skip_data_prep = False

    if not skip_data_prep:
        config.INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # --- 1.1 加载预处理后的数据 (用于有监督学习) ---
        print(f"--- 1.1 加载预处理后的数据 (有监督) ---")
        processed_data_full_path = config.DATA_PROCESSED_DIR / processed_data_file_name
        processed_df = load_processed_data(file_path=processed_data_full_path, sheet_name=sheet_name)
        
        if processed_df is not None:
            label_cols = [col for col in processed_df.columns if col.startswith('Label')]
            if not label_cols:
                raise ValueError(f"错误: 没有检测到任何以'Label'开头的标签列，请检查数据文件！")
            print(f"检测到标签列: {label_cols}")
            # 不提前拆分X/y，留到多标签循环内
        else:
            print("由于有监督数据加载失败,跳过有监督学习数据准备。")

        # --- 1.2 加载无监督学习所需的原始数据 (.npy 文件) ---
        print("\n--- 1.2 加载无监督学习原始数据 ---")
        try:
            raw_unsupervised_data = load_npy_data(config.DATA_EXTERNAL_DIR / "split_data.npy")
            if raw_unsupervised_data is not None:
                # 将无监督学习的原始数据移动到 GPU (如果可用)
                raw_unsupervised_data_tensor = torch.tensor(raw_unsupervised_data, dtype=torch.float32).to(device) # <--- GPU 调用
                patient_ids_unsupervised = load_npy_data(config.DATA_EXTERNAL_DIR / "patient_ids.npy")
                print(f"无监督学习原始数据 (split_data) 形状: {raw_unsupervised_data_tensor.shape}")
                if patient_ids_unsupervised is not None:
                    print(f"病人ID (patient_ids) 形状: {patient_ids_unsupervised.shape}")
                else:
                    print("病人ID (patient_ids) 加载失败")
            else:
                print("未加载到无监督学习原始数据,跳过无监督数据准备。")
                raw_unsupervised_data_tensor = None
                patient_ids_unsupervised = None
        except Exception as e:
            print(f"加载无监督学习原始数据时发生错误: {e}")
            raw_unsupervised_data_tensor = None
            patient_ids_unsupervised = None

        # >>> 保存处理好的数据到中间文件 <<<
        if processed_df is not None:
            save_intermediate_data(processed_df, processed_x_cache_path)
            save_intermediate_data(y_supervised_processed, processed_y_cache_path)
            save_intermediate_data(feature_names_supervised, processed_feature_names_cache_path)
        if raw_unsupervised_data_tensor is not None:
            # 如果 raw_unsupervised_data_tensor 在 GPU 上，保存前需要移回 CPU
            if raw_unsupervised_data_tensor.is_cuda:
                save_intermediate_data(raw_unsupervised_data_tensor.cpu(), raw_unsupervised_data_tensor_cache_path)
            else:
                save_intermediate_data(raw_unsupervised_data_tensor, raw_unsupervised_data_tensor_cache_path)
            save_intermediate_data(patient_ids_unsupervised, patient_ids_unsupervised_cache_path)


    # --- 2. 运行有监督学习流程 ---
    if run_supervised and supervised_models_to_run and processed_df is not None:
        print("\n--- 2. 运行有监督学习流程 ---")
        models_to_execute_supervised = supervised_models_to_run if supervised_models_to_run else list(config.SUPERVISED_MODELS.keys())
        label_cols = [col for col in processed_df.columns if col.startswith('Label')]
        if not label_cols:
            print("未检测到任何以'Label'开头的标签列，跳过有监督训练。")
        else:
            for label in label_cols:
                print(f"\n=== 当前标签: {label} ===")
                y_label = processed_df[label]
                X_label = processed_df.drop(columns=label_cols)  # 特征去除所有Label列
                try:
                    supervised_results, supervised_figures = run_supervised_pipeline(
                        X_supervised_df=X_label,
                        y_supervised_processed=y_label,
                        supervised_models_to_run=models_to_execute_supervised,
                        experiment_name=experiment_name,
                        clean_file_name=clean_file_name,
                        clean_sheet_name=clean_sheet_name,
                        use_k_fold_cv=use_k_fold_cv,
                        n_splits_supervised=n_splits_supervised,
                        test_size_overall=test_size_overall,
                        random_state=random_state,
                        device=device,
                        label_name=label,  # 传递标签名
                        supervised_tuning_metric=supervised_tuning_metric
                    )
                    for res in supervised_results:
                        res['label'] = label 
                        
                    all_performance_results.extend(supervised_results)
                    for fig_path in supervised_figures:
                        fig_title = os.path.basename(fig_path).replace('.png', '').replace('_', ' ').title()
                        all_figure_paths.append((fig_path, fig_title))
                    for res in supervised_results:
                        if res.get("fold") == ("OOF_Overall" if use_k_fold_cv else "Overall_Test_Set"):
                            print(f"\n--- {res['model_name']}({label}) 整体性能 ---")
                            display_performance_metrics(res['model_name'], res)
                except Exception as e:
                    print(f"运行有监督学习管道失败 (标签: {label}): {e}")
                    import traceback
                    traceback.print_exc()
    else:
        print("\n--- 跳过有监督学习流程 或 缺少必要数据 ---")


    # --- 3. 运行无监督学习流程 ---
    if run_unsupervised and unsupervised_models_to_run and raw_unsupervised_data_tensor is not None and patient_ids_unsupervised is not None:
        print("\n--- 3. 运行无监督学习流程 ---")
        models_to_execute_unsupervised = unsupervised_models_to_run if unsupervised_models_to_run else list(config.UNSUPERVISED_MODELS.keys())
        
        for model_name in models_to_execute_unsupervised:
            if model_name not in config.UNSUPERVISED_MODELS:
                print(f"警告: 配置中未找到无监督模型 '{model_name}',跳过。")
                continue

            try:
                print(f"\n--- 运行无监督模型: {model_name} ---")
                unsupervised_results = run_unsupervised_analysis(
                    raw_unsupervised_data_tensor.clone().detach(), # 传递张量的副本
                    patient_ids_unsupervised.copy(),
                    model_name,
                    device=device # <--- 在这里将 device 传递给 run_unsupervised_analysis
                )
                
                if unsupervised_results:
                    performance_entry = {
                        "experiment_name": experiment_name,
                        "analysis_type": "unsupervised",
                        "model_name": model_name,
                        "processed_data_file": processed_data_file_name,
                        "sheet_name": sheet_name,
                        **unsupervised_results.get("evaluation_metrics", {})
                    }
                    all_performance_results.append(performance_entry)

                    print(f"无监督模型 {model_name} 评估指标: {unsupervised_results.get('evaluation_metrics', {})}")
                    display_performance_metrics(model_name, unsupervised_results.get('evaluation_metrics', {}))

                    # 保存图表
                    current_exp_sub_dir = f"unsupervised_models_{experiment_name}_{clean_file_name}_{clean_sheet_name}"


                    if 'cluster_df' in unsupervised_results:
                        fig_scatter = plot_cluster_scatter(
                            unsupervised_results['cluster_df'], 
                            x_col='Component_0', 
                            y_col='Component_1', 
                            cluster_col='Cluster', 
                            title=f"{model_name} 聚类散点图\n(数据: {processed_data_file_name}, Sheet: {sheet_name})",
                            legend_title="聚类"
                        )
                        scatter_path = config.VISUALIZATIONS_OUTPUT_DIR / current_exp_sub_dir / f"{model_name}_cluster_scatter.png"
                        scatter_path.parent.mkdir(parents=True, exist_ok=True)
                        fig_scatter.savefig(scatter_path, dpi=300)
                        plt.close(fig_scatter)
                        all_figure_paths.append((str(scatter_path), f"{model_name} 聚类散点图"))


                    if 'wcss_values' in unsupervised_results and config.UNSUPERVISED_MODELS[model_name].get("clusterer", {}).get("find_optimal_k"):
                        fig_elbow = plot_elbow_method(
                            unsupervised_results['wcss_values'], 
                            config.UNSUPERVISED_MODELS[model_name]["clusterer"]["max_k_for_elbow"],
                            title=f"{model_name} 肘部法则\n(数据: {processed_data_file_name}, Sheet: {clean_sheet_name})"
                        )
                        elbow_path = config.VISUALIZATIONS_OUTPUT_DIR / current_exp_sub_dir / f"{model_name}_elbow_method.png"
                        elbow_path.parent.mkdir(parents=True, exist_ok=True)
                        fig_elbow.savefig(elbow_path, dpi=300)
                        plt.close(fig_elbow)
                        all_figure_paths.append((str(elbow_path), f"{model_name} 肘部法则"))
                else:
                    print(f"无监督模型 {model_name} 未返回结果。")

            except Exception as e:
                print(f"运行无监督模型 {model_name} 失败: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("\n--- 跳过无监督学习流程 或 缺少必要数据 ---")


    # --- 4. 生成报告 ---
    if run_reporting:
        print("\n--- 4. 生成报告 ---")
        if all_performance_results:
            performance_df = pd.DataFrame(all_performance_results)
            report_filename = f"{clean_sheet_name}.csv"
            current_report_exp_dir = config.REPORTS_OUTPUT_DIR / experiment_name
            current_report_exp_dir.mkdir(parents=True, exist_ok=True)

            save_performance_data(performance_df, current_report_exp_dir / report_filename)
        else:
            print("没有性能结果可供保存。")

        if all_figure_paths:
            ppt_filename = f"{clean_sheet_name}.pptx"
            figure_paths_with_titles = [(Path(fig_path), fig_title) for fig_path, fig_title in all_figure_paths]
            generate_powerpoint_report(figure_paths_with_titles, ppt_filename, experiment_name)
        else:
            print("没有图表可供生成报告。")
    else:
        print("\n--- 跳过报告生成 ---")

    print(f"\n--- 实验 {experiment_name} 完成 ---")