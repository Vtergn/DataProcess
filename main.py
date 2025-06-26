# main.py
from main_pipeline import main_analysis_pipeline


if __name__ == "__main__":
    # --- 示例用法 ---

    # 示例 1: 针对 '产妇抑郁.xlsx' 的 '8频段' 运行 RandomForest,整体训练模式(默认(
    # main_analysis_pipeline(
    #     experiment_name="产妇抑郁_Example",
    #     processed_data_file_name="产妇抑郁.xlsx",
    #     sheet_name="8频段",
    #     target_column_supervised="Label",
    #     supervised_models_to_run=["RandomForestClassifier"],
    #     unsupervised_models_to_run=None,
    #     skip_data_prep=True, # 第一次运行请设为 False,生成缓存
    #     run_supervised=True,
    #     run_unsupervised=False,
    #     run_reporting=True,
    #     use_k_fold_cv=False,  # 默认整体训练模式
    #     test_size_overall=0.2,  # 20% 测试集
    #     random_state=42
    # )

    # 示例 2: K折交叉验证模式
    # main_analysis_pipeline(
    #     experiment_name="产妇抑郁_KFold_Example",
    #     processed_data_file_name="产妇抑郁.xlsx",
    #     sheet_name="8频段",
    #     target_column_supervised="Label",
    #     supervised_models_to_run=["RandomForestClassifier"],
    #     unsupervised_models_to_run=None,
    #     skip_data_prep=True,
    #     run_supervised=True,
    #     run_unsupervised=False,
    #     run_reporting=True,
    #     use_k_fold_cv=True,  # 使用K折交叉验证
    #     n_splits_supervised=5,  # 5折
    #     random_state=42
    # )

    # 示例 3: 针对 '产妇抑郁.xlsx' 的 '8频段' 运行 LightGBM_FocalLoss
    main_analysis_pipeline(
        experiment_name="产妇抑郁_LGBMFocalLoss_Example",
        processed_data_file_name="产妇抑郁.xlsx",
        sheet_name="32频段(0.4Hz)",
        target_column_supervised="Label",
        supervised_models_to_run=["LightGBM_FocalLoss"],
        unsupervised_models_to_run=None,
        skip_data_prep=False, # 添加数据缓存，第一次运行请设为 False
        run_supervised=True,
        run_unsupervised=False,
        run_reporting=True,
        supervised_tuning_metric = "f1_score"   #对少数类调优，可选 f1_score/recall/precision/accuracy/roc_auc
    )
    
    # # 示例 4: 运行所有有监督模型
    # main_analysis_pipeline(
    #     experiment_name="产妇抑郁_All_Supervised_Models",
    #     processed_data_file_name="产妇抑郁.xlsx",
    #     sheet_name="8频段",
    #     target_column_supervised="Label",
    #     supervised_models_to_run=list(config.SUPERVISED_MODELS.keys()), # 运行所有有监督模型
    #     unsupervised_models_to_run=None,
    #     skip_data_prep=False,
    #     run_supervised=True,
    #     run_unsupervised=False,
    #     run_reporting=True
    # )

    # # 示例 5: 运行无监督模型 (假设缓存已生成)
    # main_analysis_pipeline(
    #     experiment_name="产妇抑郁_Unsupervised_AE_UMAP_KMeans",
    #     processed_data_file_name="产妇抑郁.xlsx",
    #     sheet_name="8频段",
    #     target_column_supervised="Label", # 即使不运行有监督,也需要提供标签列以便数据加载
    #     supervised_models_to_run=None,
    #     unsupervised_models_to_run=["Autoencoder_UMAP_KMeans"],
    #     skip_data_prep=True, # 如果之前运行过,可以设置为 True
    #     run_supervised=False,
    #     run_unsupervised=True,
    #     run_reporting=True
    # )
    
    # # 示例 6: 仅生成报告 (假设模型运行和图表已生成)
    # # 这对于调试报告格式或只更新PPT很有用
    # main_analysis_pipeline(
    #     experiment_name="产妇抑郁_RF_Example", # 确保实验名称与之前生成图表的实验名称一致
    #     processed_data_file_name="产妇抑郁.xlsx",
    #     sheet_name="8频段",
    #     target_column_supervised="Label",
    #     supervised_models_to_run=None,
    #     unsupervised_models_to_run=None,
    #     skip_data_prep=True, # 需要有数据来确定缓存路径,但实际不加载
    #     run_supervised=False, # 跳过模型运行
    #     run_unsupervised=False, # 跳过模型运行
    #     run_reporting=True # 仅生成报告
    # )