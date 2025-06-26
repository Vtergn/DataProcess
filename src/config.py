# src/config.py

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier # 仍保留，作为类型提示或备用
import xgboost as xgb

# --- 项目路径配置 ---
BASE_DIR = Path(__file__).resolve().parent.parent # 指向项目根目录 (LabDataProcess)

DATA_DIR = BASE_DIR / "Data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_EXTERNAL_DIR = DATA_DIR / "external" # 用于存放 split_data.npy 和 patient_ids.np
REPORTS_DIR = BASE_DIR / "reports" / "产妇抑郁" / "analysis_output"
REPORTS_OUTPUT_DIR = REPORTS_DIR
MODELS_OUTPUT_DIR = BASE_DIR / "models" # 用于保存训练好的模型

# >>> 新增：可视化输出目录 (用于保存所有图表，包括 SHAP 图) <<<
VISUALIZATIONS_OUTPUT_DIR = BASE_DIR / "visualizations"

# >>> 新增：中间数据保存目录 (保持不变) <<<
INTERIM_DATA_DIR = DATA_DIR / "interim"

# 确保输出目录存在
REPORTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATIONS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # 确保可视化输出目录存在
INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True) # 确保中间数据目录也存在

# --- 文件路径 ---
# PROCESSED_DATA_INPUT_PATH = DATA_PROCESSED_DIR / "data.xlsx" # <<< 移除此行，现在由main.py动态传入
MODEL_PERFORMANCE_OUTPUT_FILE = REPORTS_OUTPUT_DIR / "model_performance_summary.csv"

# --- 全局参数 ---
RANDOM_STATE = 42
N_SPLITS_SUPERVISED = 5 # 有监督学习交叉验证的折数
N_SPLITS_GRIDSEARCH = 3 # GridSearchCV 内部交叉验证的折数 (supervised_models.py 中使用)
N_SPLITS_UNSUPERVISED = 5 # 无监督学习交叉验证的折数 (如果需要)
DEFAULT_TARGET_COLUMN = "Label" # 默认的目标变量列名 (例如 "SDS_group" 或 "BDI_group")
SAVE_BEST_MODELS = True # 是否保存训练好的最佳模型

# --- SHAP 可解释性配置 ---
GENERATE_SHAP_PLOTS = True # 是否生成 SHAP 解释图
# 定义支持 SHAP 可视化的模型列表。这些模型通常是基于树的模型，或者可以通过 KernelExplainer 处理的
# 如果 LightGBM_FocalLoss 的 'best_model' 是 lgb.Booster 类型，它通常也支持 TreeExplainer
SHAP_SUPPORTED_MODELS = [
    "RandomForestClassifier",
    "XGBClassifier",
    "LightGBM_FocalLoss", # 虽然 LightGBM_FocalLoss 是自定义的，但底层的 LightGBM 模型支持 SHAP
    "LogisticRegression", # LogisticRegression 支持 KernelExplainer (较慢)
    "SVC" # SVC 也支持 KernelExplainer (较慢)
]
SHAP_MAX_DISPLAY = 1000 # 用于 SHAP 图绘制时，X数据的最大行数，防止内存溢出
SHAP_TOP_DEPENDENCE_PLOTS = 3 # 绘制最重要的N个特征的依赖图 (如果以后要用)

# --- 有监督学习模型配置 ---
SUPERVISED_MODELS = {
    "RandomForestClassifier": {
        "type": "classification",
        "model_class": RandomForestClassifier,
        "params": {"random_state": RANDOM_STATE, "class_weight": "balanced"},
        "param_grid": {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        "scoring": "f1_weighted"
    },
    "LogisticRegression": {
        "type": "classification",
        "model_class": LogisticRegression,
        "params": {"random_state": RANDOM_STATE, "solver": "liblinear", "class_weight": "balanced"},
        "param_grid": {
            'C': [0.01, 0.1, 1, 10, 100]
        },
        "scoring": "f1_weighted"
    },
    "SVC": {
        "type": "classification",
        "model_class": SVC,
        "params": {"random_state": RANDOM_STATE, "probability": True, "class_weight": "balanced"},
        "param_grid": {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear']
        },
        "scoring": "f1_weighted"
    },
    "KNeighborsClassifier": {
        "type": "classification",
        "model_class": KNeighborsClassifier,
        "params": {},
        "param_grid": {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        },
        "scoring": "f1_weighted"
    },
    "XGBClassifier": {
        "type": "classification",
        "model_class": xgb.XGBClassifier,
        "params": {"objective": "binary:logistic", "eval_metric": "logloss", "use_label_encoder": False, "random_state": RANDOM_STATE},
        "param_grid": {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7]
        },
        "scoring": "f1_weighted"
    },
    "LightGBM_FocalLoss": {
        "type": "classification",
        "params": {
            # 这些参数将作为 LightGBM.train 的 params 字典的一部分
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": -1,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "n_estimators": 100,   # num_boost_round
            "n_jobs": -1,
            "scale_pos_weight": 5, 
            "verbosity": -1,       
            "random_state": RANDOM_STATE 
        },
        "focal_loss_params": {
            "alpha_list": [0.04,0.07,0.25,0.26,0.95,0.43,0.46,0.57,0.64,0.74,0.76,0.19,0.82,0.87],   #[i/100.0 for i in range(1,100)]     [0.25,0.26,0.95,0.43,0.46,0.64,0.74,0.82,0.87]
            "gamma_list": [2,3,4,5,6,7,8,9],
            "thresholds": [0.2,0.3,0.4,0.5,0.7]
        },
        "scoring": "f1_weighted"
    }
}

# --- 无监督学习模型配置 ---
UNSUPERVISED_MODELS = {
    "Autoencoder_UMAP_KMeans": {
        "type": "clustering",
        "autoencoder": {
            "input_dim": 6500, # 根据你的数据维度调整
            "encoding_dim": 64,
            "hidden_dims": [128, 256],
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "pretrain_path": MODELS_OUTPUT_DIR / "autoencoder_model.pth",
            "load_pretrained": True,
            "save_pretrained": True
        },
        "reducer": {
            "type": "UMAP",
            "n_components": 2,
            "random_state": RANDOM_STATE
        },
        "clusterer": {
            "type": "KMeans",
            "n_clusters": 3,
            "random_state": RANDOM_STATE,
            "find_optimal_k": True,
            "max_k_for_elbow": 10
        },
        "evaluation_metrics": ["silhouette_score", "calinski_harabasz_score"]
    },
    "Autoencoder_UMAP_DBSCAN": {
        "type": "clustering",
        "autoencoder": {
            "input_dim": 6500, # 根据你的数据维度调整
            "encoding_dim": 64,
            "hidden_dims": [128, 256],
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "pretrain_path": MODELS_OUTPUT_DIR / "autoencoder_model.pth",
            "load_pretrained": True,
            "save_pretrained": True
        },
        "reducer": {
            "type": "UMAP",
            "n_components": 2,
            "random_state": RANDOM_STATE
        },
        "clusterer": {
            "type": "DBSCAN",
            "eps": 0.5,
            "min_samples": 5,
        },
        "evaluation_metrics": ["silhouette_score", "calinski_harabasz_score"]
    }
}


# --- K 折交叉验证 / 整体训练模式选择 ---
# 建议：实际实验参数（是否K折、折数、测试集比例等）请在 main.py 里配置和传递。
# 这里仅提供默认值，供 main.py 或 pipeline 调用时作为默认参数。
USE_K_FOLD_CV: bool = False # 默认不使用K折，main.py可覆盖
TEST_SIZE_OVERALL: float = 0.2 # 默认20%测试集，main.py可覆盖
N_SPLITS_SUPERVISED: int = 5 # 默认5折，main.py可覆盖

# GridSearchCV 内部交叉验证的折叠数
N_SPLITS_GRIDSEARCH: int = 3 

# --- SHAP Configuration ---
GENERATE_SHAP_PLOTS: bool = True
SHAP_SUPPORTED_MODELS: list = ["RandomForestClassifier", "XGBClassifier", "LGBMClassifier", "LightGBM_FocalLoss", "LogisticRegression"]
SHAP_MAX_DISPLAY: int = 20 # SHAP 图中显示的最大样本数，避免内存问题


