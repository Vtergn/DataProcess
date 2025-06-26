import pandas as pd
import re
import numpy as np
from pathlib import Path
import os
import openpyxl

# --- 1. 定义绝对路径变量 ---
BASE_DIR = Path(__file__).resolve().parent

# 请确保这些路径指向你的实际文件
# 注意:文件路径使用 Path 对象时，可以直接使用正斜杠 '/'，或者使用原始字符串 r"..."
clinical_data_file_path = Path("D:/jinlu/Documents/Lab_project/Lab_project/产妇抑郁/产妇抑郁数据/孕妇异常情绪(临床信息).xlsx")
still_data_path = Path("D:/jinlu/Documents/Lab_project/Lab_project/产妇抑郁/产妇抑郁数据/分段特征/test1-4-32.xlsx")
                        
mission_data_path = Path("D:/jinlu/Documents/Lab_project/Lab_project/产妇抑郁/产妇抑郁数据/分段特征/test2-4-32.xlsx")
frq_split_num = 32       # 频段分割数目,用于计算能量比值特征
output_dir = Path("D:/MyProjects/LabDataProcess/Data/processed")
output_dir.mkdir(parents=True, exist_ok=True) # 确保输出目录存在
output_file_path = output_dir / "产妇抑郁.xlsx"
output_sheet_name = "32频段(0.4Hz)" # 你可以修改这个默认值

# --- 2. 核心数据加载函数 ---
def clean_numeric_data(column):
    """
    清理 Series 中的非数字字符,并转换为数值类型。
    """
    column = column.astype(str).apply(lambda x: re.sub(r'[^0-9.-]', '', x))
    return pd.to_numeric(column, errors='coerce')

def process_excel(file_path, selected_columns):
    """
    加载并初步处理临床数据，包括数值清理和手动行丢弃。
    特别注意:这里的行号指的是原始 Excel 文件中的行号 (从1开始计数)。
    """
    print(f"正在加载临床数据: {file_path}")
    # 读取Excel文件，默认header=0 (第一行是表头)
    df = pd.read_excel(file_path, usecols=selected_columns)
    df_cleaned = pd.DataFrame(columns=selected_columns)

    for col in df.columns:
        df_cleaned[col] = clean_numeric_data(df[col])

    # 根据用户指定索引手动丢弃行,并重置索引
    # 这些行被认为是无效的，其对应的阻抗数据也应被移除 (但在这里我们只处理临床数据)
    # rows_to_drop_clinical 是原始Excel的行号 (从1开始)
    # 所以需要转换为0-based索引:row_num - 1
    rows_to_drop_clinical_1_based = [62, 204, 205, 206, 207]
    # 转换为 0-based 索引
    rows_to_drop_0_based = [idx - 1 for idx in rows_to_drop_clinical_1_based if (idx - 1) >= 0 and (idx - 1) < len(df_cleaned)]

    if rows_to_drop_0_based:
        print(f"正在从临床数据中丢弃0-based索引为 {rows_to_drop_0_based} 的行...")
        # drop方法需要的是DataFrame的索引值，而不是行号
        df_cleaned = df_cleaned.drop(index=rows_to_drop_0_based)
        df_cleaned = df_cleaned.reset_index(drop=True) # 重新设置索引，确保连续性
    else:
        print("没有找到要手动丢弃的临床数据行或索引已不存在。")

    print("\n临床数据缺失值统计 (处理后):")
    print(df_cleaned.isnull().sum())

    stats = df_cleaned.describe().loc[['mean', 'std']]
    return df_cleaned, stats

# --- 3. 标签生成和数据对齐函数 ---
def label_groups(df):
    """
    根据临床数据生成分类标签。
    注意:如果数值不在任何 bin 范围内,pd.cut 会生成 NaN。
    """
    print("\n正在生成临床标签...")
    # 使用 pd.cut 时，bins 的边界值需要能够覆盖所有数据。
    # -inf 和 inf 用于确保所有值都被包含。
    df['AFI_group'] = pd.cut(df['产前超声AFI'], bins=[-np.inf, 10, 20, np.inf], ordered=False, labels=[0, 1, 0])
    df['出血量_group'] = pd.cut(df['出血量'], bins=[-np.inf, 300, 1000, np.inf], ordered=False, labels=[0, 1, 1])
    df['SAS_group'] = pd.cut(df['SAS焦虑'], bins=[-np.inf, 40, 50, np.inf], ordered=False, labels=[0, 1, 1])
    df['SDS_group'] = pd.cut(df['SDS抑郁'], bins=[-np.inf, 40, 50, np.inf], ordered=False, labels=[0, 1, 1])
    return df

def align_labels_to_impedance(labels_df):
    """
    将标签 DataFrame 中的每一列重复两次,以匹配阻抗数据的行数。
    不执行 NaN 过滤,NaN 值会被保留和扩展。

    Args:
        labels_df (pd.DataFrame): 包含所有组标签的 DataFrame (N_clinical_valid, L)。
                                  N_clinical_valid 是经过手动行删除后的临床样本数。

    Returns:
        pd.DataFrame: 扩展后的标签 DataFrame (2*N_clinical_valid, L),准备与阻抗特征合并。
    """
    print("正在扩展标签以匹配阻抗数据行...")
    expanded_labels_data = {}
    for col_name in labels_df.columns:
        # 对每个标签列,将其值重复两次
        expanded_labels_data[col_name] = np.repeat(labels_df[col_name].to_numpy(), 2)

    # 创建新的 DataFrame,保留原始列名
    return pd.DataFrame(expanded_labels_data, columns=labels_df.columns)

# --- 主执行流程 ---
if __name__ == "__main__":
    print(f"--- 启动数据预处理脚本 ---")

    # 1. 临床数据处理
    selected_clinical_columns = ["产前超声AFI", "出血量", "SAS焦虑", "SDS抑郁"]
    # cleaned_data 包含了原始数据以及经过手动删除行后的临床信息
    cleaned_data, statistics = process_excel(clinical_data_file_path, selected_clinical_columns)
    print("\n临床数据统计信息:")
    print(statistics)

    # 生成所有标签列,其中可能包含 NaN
    cleaned_data = label_groups(cleaned_data)

    print("\n临床数据分组结果 (包含 NaN 统计):")
    group_counts = {
        'AFI_group': cleaned_data['AFI_group'].value_counts(dropna=False), # 包含 NaN 计数
        '出血量_group': cleaned_data['出血量_group'].value_counts(dropna=False),
        'SAS_group': cleaned_data['SAS_group'].value_counts(dropna=False),
        'SDS_group': cleaned_data['SDS_group'].value_counts(dropna=False)
    }
    for key, value in group_counts.items():
        print(f"{key}:")
        print(value, "\n")

    # 2. 阻抗数据处理
    print("正在加载阻抗数据...")
    # header=None 表示没有表头，第一行是数据
    # df_still 和 df_mission 都是从Excel文件读取的原始DataFrame
    df_still = pd.read_excel(still_data_path, header=None)
    df_mission = pd.read_excel(mission_data_path, header=None)

    # 从 Excel 中提取数值部分 (跳过第一行作为表头,第一列作为索引/ID)
    # 转换为numpy数组时，依然是0-based索引
    data1_raw = df_still.to_numpy()[1:, 1:]
    data2_raw = df_mission.to_numpy()[1:, 1:]

    # 显式地将数据转换为浮点类型,确保后续数值运算的正确性
    try:
        data1_raw = data1_raw.astype(float)
        data2_raw = data2_raw.astype(float)
        print("阻抗数据成功转换为浮点类型。")
    except ValueError as e:
        print(f"警告: 阻抗数据转换浮点类型时发生错误,可能存在非数字值。错误信息: {e}")
        print("尝试使用 pandas.to_numeric 进行更鲁棒的转换...")
        # 更鲁棒的转换，但会失去原始的二维结构，需要 reshape
        data1_raw = pd.to_numeric(data1_raw.flatten(), errors='coerce').to_numpy().reshape(data1_raw.shape)
        data2_raw = pd.to_numeric(data2_raw.flatten(), errors='coerce').to_numpy().reshape(data2_raw.shape)
        print("已尝试使用 pandas.to_numeric 转换,非数字值将变为 NaN。")

    # ！！！关键修正点:根据最新理解，阻抗数据本身是完整且无需通过行号过滤的！！！
    # 之前基于临床数据删除阻抗行的逻辑已移除。
    # 此时，data1_raw 和 data2_raw 应该就是最终用于合并的阻抗数据。
    data1_filtered = data1_raw
    data2_filtered = data2_raw
    # 注意:如果未来发现阻抗数据也需要过滤，需要在clinical_data_file_path和still_data_path中添加ID
    # 然后基于ID进行匹配过滤，而不是依赖行号。

    merged_data = np.hstack((data1_filtered, data2_filtered))

    one_data_size = data1_filtered.shape[1]
    print(f"\n单侧阻抗数据特征数量 (过滤后): {one_data_size}")

    # 获取原始阻抗特征的名称列表 (从原始 df_still 的表头获取)
    # 假设第一行是特征名
    original_impedance_feature_names = df_still.to_numpy()[0, 1:].tolist()

    # 添加能量比值特征
    # 这里使用过滤后的数据切片 (现在就是完整的阻抗数据)
    # 确保切片范围正确，-frq_split_num-1:-1 表示倒数 frq_split_num+1 个元素到倒数第二个元素
    # 如果 frq_split_num = 8，这个切片会取出 8 个频段数据
    data1_slice = data1_filtered[:, -frq_split_num-1:-1]
    data2_slice = data2_filtered[:, -frq_split_num-1:-1]

    # 使用 np.true_divide 处理除以零的情况,将结果设为 0
    ratio_feature = np.true_divide(data1_slice, data2_slice, out=np.zeros_like(data1_slice, dtype=float), where=data2_slice != 0)
    merged_data = np.hstack((merged_data, ratio_feature))

    # 生成最终特征列名
    still_cols_expanded = [f"(静坐){name}" for name in original_impedance_feature_names]
    mission_cols_expanded = [f"(任务){name}" for name in original_impedance_feature_names]

    # 确保 ratio_base_feature_names 的长度和实际计算的比例特征数量一致
    # 假设 original_impedance_feature_names 已经包含了需要计算比值的特征
    # 这里的切片应该与上面 data1_slice 和 data2_slice 的切片逻辑一致
    ratio_base_feature_names = original_impedance_feature_names[-frq_split_num-1:-1]
    ratio_cols_expanded = [f"{name}_比值" for name in ratio_base_feature_names]

    final_feature_names = still_cols_expanded + mission_cols_expanded + ratio_cols_expanded

    # 检查特征名列表长度与合并后数据列数是否匹配
    if len(final_feature_names) != merged_data.shape[1]:
        print(f"警告: 生成的特征名列表长度 ({len(final_feature_names)}) 与合并后的数据列数 ({merged_data.shape[1]}) 不匹配！")
        print("请检查 `frq_split_num`、切片范围以及 `original_impedance_feature_names` 的来源。")


    # 3. 标签对齐 (仅扩展，不过滤 NaN)
    # cleaned_data 已经是过滤掉无效行后的临床数据
    labels_to_expand = cleaned_data[['AFI_group', '出血量_group', 'SAS_group', 'SDS_group']]
    expanded_labels_df = align_labels_to_impedance(labels_to_expand)

    # --- ！！！核心对齐检查:确保最终阻抗数据和扩展标签行数匹配 ！！！ ---
    current_impedance_rows = merged_data.shape[0]
    current_labels_rows_expanded = expanded_labels_df.shape[0] # 已经经过 expand_labels_to_impedance 处理

    print(f"\n对齐前最终检查:")
    print(f"阻抗数据 (处理后)行数: {current_impedance_rows}")
    print(f"扩展标签数据行数: {current_labels_rows_expanded}")

    if current_impedance_rows != current_labels_rows_expanded:
        print(f"\n!!!!致命错误!!!! 阻抗数据行数 ({current_impedance_rows}) 与扩展标签行数 ({current_labels_rows_expanded}) 不匹配！")
        print("这表明临床数据和阻抗数据在病人数量上存在根本性不一致，无法继续。")
        print("请再次检查:")
        print("1. `临床数据`中需要移除的行号是否准确。")
        print("2. `测试数据` (test18.xlsx, test28.xlsx) 是否**完全**对应剩余的有效临床病人 (每人两组数据)。")
        exit() # 遇到致命错误，程序退出

    print(f"\n最终对齐后的阻抗数据形状: {merged_data.shape}")
    print(f"最终扩展后的标签数据形状: {expanded_labels_df.shape}")

    # --- 4. 保存最终数据到 Excel ---
    print("\n--- 准备保存数据到 Excel ---")

    # 将 numpy 数组转换为 DataFrame, 使用 `final_feature_names` 作为列名
    df_final_features = pd.DataFrame(merged_data, columns=final_feature_names)

    # 添加扩展后的标签列 (包含 NaN)
    # 使用 .astype(object) 可以确保 NaN 被保留为 Python 的 None 或 NaN 对象,而不是强制转换为整数导致 NaN 丢失。
    # Excel 导出时会正确处理这些 NaN。
    df_final_features['Label1_AFI'] = expanded_labels_df['AFI_group'].astype(object)
    df_final_features['Label2_出血量'] = expanded_labels_df['出血量_group'].astype(object)
    df_final_features['Label3_SAS焦虑'] = expanded_labels_df['SAS_group'].astype(object)
    df_final_features['Label4_SDS抑郁'] = expanded_labels_df['SDS_group'].astype(object)

    print(f"最终待保存数据的形状: {df_final_features.shape}")
    print(f"数据将保存到: {output_file_path}")

    try:
        # 使用 pandas.ExcelWriter 来追加写入到不同的 sheet
        # mode='a' 表示追加模式，如果文件不存在则创建
        # if_sheet_exists='replace' 表示如果 sheet 已存在则替换，
        # 也可以使用 'overlay' 或 'error'。'replace'通常更安全。
        with pd.ExcelWriter(output_file_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            df_final_features.to_excel(writer, index=False, sheet_name=output_sheet_name)
        print(f"数据成功保存到 {output_file_path} 的 '{output_sheet_name}' sheet!")
        print("注意：如果这是第一次写入此文件，它会自动创建。")
        print("如果文件或 sheet 之前存在，'replace' 模式会替换该 sheet。")
    except FileNotFoundError:
        # 如果文件不存在，ExcelWriter 会自动创建，但是如果父目录不存在，则会抛出
        # 这里的 FileNotFoundError 检查更偏向于文件本身不存在，而不是目录
        # 但由于之前已经用 output_dir.mkdir(parents=True, exist_ok=True) 确保了目录，
        # 这种情况通常不会发生，除非 output_file_path 有问题
        print(f"错误: 文件 '{output_file_path}' 不存在或无法创建。请检查路径。")
    except Exception as e:
        print(f"保存数据到 Excel 时发生错误: {e}")
        print("请确保 'openpyxl' 库已安装 (pip install openpyxl)。")

    print(f"--- 数据预处理脚本执行完毕 ---")