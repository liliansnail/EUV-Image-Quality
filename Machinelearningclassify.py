import pandas as pd
import time
import joblib
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'
import os
"""
# 1. 加载数据
fused = np.load(r"D:\FY03IMG\database\simdegration\processed_out\result\fused_features.npy")

# 2. 分离特征和标签

X = fused[:, :-1]  # 所有列，除了最后一列
y = fused[:, -1]   # 最后一列是 level 标签

print("X shape:", X.shape)  # 应该是 [N, D]
print("y shape:", y.shape)  # 应该是 [N,]
"""
# 1读取数据
csv_path = r"D:\FY03IMG\database\simdegration\processed_out\result\average_newmetrics_top5withHyperIQAPCA.csv"
data = pd.read_csv(csv_path)
# 2. 提取特征和标签
X = data.drop(columns=['filename', 'level'])
y = data['level']
# 3. 划分训练集和测试集
# 3. 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 定义模型
models = {
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42)
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    # 训练时间
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    train_time = end_train - start_train

    # 保存模型
    model_filename = rf"D:\FY03IMG\database\simdegration\processed_out\result\{name}_feature22withHyperIQAPCA.pkl"

    joblib.dump(model, model_filename)
    print(f"{name} 模型已保存到 {model_filename}")

    # 预测时间
    start_pred = time.time()
    y_pred = model.predict(X_test)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_test)
    except Exception:
        # 有些模型可能不支持predict_proba
        pass
    end_pred = time.time()
    pred_time = end_pred - start_pred

    # 计算性能指标
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # AUC 多分类处理
    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
        except Exception as e:
            print(f"AUC计算异常: {e}")

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{name} Performance:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    if auc is not None:
        print(f"AUC      : {auc:.4f}")
    print(f"Train time  : {train_time:.4f} sec")
    print(f"Predict time: {pred_time:.4f} sec")
    print(f"Confusion Matrix:\n{cm}")

    # 保存结果
    result = {
        'model': name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'auc': auc,
        'train_time_sec': train_time,
        'predict_time_sec': pred_time,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }
    results.append(result)
model_names = [r['model'] for r in results]
train_times = [r['train_time_sec'] for r in results]
pred_times = [r['predict_time_sec'] for r in results]
metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
# 5. 保存测试集的预测结果到csv
pred_df = pd.DataFrame({
    'filename': y_pred,
    'true_level': y_test
})
for r in results:
    pred_df[f'pred_{r["model"]}'] = r['y_pred']

save_path = r"D:\FY03IMG\database\simdegration\processed_out\result\model_predictions_with_feature22withHyperIQAPCA.csv"
save_fig_dir=r"D:\FY03IMG\database\simdegration\processed_out\result"
pred_df.to_csv(save_path, index=False)
print(f"\n所有模型的测试集预测结果已保存到 {save_path}")
# 1. 训练时间和预测时间对比条形图
plt.figure(figsize=(10, 5))
bar_width = 0.35
index = np.arange(len(model_names))

plt.bar(index, train_times, bar_width, label='Train Time (s)')
plt.bar(index + bar_width, pred_times, bar_width, label='Predict Time (s)')

plt.xlabel('Model')
plt.ylabel('Time (seconds)')
plt.title('Training and Prediction Time Comparison')
plt.xticks(index + bar_width / 2, model_names)
plt.legend()
plt.tight_layout()

time_fig_path = os.path.join(save_fig_dir, r"time_comparison_feature22withHyperIQAPCA.png")
plt.savefig(time_fig_path)
print(f"训练和预测时间图已保存到 {time_fig_path}")
plt.show()

# 2. 评价指标条形图
plt.figure(figsize=(12, 6))

for i, metric in enumerate(metrics):
    plt.subplot(2, 3, i+1)
    values = []
    for r in results:
        v = r[metric]
        if v is None:
            v = 0
        values.append(v)
    sns.barplot(x=model_names, y=values)
    plt.title(metric.capitalize())
    plt.ylim(0, 1)
    plt.ylabel('')
    plt.xlabel('')

plt.tight_layout()

metrics_fig_path = os.path.join(save_fig_dir, r"metrics_comparison_feature22withHyperIQAPCA.png")
plt.savefig(metrics_fig_path)
print(f"评价指标图已保存到 {metrics_fig_path}")
plt.show()

# 3. 混淆矩阵热力图保存
for r in results:
    cm = r['confusion_matrix']
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{r['model']} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')

    cm_fig_path = os.path.join(save_fig_dir, f"confusion_matrix_{r['model']}_feature22withHyperIQAPCA.png")
    plt.savefig(cm_fig_path)
    print(f"{r['model']} 混淆矩阵图已保存到 {cm_fig_path}")
    plt.show()
