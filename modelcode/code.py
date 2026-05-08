import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 设置图片清晰度和字体
plt.rcParams['figure.dpi'] = 600
plt.rcParams['font.family'] = ['Times New Roman']

# 1. 直接读取训练集和测试集文件
X_train_path = 'X_train1.xlsx'
y_train_path = 'Y_train_YD.xlsx'
X_test_path = 'X_test1.xlsx'
y_test_path = 'Y_test_YD.xlsx'

try:
    # 读取训练集
    X_train_original = pd.read_excel(X_train_path)
    y_train = pd.read_excel(y_train_path).iloc[:, 0] 
    
    # 读取测试集
    X_test_original = pd.read_excel(X_test_path)
    y_test = pd.read_excel(y_test_path).iloc[:, 0] 
    
    print(f"训练集输入形状: {X_train_original.shape}, 训练集输出形状: {y_train.shape}")
    print(f"测试集输入形状: {X_test_original.shape}, 测试集输出形状: {y_test.shape}")

    # 2. 数据预处理
    categorical_cols = ['灌溉方式', '耕作模式', '覆盖模式', '作物类型', '灌溉时期']
    label_encoders = {}

    # 分类特征编码
    for col in categorical_cols:
        if col in X_train_original.columns:
            le = LabelEncoder()
            X_train_original[col] = le.fit_transform(X_train_original[col].astype(str))
            X_test_original[col] = le.transform(X_test_original[col].astype(str))
            label_encoders[col] = le

    # 填补缺失值
    X_train_original.fillna(X_train_original.mean(numeric_only=True), inplace=True)
    X_test_original.fillna(X_test_original.mean(numeric_only=True), inplace=True)

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_original)
    X_test = scaler.transform(X_test_original)

    # 3. 建立随机森林模型（设置random_state=100）
    model = RandomForestRegressor(
        n_estimators=100,          
        random_state=10000,          
        n_jobs=-1,                 
        verbose=0
    )
    model.fit(X_train, y_train)

    # 4. 模型评估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R²:", round(r2, 4))

    # 5. 生成预测对比图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='#00A1FF', zorder=3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, color='#5ed935', zorder=2)
    plt.grid(True, linestyle='--', alpha=0.3, zorder=1)
    plt.xlabel('Actual Yield', fontsize=12)
    plt.ylabel('Predicted Yield', fontsize=12)
    plt.title(f'Random Forest Prediction vs Actual\n(R²={r2:.4f}, RMSE={rmse:.2f})', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('RF_YD_Prediction_Comparison.png', dpi=600)
    plt.show()

    # 6. 保存对比结果
    result_df = pd.DataFrame({
        'Actual_Yield': y_test,
        'Predicted_Yield': y_pred,
        'Absolute_Error': y_test - y_pred,
        'Relative_Error(%)': ((y_test - y_pred) / y_test * 100).round(2)
    })
    result_df = result_df.sort_values('Actual_Yield')
    result_df.to_excel('RF_YD_Prediction_Results.xlsx', index=False)
    print("对比结果已保存")

    # 7. 特征重要性分析
    def get_feature_importance(model, X_train_original, categorical_cols, top_n=15):
        importance = model.feature_importances_
        feature_names = X_train_original.columns.tolist()
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        return importance_df.sort_values('Importance', ascending=False).head(top_n)

    # 计算特征重要性
    importance_df = get_feature_importance(model, X_train_original, categorical_cols)
    print("\nTop 5 Features by Importance:")
    print(importance_df.head(5))

    # 8. 绘制特征重要性图
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='#00A1FF')
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Feature Name', fontsize=12)
    plt.title('Random Forest Feature Importance Ranking', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('RF_YD_Feature_Importance.png', dpi=600)
    plt.show()

    # 9. 保存特征重要性到Excel
    importance_path = 'RF_YD_Feature_Importance_Results.xlsx'
    importance_df.to_excel(importance_path, index=False)
    print(f"特征重要性已保存至: {importance_path}")

except Exception as e:
    print(f"处理文件时出错: {str(e)}")
