import os  # 导入操作系统模块
import re  # 导入正则表达式模块用于处理文件名中的特殊字符
import numpy as np  # 导入 NumPy 进行数值计算
import pandas as pd  # 导入 Pandas 进行数据处理
import matplotlib.pyplot as plt  # 导入 Matplotlib 绘图模块
import matplotlib.gridspec as gridspec  # 导入 GridSpec 用于自定义网格布局
import matplotlib.colors as mcolors  # 导入 mcolors 用于自定义颜色映射
import matplotlib.font_manager as fm  # 导入字体管理器
import seaborn as sns  # 导入 Seaborn 用于统计图表绘制
import shap  # 导入 SHAP 库用于模型可解释性分析
import xgboost as xgb  # 导入 XGBoost 模型库
import networkx as nx # 导入 NetworkX 用于绘制关系网络图
from sklearn.pipeline import Pipeline  # 导入 Pipeline 用于构建数据流水线
from sklearn.impute import SimpleImputer  # 导入 SimpleImputer 用于缺失值插补
from sklearn.feature_selection import VarianceThreshold  # 导入 VarianceThreshold 移除零方差特征
from sklearn.preprocessing import StandardScaler  # 导入 StandardScaler 进行特征标准化
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split  # 导入超参数搜索与数据集划分模块
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入回归评估指标
import statsmodels.api as sm  # 导入 statsmodels 用于拟合平滑曲线
from mpl_toolkits.axes_grid1 import make_axes_locatable  # 导入坐标轴分割工具
import matplotlib.lines as mlines # 导入图例线条对象工具

# ==========================================
# 1. 全局配置与参数设置
# ==========================================
CONFIG = {  # 全局参数配置字典
    "data_file": "data-yield.xlsx",  # 输入数据路径
    #"data_file": "data-WUE.xlsx",  # 运行时去掉注释
    "output_dir": "output",  # 输出结果目录
    "dpi": 300,  # 图像输出分辨率
    "formats": ["png", "pdf"],  # 图像保存格式
    
    # 字体配置
    "font_family_en": "Times New Roman",  # 默认英文字体
    "font_family_zh": "SimSun",  # 默认中文字体 (宋体)
    "label_fontsize": 14,  # 坐标轴标签字体大小
    "title_fontsize": 16,  # 图表标题字体大小
    "tick_fontsize": 14,  # 坐标轴刻度字体大小
    
    # 图1 模型性能验证图 参数配置
    "fig1_kde_linewidth": 1.5,       # 图1 KDE 曲线线宽
    "fig1_hist_bins": 50,            # 图1 直方图箱数
    "fig1_train_color": "#A9A9A9",   # 图1 训练集散点颜色
    "fig1_test_color": "#9F3E3F",    # 图1 测试集散点颜色
    "fig1_scatter_s": 30,            # 图1 散点大小
    "fig1_scatter_edgecolor": "white", # 图1 散点描边颜色
    "fig1_scatter_linewidth": 0.5,   # 图1 散点描边线宽
    
    # 图2 SHAP全局特征贡献图参数配置
    "fig2_scatter_s": 12,            # 图2 散点大小
    "fig2_bar_color": "#C3D3F2",     # 图2 柱状图颜色
    
    # 图3 SHAP单特征依赖图 参数配置
    "fig3_scatter_s": 20,            # 图3 散点大小
    "fig3_curve_color": "#FF451B",   # 图3 拟合曲线颜色  
    "fig3_curve_linewidth": 1.5,     # 图3 拟合曲线线宽
    "fig3_cbar_fontsize": 10,        # 图3 色带刻度字体大小
    "fig3_positive_color": "#DEF4F1",# 图3 SHAP正向(最优)象限填充颜色 
    "fig3_negative_color": "#E6DADA",# 图3 SHAP负向(敏感)象限填充颜色 
    
    # 图4 SHAP主效应与交互效应对比图 参数配置
    "fig4_main_color": "#71BCB1",    # 图4 主效应柱状图颜色
    "fig4_inter_color": "#9F6566",   # 图4 交互效应柱状图颜色
    
    # =======================================================
    # 图5、图7、图8 动态特征展示数量配置
    # =======================================================
    "fig5_max_features": 8,          # 控制图5复合矩阵图展示的核心特征数量
    "fig7_max_features": 18,         # 控制图7网络图展示的节点数量
    "fig8_max_display": 15,          # 图8 热力图 Y 轴最大显示特征数

    # 图5 特征交互效应复合矩阵图 细节参数配置
    "fig5_label_fontsize": 16,       # 复合矩阵图的行列特征名称字体大小
    "fig5_tick_fontsize": 11,        # 复合矩阵图顶部X轴刻度(标签)字体大小
    "fig5_cbar_fontsize": 16,        # 色带刻度字体大小
    
    # 图7 特征影响-交互网络图 参数配置
    "fig7_node_color": "#48A597",    # 图7 节点重要性渐变主色
    "fig7_edge_color": "#9C1A1C",    # 图7 连线交互强度渐变主色

    # 图8 参数配置
    "fig8_figsize": (10, 6),         # 图8 画布尺寸
    
    # 图9 个体样本SHAP力图(SHAP Force Plot) 参数配置
    "fig9_max_instances": 50,        # 图9 最大绘制样本数量限制
    "fig9_figsize": (16, 4),         # 图9 画布尺寸
    "fig9_color_pos": "#9C1A1C",     # 图9 正向 SHAP 值颜色
    "fig9_color_neg": "#48A597",     # 图9 负向 SHAP 值颜色

    # 图10 二维 PDP 部分偏依赖图(2D PDP) 参数配置
    "fig10_line_color_q1": "#2A6F97", # 图10 Q1等高线颜色
    "fig10_line_color_med": "#48A597",# 图10 中位数等高线颜色
    "fig10_line_color_q3": "#9C1A1C", # 图10 Q3等高线颜色
    
    # 全局色彩映射配置，Teal-Grey-Red Diverging Colormap
    "shap_cmap": mcolors.LinearSegmentedColormap.from_list("custom_cmap", [ "#48A597", "#FFFFFF", "#9C1A1C"]), # 自定义连续色彩映射
    "random_state": 42  # 随机数种子
}

# ==========================================
# 2. 字体与系统底层配置
# ==========================================
available_fonts = [f.name for f in fm.fontManager.ttflist]  # 获取系统可用字体列表
en_font = CONFIG["font_family_en"]  # 获取英文字体设置
zh_font = CONFIG["font_family_zh"]  # 获取中文字体设置

if en_font not in available_fonts:  # 检查英文字体是否安装
    print(f"警告：系统缺失英文字体 '{en_font}'，将使用备用字体。")
if zh_font not in available_fonts:  # 检查中文字体是否安装
    print(f"警告：系统缺失中文字体 '{zh_font}'，中文可能无法正常显示。")

font_fallback_list = [en_font, zh_font, 'sans-serif']  # 设置字体回退列表
plt.rcParams['font.sans-serif'] = font_fallback_list  # 配置全局无衬线字体
plt.rcParams['font.serif'] = font_fallback_list  # 配置全局有衬线字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示坐标轴负号

sns.set_theme(style="ticks", rc={  # 统一 Seaborn 图表样式与字体
    "font.sans-serif": font_fallback_list,
    "font.serif": font_fallback_list,
    "font.family": font_fallback_list,
    "axes.unicode_minus": False
})

# ==========================================
# 3. 辅助函数
# ==========================================
def save_figure(fig, filename_base, subfolder=None):
    """保存 Matplotlib 图形至指定目录"""
    save_dir = CONFIG["output_dir"]  # 获取输出根目录
    if subfolder:  # 若存在子文件夹
        save_dir = os.path.join(save_dir, subfolder)  # 拼接路径
    os.makedirs(save_dir, exist_ok=True)  # 创建目标目录
    
    # 替换文件名中的特殊字符（把斜杠换成下划线），避免路径解析错误
    safe_filename_base = str(filename_base).replace('/', '_').replace('\\', '_')
    
    for fmt in CONFIG["formats"]:  # 遍历保存格式
        filepath = os.path.join(save_dir, f"{safe_filename_base}.{fmt}")  # 生成完整路径
        fig.savefig(filepath, dpi=CONFIG["dpi"], format=fmt, bbox_inches='tight')  # 保存图像
    plt.close(fig)  # 关闭图像对象释放内存

# ==========================================
# 4. 核心分析类构建
# ==========================================
class XGBoostXAIAnalyzer:
    def __init__(self):
        """初始化分析器属性"""
        self.random_state = CONFIG["random_state"]  # 初始化随机数种子
        self.cv = 5  # 交叉验证折数
        self.model = None  # 模型实例
        self.pipeline = None  # 流水线实例
        self.explainer = None  # SHAP 解释器实例
        self.shap_values_obj = None  # SHAP 值对象
        self.shap_interaction_values = None  # SHAP 交互作用值矩阵
        self.X_train_processed = None  # 预处理后的训练集特征
        
    def _preprocess_steps(self, scale: bool):
        """定义数据预处理步骤"""
        steps = [
            ('imputer', SimpleImputer(strategy="median")),  # 中位数插补
            ('var', VarianceThreshold(threshold=0.0)),  # 移除零方差特征
        ]
        if scale:  # 判断是否进行特征标准化
            steps.append(('scaler', StandardScaler()))  # 标准化处理
        return steps

    def _fit_random(self, name, reg, param_dist, X_train, y_train, scale: bool, n_iter=20):
        """随机搜索优化超参数"""
        pipe = Pipeline(self._preprocess_steps(scale=scale) + [('reg', reg)])  # 构建流水线
        rscv = RandomizedSearchCV(
            pipe, param_distributions=param_dist, n_iter=n_iter,  # 配置参数分布和迭代次数
            scoring='r2', cv=self.cv, n_jobs=-1, random_state=self.random_state, verbose=0
        )
        rscv.fit(X_train, y_train)  # 执行随机搜索
        print(f"{name} 最优参数: {rscv.best_params_}")  # 打印最优参数
        return name, rscv.best_estimator_

    def train_xgboost(self, X_train, y_train):
        """训练 XGBoost 并进行参数调优"""
        xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=self.random_state, tree_method='hist', n_jobs=-1)  # 初始化 XGBoost 回归器
        
        param_dist = {  # 定义随机搜索超参数空间
            'reg__n_estimators': [300, 500, 800, 1000],  # 树数量
            'reg__max_depth': [3, 4, 6, 8],  # 最大深度
            'reg__learning_rate': [0.01, 0.05, 0.1],  # 学习率
            'reg__subsample': [0.7, 0.85, 1.0],  # 样本采样率
            'reg__colsample_bytree': [0.7, 0.85, 1.0],  # 特征采样率
            'reg__reg_alpha': [0, 0.1, 0.5],  # L1 正则化系数
            'reg__reg_lambda': [1.0, 1.5, 2.0]  # L2 正则化系数
        }
        
        _, best_pipeline = self._fit_random("XGBoost", xgb_reg, param_dist, X_train, y_train, scale=False, n_iter=28)  # 执行调参拟合
        self.pipeline = best_pipeline  # 保存最优流水线
        self.model = best_pipeline.named_steps['reg']  # 提取模型实例
        return self.pipeline

    def prepare_data(self):
        """读取数据并划分数据集"""
        print("正在读取数据表...")
        df = pd.read_excel(CONFIG["data_file"])  # 读取 Excel 数据
        self.y = df.iloc[:, 0]  # 提取目标变量 (第一列)
        self.X = df.iloc[:, 1:]  # 提取特征变量
        self.feature_names = self.X.columns.tolist()  # 获取特征名称
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=self.random_state  # 按 7:3 划分训练集和测试集
        )
        print(f"数据读取完毕。特征数量: {self.X.shape[1]}")
        
    def calculate_shap(self):
        """计算 SHAP 值及交互作用矩阵"""
        print("正在计算 SHAP 值 (交互作用计算较耗时)...")
        preprocessor = Pipeline(self.pipeline.steps[:-1])  # 提取预处理流水线
        X_train_transformed = preprocessor.transform(self.X_train)  # 预处理训练集
        self.X_train_processed = pd.DataFrame(X_train_transformed, columns=self.feature_names)  # 转换为 DataFrame
        
        self.explainer = shap.TreeExplainer(self.model)  # 初始化 SHAP TreeExplainer
        
        # ========= 增加 check_additivity=False =============
        self.shap_values_obj = self.explainer(self.X_train_processed, check_additivity=False)  # 计算 SHAP Explanation 对象
        # ==================================================
        
        self.shap_interaction_values = self.explainer.shap_interaction_values(self.X_train_processed)  # 计算交互作用矩阵
        print("SHAP 计算完成。")

    def plot_figure_1(self):
        """绘制模型预测散点图与残差图"""
        print("正在绘制图1：模型评估组合图...")
        y_train_pred = self.pipeline.predict(self.X_train)  # 获取训练集预测值
        y_test_pred = self.pipeline.predict(self.X_test)  # 获取测试集预测值
        
        r2_train = r2_score(self.y_train, y_train_pred)  # 计算训练集 R2
        rmse_train = np.sqrt(mean_squared_error(self.y_train, y_train_pred))  # 计算训练集 RMSE
        r2_test = r2_score(self.y_test, y_test_pred)  # 计算测试集 R2
        rmse_test = np.sqrt(mean_squared_error(self.y_test, y_test_pred))  # 计算测试集 RMSE
        
        mae_train = mean_absolute_error(self.y_train, y_train_pred)  # 计算训练集 MAE
        mae_test = mean_absolute_error(self.y_test, y_test_pred)  # 计算测试集 MAE
        res_train = y_train_pred - self.y_train  # 计算训练集残差
        res_test = y_test_pred - self.y_test  # 计算测试集残差

        fig = plt.figure(figsize=(8, 10))  # 创建画布
        gs = gridspec.GridSpec(3, 2, width_ratios=[4, 1], height_ratios=[1, 4, 1.5], wspace=0.05, hspace=0.05)  # 定义子图网格布局
        
        ax_main = fig.add_subplot(gs[1, 0])  # 添加主散点图轴
        ax_main.scatter(self.y_train, y_train_pred, color=CONFIG["fig1_train_color"],   # 绘制训练集散点
                        s=CONFIG["fig1_scatter_s"], edgecolor=CONFIG["fig1_scatter_edgecolor"],   
                        linewidth=CONFIG["fig1_scatter_linewidth"], label="Train data", alpha=0.8, zorder=2)
        ax_main.scatter(self.y_test, y_test_pred, color=CONFIG["fig1_test_color"],   # 绘制测试集散点
                        s=CONFIG["fig1_scatter_s"], edgecolor=CONFIG["fig1_scatter_edgecolor"],   
                        linewidth=CONFIG["fig1_scatter_linewidth"], label="Test data", alpha=0.8, zorder=2)
        
        xlims = ax_main.get_xlim()  # 获取当前 X 轴范围
        ylims = ax_main.get_ylim()  # 获取当前 Y 轴范围
        
        ax_main.plot(xlims, ylims, 'k--', zorder=5)  # 绘制理想预测对角线
        m, b = np.polyfit(self.y_test, y_test_pred, 1)  # 拟合测试集线性回归方程
        y_fit_start, y_fit_end = m * xlims[0] + b, m * xlims[1] + b  # 计算拟合线端点坐标
        ax_main.plot(xlims, [y_fit_start, y_fit_end], color='black', linewidth=2, label="Fitted line", zorder=5)  # 绘制拟合直线
        
        ax_main.set_xlim(xlims)  # 固定 X 轴范围
        ax_main.set_ylim(ylims)  # 固定 Y 轴范围
        
        ax_main.text(0.5, 0.96, "XGBoost", transform=ax_main.transAxes, ha='center', va='top',   # 标注模型名称
                     fontsize=CONFIG["title_fontsize"]+4, fontweight='bold', zorder=10)
        
        metrics_text = f"Train $R^2$ = {r2_train:.2f}, $RMSE$ = {rmse_train:.2f}\nTest $R^2$ = {r2_test:.2f}, $RMSE$ = {rmse_test:.2f}"  # 组合评估指标文本
        ax_main.text(0.05, 0.85, metrics_text, transform=ax_main.transAxes, va='top', ha='left',  # 添加评估指标文本框
                     fontsize=CONFIG["title_fontsize"]-2, color="darkred",   
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3), zorder=10)
                     
        ax_main.legend(loc="lower right", frameon=False, fontsize=CONFIG["label_fontsize"])  # 显示图例
        ax_main.grid(True, linestyle='--', alpha=0.5, color='lightgray')  # 开启背景网格

        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)  # 添加顶部边缘分布子图
        sns.histplot(self.y_train, color=CONFIG["fig1_train_color"], bins=CONFIG["fig1_hist_bins"],   # 绘制训练集直方图
                     ax=ax_top, alpha=0.5, stat="density", element="bars", edgecolor="white", linewidth=0.3)
        sns.histplot(self.y_test, color=CONFIG["fig1_test_color"], bins=CONFIG["fig1_hist_bins"],   # 绘制测试集直方图
                     ax=ax_top, alpha=0.5, stat="density", element="bars", edgecolor="white", linewidth=0.3)
        sns.kdeplot(self.y_train, color=CONFIG["fig1_train_color"], ax=ax_top,   # 绘制训练集核密度曲线
                    linewidth=CONFIG["fig1_kde_linewidth"], zorder=10, cut=0)
        sns.kdeplot(self.y_test, color=CONFIG["fig1_test_color"], ax=ax_top,   # 绘制测试集核密度曲线
                    linewidth=CONFIG["fig1_kde_linewidth"], zorder=10, cut=0)
        ax_top.axis('off')  # 隐藏坐标轴边框

        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)  # 添加侧边边缘分布子图
        sns.histplot(y=y_train_pred, color=CONFIG["fig1_train_color"], bins=CONFIG["fig1_hist_bins"],   # 绘制训练集垂直直方图
                     ax=ax_right, alpha=0.5, stat="density", element="bars", edgecolor="white", linewidth=0.3)
        sns.histplot(y=y_test_pred, color=CONFIG["fig1_test_color"], bins=CONFIG["fig1_hist_bins"],   # 绘制测试集垂直直方图
                     ax=ax_right, alpha=0.5, stat="density", element="bars", edgecolor="white", linewidth=0.3)
        sns.kdeplot(y=y_train_pred, color=CONFIG["fig1_train_color"], ax=ax_right,   # 绘制训练集垂直核密度曲线
                    linewidth=CONFIG["fig1_kde_linewidth"], zorder=10, cut=0)
        sns.kdeplot(y=y_test_pred, color=CONFIG["fig1_test_color"], ax=ax_right,   # 绘制测试集垂直核密度曲线
                    linewidth=CONFIG["fig1_kde_linewidth"], zorder=10, cut=0)
        ax_right.axis('off')  # 隐藏坐标轴边框

        ax_res = fig.add_subplot(gs[2, 0], sharex=ax_main)  # 添加底侧残差子图
        ax_res.scatter(self.y_train, res_train, color=CONFIG["fig1_train_color"],   # 绘制训练集残差散点
                       s=CONFIG["fig1_scatter_s"], edgecolor=CONFIG["fig1_scatter_edgecolor"], linewidth=CONFIG["fig1_scatter_linewidth"], alpha=0.8)
        ax_res.scatter(self.y_test, res_test, color=CONFIG["fig1_test_color"],   # 绘制测试集残差散点
                       s=CONFIG["fig1_scatter_s"], edgecolor=CONFIG["fig1_scatter_edgecolor"], linewidth=CONFIG["fig1_scatter_linewidth"], alpha=0.8)
        ax_res.plot(xlims, [0, 0], color='black', linewidth=1.5, zorder=5)  # 绘制残差基准线 (y=0)
        ax_res.grid(True, linestyle='--', alpha=0.5, color='lightgray')  # 开启背景网格
        
        res_text = f"MAE (Train) = {mae_train:.3f}\nMAE (Test) = {mae_test:.3f}"  # 组合 MAE 文本
        ax_res.text(0.95, 0.95, res_text, transform=ax_res.transAxes, fontsize=CONFIG["label_fontsize"],   # 添加 MAE 文本标签
                    verticalalignment='top', horizontalalignment='right',   
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3), zorder=6)
        
        save_figure(fig, "Fig1_Prediction_Residuals")  # 保存图像文件

    def plot_figure_2(self):
        """绘制全局 SHAP 特征重要性条形图与蜂群图"""
        print("正在绘制图2：全局特征贡献度分析图...")
        fig, ax1 = plt.subplots(figsize=(10, 8))  # 创建画布
        
        mean_abs_shap = np.abs(self.shap_values_obj.values).mean(axis=0)  # 计算各个特征的平均绝对 SHAP 值
        sort_inds = np.argsort(mean_abs_shap)  # 获取升序排列的索引
        sorted_features = [self.feature_names[i] for i in sort_inds]  # 按升序排列特征名称
        sorted_mean_shap = mean_abs_shap[sort_inds]  # 按升序排列平均 SHAP 值
        total_shap_sum = np.sum(mean_abs_shap)  # 计算 SHAP 总值
        
        shap_vals = self.shap_values_obj.values[:, sort_inds]  # 重排 SHAP 值矩阵列顺序
        feat_vals = self.X_train_processed.values[:, sort_inds]  # 重排特征数值矩阵列顺序
        y_pos = np.arange(len(sorted_features))  # 生成纵坐标序列
        
        ax2 = ax1.twiny()  # 添加共享 Y 轴的副图
        ax2.barh(y_pos, sorted_mean_shap, color=CONFIG["fig2_bar_color"], align='center', alpha=0.8, height=0.6, zorder=2)  # 在副图绘制背景条形图
        
        cmap = CONFIG["shap_cmap"]  # 获取颜色映射表
        for i in range(len(sorted_features)):  # 遍历特征绘制蜂群图
            row_shap = shap_vals[:, i]  # 当前特征的 SHAP 值
            row_feat = feat_vals[:, i]  # 当前特征的原始数值
            feat_min, feat_max = np.min(row_feat), np.max(row_feat)  # 计算极值
            row_feat_norm = (row_feat - feat_min) / (feat_max - feat_min) if feat_max > feat_min else np.zeros_like(row_feat)  # Min-Max 归一化特征值用于颜色映射
                
            jitter = np.random.normal(0, 0.1, size=len(row_shap))  # 生成 Y 轴高斯噪声以展示密度
            scatter = ax1.scatter(row_shap, np.repeat(i, len(row_shap)) + jitter,   # 绘制散点
                                  c=row_feat_norm, cmap=cmap, s=CONFIG["fig2_scatter_s"], alpha=0.8, edgecolors='none', zorder=4)

        max_mean_val = np.max(sorted_mean_shap)  # 获取最大均值
        ax2.set_xlim(0, max_mean_val * 1.2)  # 扩展副图 X 轴范围
        
        for i, v in enumerate(sorted_mean_shap):  # 遍历添加百分比文本标签
            pct = (v / total_shap_sum) * 100  # 计算占比
            offset = max_mean_val * 0.01  # 计算标签偏移量
            ax1.text(v + offset, i, f"{pct:.1f}%", va='center', ha='left', fontsize=CONFIG["label_fontsize"],   
                     transform=ax2.transData, zorder=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

        ax1.set_zorder(ax2.get_zorder() + 1)  # 调整散点图层级至顶部
        ax1.patch.set_visible(False)  # 使主图背景透明
        
        ax1.set_yticks(y_pos)  # 配置 Y 轴刻度
        ax1.set_yticklabels(sorted_features, fontsize=CONFIG["tick_fontsize"])  # 配置特征名称标签
        ax1.set_xlabel("SHAP value (impact on model output)", fontsize=CONFIG["label_fontsize"])  # 设置主 X 轴标签
        ax2.set_xlabel("Mean Absolute SHAP Value", fontsize=CONFIG["label_fontsize"])  # 设置副 X 轴标签
        ax1.grid(True, axis='x', linestyle='--', alpha=0.4)  # 开启垂直参考线
        
        divider = make_axes_locatable(ax1)  # 调用坐标轴分割工具
        cax = divider.append_axes("right", size="3%", pad=0.1)  # 添加色带区域
        
        sm_map = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))  # 生成颜色映射标量
        sm_map.set_array([])  # 空载数据
        cbar = fig.colorbar(sm_map, cax=cax)  # 添加色带
        cbar.set_ticks([0, 1])  # 指定刻度位置
        cbar.set_ticklabels(['Low', 'High'])  # 指定刻度文本
        cbar.set_label('Feature value', rotation=270, labelpad=15, fontsize=CONFIG["label_fontsize"])  # 设置色带标签
        
        save_figure(fig, "Fig2_Global_Contribution")  # 保存图像文件

    def plot_figure_3(self):
        """批量绘制特征的偏依赖散点图与平滑曲线 (依据截图识别 Positive/Negative 象限)"""
        print("正在绘制图3：单特征偏依赖图 (自动按截距划分 Positive/Negative 区间)...")
        mean_abs_shap = np.abs(self.shap_values_obj.values).mean(axis=0)  # 获取特征重要性
        sort_inds_desc = np.argsort(mean_abs_shap)[::-1]  # 获取降序排列索引
        folder_name = "Fig3_Dependence_Plots"  # 设定输出子文件夹
        
        for rank, feat_idx in enumerate(sort_inds_desc):  # 遍历降序特征
            feature_name = self.feature_names[feat_idx]  # 获取目标特征名称
            fig, ax = plt.subplots(figsize=(6, 5))  # 创建子画布
            
            x_vals = self.X_train_processed[feature_name].values  # 提取横坐标数值
            y_vals = self.shap_values_obj.values[:, feat_idx]  # 提取纵坐标数值 (SHAP 值)
            
            color_feat_idx = sort_inds_desc[1] if rank == 0 else sort_inds_desc[0]  # 获取用于着色的对比特征索引
            color_feat_name = self.feature_names[color_feat_idx]  # 获取对比特征名称
            c_vals = self.X_train_processed[color_feat_name].values  # 提取对比特征数值
            
            scatter = ax.scatter(x_vals, y_vals, c=c_vals, cmap=CONFIG["shap_cmap"],   # 绘制散点图
                                 s=CONFIG["fig3_scatter_s"], alpha=0.9, edgecolors='none', zorder=4)
            
            sorted_indices = np.argsort(x_vals)  # 获取按 X 升序排列的索引
            x_sorted = x_vals[sorted_indices]  # 重排 X 数组
            y_sorted = y_vals[sorted_indices]  # 重排 Y 数组
            
            lowess = sm.nonparametric.lowess(y_sorted, x_sorted, frac=0.3)  # 拟合局部加权回归散点平滑曲线 (Lowess)
            ax.plot(lowess[:, 0], lowess[:, 1], color=CONFIG["fig3_curve_color"],   # 绘制拟合曲线
                    linewidth=CONFIG["fig3_curve_linewidth"], alpha=0.9, label="Lowess curve", zorder=5)
            
            window_size = max(5, int(len(x_sorted) * 0.1))  # 动态计算窗口大小
            rolling_std = pd.Series(y_sorted - lowess[:, 1]).rolling(window=window_size, min_periods=1, center=True).std().values  # 计算局部标准差
            ax.fill_between(lowess[:, 0], lowess[:, 1] - rolling_std, lowess[:, 1] + rolling_std,   # 填充置信度区间
                            color=CONFIG["fig3_curve_color"], alpha=0.15, edgecolor='none', linewidth=0, zorder=2)
            
            # =============== 完全参考截图识别 Positive/Negative 阈值和填充方式 ===============
            sign_changes = np.diff(np.sign(lowess[:, 1]))  # 计算一阶差分获取交叉零点
            zero_crossings = np.where(sign_changes != 0)[0]  
            
            if len(zero_crossings) > 0:
                zc_idx = zero_crossings[0] # 获取第一个与 SHAP=0 相交的位置
                x_cross = lowess[zc_idx, 0]
                
                # 记录原始的作图范围以防 fill_between 拉大画布
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()
                
                # SHAP > 0 (右上象限 - Positive)
                ax.fill_between([x_cross, x_max], 0, y_max, color=CONFIG["fig3_positive_color"], alpha=0.5, zorder=1, label="Positive")
                # SHAP < 0 (左下象限 - Negative)
                ax.fill_between([x_min, x_cross], y_min, 0, color=CONFIG["fig3_negative_color"], alpha=0.5, zorder=1, label="Negative")
                
                # 在阈值处绘制贯通上下的垂直红色虚线
                ax.axvline(x=x_cross, color='#D32F2F', linestyle='--', linewidth=1.2, alpha=0.8, zorder=3)
                
                # 在交叉点处画中心红点
                ax.scatter(x_cross, 0, color='#D32F2F', s=25, zorder=6)
                
                # 阈值标签加上白色透明背景，不要描边
                ax.text(x_cross, 0.05 * (y_max - y_min), f"{x_cross:.2f}", va='bottom', ha='center', 
                        fontsize=CONFIG["title_fontsize"], color='#D32F2F', 
                        bbox=dict(facecolor=(1, 1, 1, 0.7), edgecolor='none', pad=2), zorder=6)
                
                # 锁死坐标系，避免色块破坏图表比例
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
            # =========================================================================

            ax.set_xlabel("Feature value", fontsize=CONFIG["label_fontsize"])  # 设定 X 轴标签
            ax.set_ylabel("SHAP", fontsize=CONFIG["label_fontsize"])  # 设定 Y 轴标签
            ax.set_title(feature_name, fontsize=CONFIG["title_fontsize"])  # 设定子图标题
            ax.grid(True, linestyle='--', alpha=0.3)  # 开启背景网格线
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5, zorder=2)  # 绘制 Y=0 横向参考线
            
            # 图例自动识别四周空白区域放置(loc='best')，背景不要描边
            ax.legend(loc='best', fontsize=CONFIG["label_fontsize"] - 2, framealpha=0, edgecolor='none') 
            
            divider = make_axes_locatable(ax)  # 调用坐标轴分割器
            cax = divider.append_axes("right", size="5%", pad=0.0)  # 添加侧边色带绘制区域
            cbar = fig.colorbar(scatter, cax=cax)  # 添加色带
            cbar.ax.tick_params(labelsize=CONFIG["fig3_cbar_fontsize"])  # 配置色带字体
            # 已根据要求删除 Y 轴右侧的标签：cbar.set_label(color_feat_name, rotation=270, labelpad=15, fontsize=CONFIG["fig3_cbar_fontsize"]) 
            
            clean_feature_name = re.sub(r'[^\w]', '', feature_name) # 清除特征名称中的特殊符号用于文件命名
            filename = f"{rank+1:02d}_{clean_feature_name}_Dependence"  # 生成统一编号且剥除特殊字符的文件名
            save_figure(fig, filename, subfolder=folder_name)  # 批量保存图像

    def plot_figure_4(self):
        """绘制主效应与交互效应强度对比图"""
        print("正在绘制图4：主效应与交互效应对比图...")
        num_features = len(self.feature_names)  # 获取特征数量
        main_effects = np.zeros(num_features)  # 初始化主效应数组
        inter_effects = np.zeros(num_features)  # 初始化交互效应数组
        
        for i in range(num_features):  # 遍历计算效应量
            main_effects[i] = np.abs(self.shap_interaction_values[:, i, i]).mean()  # 提取对角线平均绝对主效应
            mask = np.ones(num_features, dtype=bool)  # 构建筛选掩码
            mask[i] = False  # 屏蔽主对角线
            inter_effects[i] = np.abs(self.shap_interaction_values[:, i, mask]).sum(axis=1).mean()  # 聚合平均非自身交互效应
            
        total_effects = main_effects + inter_effects  # 聚合计算总体效应
        sort_inds = np.argsort(total_effects)[::-1]  # 生成总体效应降序索引
        
        top_names = [self.feature_names[i] for i in sort_inds]  # 提取重排的特征名
        top_main = main_effects[sort_inds]  # 提取重排的主效应序列
        top_inter = inter_effects[sort_inds]  # 提取重排的交互效应序列
        
        fig_width = max(10, num_features * 0.8)  # 动态计算画板宽度以适应特征数量
        fig, ax = plt.subplots(figsize=(fig_width, 6))  # 实例化自适应画布
        x = np.arange(len(top_names))  # 定义底座间隔坐标阵列
        width = 0.4  # 设置柱体宽度
        
        rects1 = ax.bar(x - width/2, top_main, width, label='Main effect (Mean |SHAP|)', color=CONFIG["fig4_main_color"])  # 绘制主效应柱体
        rects2 = ax.bar(x + width/2, top_inter, width, label='Interaction (sum over others)', color=CONFIG["fig4_inter_color"])  # 绘制交互效应柱体
        
        max_height = max(np.max(top_main), np.max(top_inter))  # 获取最大高度用于动态设定 Y 轴
        # 增加Y轴顶部留白，Y轴从0开始
        ax.set_ylim(0, max_height * 1.15)  
        
        overlap_threshold = max_height * 0.05  # 判定拥挤的纵向高度差阈值

        for i in range(len(top_names)):
            h1 = top_main[i]
            h2 = top_inter[i]
            
            # 保留3位小数的标签
            text1 = f'{h1:.3f}'  
            text2 = f'{h2:.3f}'
            
            # 默认两个标签都在各自的柱子上方，不设负偏移量
            offset_1, offset_2 = 2, 2  
            va_1, va_2 = 'bottom', 'bottom'
            
            # 若高度接近，则将较高的柱子标签进一步上移，较低的正常放置，以此避免打架且绝不越界
            if abs(h1 - h2) < overlap_threshold:
                if h1 >= h2:
                    offset_1 = 12     # 较高的柱子标签上移
                    offset_2 = 1      # 较低的贴近柱顶
                else:
                    offset_1 = 1    
                    offset_2 = 12      

            # 写入数值标签
            ax.annotate(text1, xy=(x[i] - width/2, h1), xytext=(0, offset_1), textcoords="offset points", ha='center', va=va_1, fontsize=9)
            ax.annotate(text2, xy=(x[i] + width/2, h2), xytext=(0, offset_2), textcoords="offset points", ha='center', va=va_2, fontsize=9)
        
        ax.set_ylabel('Magnitude (Mean |SHAP|)', fontsize=CONFIG["label_fontsize"])  # 配置纵坐标标签
        ax.set_title('All Features: Main vs Interaction', fontsize=CONFIG["title_fontsize"])  # 配置图表主标题
        ax.set_xticks(x)  # 设置 X 轴刻度位置
        
        # 将X轴标签旋转90度垂直显示
        ax.set_xticklabels(top_names, fontsize=CONFIG["tick_fontsize"], rotation=90)  
        
        ax.legend(fontsize=CONFIG["label_fontsize"], frameon=False)  # 显示无框图例
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)  # 开启水平辅助线
        
        save_figure(fig, "Fig4_Main_vs_Interaction_All")  # 保存图像文件

    def plot_figure_5(self):
        """绘制对角线特征交互复合矩阵图 (统一命名为图5)"""
        print("正在绘制图5：特征交互对角矩阵 (左下:热力值, 右上:分布散点)...")
        mean_abs_shap = np.abs(self.shap_values_obj.values).mean(axis=0)
        
        # 控制展示的特征数量，依赖于类初始化的全局配置项
        num_top = min(CONFIG["fig5_max_features"], len(self.feature_names))
        top_inds = np.argsort(mean_abs_shap)[::-1][:num_top]
        
        # 提取最大交互作用均值，用于左下角热力图背景色映射统一标尺
        inter_vals = []
        for r in top_inds:
            for c in top_inds:
                if r != c: # 排除主对角线的自身作用
                    inter_vals.append(np.abs(self.shap_interaction_values[:, r, c]).mean())
        max_inter = max(inter_vals) if inter_vals and max(inter_vals) > 0 else 1
        
        fig, axes = plt.subplots(num_top, num_top, figsize=(14, 14))
        # 调整画布边界，给四周的标签及顶部的刻度留出足够的排版空间
        plt.subplots_adjust(wspace=0.08, hspace=0.08, bottom=0.12, left=0.12, top=0.9)
        
        for row, feat_r_idx in enumerate(top_inds):
            for col, feat_c_idx in enumerate(top_inds):
                ax = axes[row, col]
                feat_row_name = self.feature_names[feat_r_idx]
                feat_col_name = self.feature_names[feat_c_idx]
                
                # 若行号大于列号，则位于左下三角 (包含热力图)
                if row > col:
                    # ====================
                    # 绘制左下三角：热力图
                    # ====================
                    ax.set_xticks([])  # 清除内侧格子刻度
                    ax.set_yticks([])
                    inter_val = np.abs(self.shap_interaction_values[:, feat_r_idx, feat_c_idx]).mean()
                    
                    # 绝对值色彩映射：将其归一化到 colormap 的 0.5 到 1.0 区间 (从灰色渐变为红色)
                    norm_val = 0.5 + 0.5 * (inter_val / max_inter)
                    bg_color = CONFIG["shap_cmap"](norm_val)
                    ax.set_facecolor(bg_color)  # 填涂单元格背景
                    
                    text_color = 'white' if norm_val > 0.85 else 'black'  # 动态字体颜色，避免红底黑字看不清
                    ax.text(0.5, 0.5, f"{inter_val:.3f}", ha='center', va='center',
                            fontweight='bold', fontsize=CONFIG["tick_fontsize"], color=text_color)
                
                # 若行号小于等于列号，则位于右上三角及对角线 (包含散点分布)
                else:
                    # ====================
                    # 绘制右上三角及对角线：分布散点图
                    # ====================
                    ax.set_facecolor('white')  # 散点图背景修改为纯白色
                    ax.grid(True, color='lightgray', linestyle='-', linewidth=0.8)  # 开启辅助网格线
                    
                    if row == col:  # 主对角线展示特征的主效应散点分布
                        x_vals = self.shap_interaction_values[:, feat_c_idx, feat_c_idx]
                    else:  # 非对角线展示特征对的双边总交互效应分布
                        x_vals = self.shap_interaction_values[:, feat_r_idx, feat_c_idx] * 2
                        
                    y_jitter = np.random.normal(0, 0.1, size=len(x_vals))
                    c_vals = self.X_train_processed[feat_row_name].values
                    
                    # 为了在统一色带下展示不同特征，采用局部 Min-Max 归一化映射色彩
                    c_min, c_max = c_vals.min(), c_vals.max()
                    c_norm = (c_vals - c_min) / (c_max - c_min + 1e-8)
                    
                    ax.scatter(x_vals, y_jitter, c=c_norm, cmap=CONFIG["shap_cmap"], 
                               s=8, alpha=0.7, edgecolors='none', vmin=0, vmax=1)
                    
                    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')  # 零基准线
                    ax.set_yticks([])
                    
                    # =======================================================
                    # 统一将散点图的X轴刻度(标签)置于最顶端(第一行)，并平行X轴放置
                    # =======================================================
                    if row == 0:
                        ax.xaxis.tick_top()  # 将 X 轴刻度移到图表上方
                        ax.tick_params(axis='x', labelsize=CONFIG["fig5_tick_fontsize"], pad=2)
                        ax.locator_params(axis='x', nbins=3)  # 将最大刻度数限制为3，确保空间充足
                        # 刻度文本平行于X轴放置 (水平，rotation=0)，居中对齐
                        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
                    else:
                        ax.set_xticks([])

                # 统一渲染外边框，防止散点图与热力图边框粗细割裂
                for spine in ax.spines.values():
                    spine.set_linewidth(0.8)
                    spine.set_color('#333333')

                # =========================================================
                # == 外部边缘标签定位：标签分别垂直于XY轴放置 ==
                # X轴(水平方向)的垂线为垂直方向(竖排)，故 rotation=90
                # Y轴(垂直方向)的垂线为水平方向(横排)，故 rotation=0
                # =========================================================
                if row == num_top - 1:  # 放置在矩阵整体的最底端一行
                    ax.set_xlabel(feat_col_name, fontsize=CONFIG["fig5_label_fontsize"], rotation=90,
                                  ha='center', va='top', labelpad=10)
                    
                if col == 0:  # 放置在矩阵整体的最左侧一列
                    ax.set_ylabel(feat_row_name, fontsize=CONFIG["fig5_label_fontsize"], rotation=0,
                                  ha='right', va='center', labelpad=15)

        # 整体正下方说明标题
        fig.text(0.5, 0.02, "SHAP interaction value", ha='center', va='center', fontsize=CONFIG["title_fontsize"])

        # 生成右侧侧边的全局图例颜色带指示器
        cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
        sm_map = plt.cm.ScalarMappable(cmap=CONFIG["shap_cmap"], norm=plt.Normalize(vmin=0, vmax=1))
        sm_map.set_array([])
        cbar = fig.colorbar(sm_map, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=CONFIG["fig5_cbar_fontsize"])
        
        # 设置共用标签，体现兼收并蓄：散点代表Raw Feature(低到高)，热力图代表Interaction(灰到红)
        cbar.set_label('Raw feature value / Mean |Interaction|', rotation=270, labelpad=20, fontsize=CONFIG["fig5_cbar_fontsize"])
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Low', 'High'])

        # 统一输出命名为图5
        save_figure(fig, "Fig5_Combined_Matrix")

    def plot_figure_6(self):
        """批量绘制成对特征交互作用散点图 (增加SHAP=0基准及白色透明阈值标签)"""
        print("正在绘制图6：成对交互作用散点图 (稍耗时)...")
        num_features = len(self.feature_names)  # 获取特征数量
        folder_name = "Fig6_Pairwise_Interactions"  # 指定输出子文件夹
        
        plot_idx = 1  # 初始化图像保存序号
        for i in range(num_features):  # 外层循环遍历主特征
            for j in range(num_features):  # 内层循环遍历交互特征
                if i >= j: continue  # 跳过下三角和主对角线以避免重复绘制
                
                feat_i_name = self.feature_names[i]  # 提取主特征名
                feat_j_name = self.feature_names[j]  # 提取对比特征名
                
                x_vals = self.X_train_processed[feat_i_name].values  # 获取主特征原始数据用于 X 轴
                y_vals = self.shap_interaction_values[:, i, j] * 2  # 提取交互值并乘 2 还原总作用量作为 Y 轴
                c_vals = self.X_train_processed[feat_j_name].values  # 获取伴随特征原始数据用于颜色映射
                
                fig, ax = plt.subplots(figsize=(6, 5))  # 创建单独画布
                scatter = ax.scatter(x_vals, y_vals, c=c_vals, cmap=CONFIG["shap_cmap"],   # 绘制交互散点分布图
                                     s=30, alpha=0.8, edgecolors='none', zorder=2)  # 设定散点透明度和无描边属性
                
                # =============== 绘制 SHAP=0 灰色虚线和带标签的橙色阈值虚线 ===============
                ax.axhline(0, color='gray', linestyle='--', alpha=0.5, zorder=1) # 增加 SHAP=0 灰色基准线
                
                # 使用 Lowess 拟合交互效应趋势并寻找交叉零点阈值
                sorted_indices = np.argsort(x_vals)
                x_sorted = x_vals[sorted_indices]
                y_sorted = y_vals[sorted_indices]
                lowess_inter = sm.nonparametric.lowess(y_sorted, x_sorted, frac=0.3)
                
                sign_changes = np.diff(np.sign(lowess_inter[:, 1]))
                zero_crossings = np.where(sign_changes != 0)[0]
                for zc in zero_crossings:
                    threshold_val = lowess_inter[zc, 0]
                    ax.axvline(x=threshold_val, color='orange', linestyle='--', alpha=0.9, zorder=3) # 增加橙色阈值虚线
                    
                    # 增加带有白色透明背景无描边的阈值文本标签
                    ax.text(threshold_val, np.median(y_vals), f"{threshold_val:.2f}", rotation=90,
                            va='center', ha='right', fontsize=CONFIG["tick_fontsize"],
                            bbox=dict(facecolor=(1, 1, 1, 0.7), edgecolor='none', pad=2), zorder=4)
                # ================================================================

                ax.set_xlabel(feat_i_name, fontsize=CONFIG["label_fontsize"])  # 设置 X 轴标签为特征名
                ax.set_ylabel("SHAP Interaction Value", fontsize=CONFIG["label_fontsize"])  # 设置 Y 轴标签说明
                ax.set_title(f"{feat_i_name} × {feat_j_name}", fontsize=CONFIG["title_fontsize"])  # 设置组合特征为图表标题
                ax.grid(True, linestyle='--', alpha=0.3)  # 开启虚线参考网格
                
                divider = make_axes_locatable(ax)  # 分割坐标轴空间
                cax = divider.append_axes("right", size="5%", pad=0.0)  # 在边缘追加颜色带预留空间
                cbar = fig.colorbar(scatter, cax=cax)  # 渲染附带色带指示器
                cbar.set_label(feat_j_name, rotation=270, labelpad=15)  # 标注色带对应的特征名并旋转方向
                
                clean_i_name = re.sub(r'[^\w]', '', feat_i_name) # 命名时删除特征1中的特殊符号
                clean_j_name = re.sub(r'[^\w]', '', feat_j_name) # 命名时删除特征2中的特殊符号
                filename = f"{plot_idx:02d}_{clean_i_name}_cross_{clean_j_name}"  # 构建剥离特殊符号的保存文件名称
                save_figure(fig, filename, subfolder=folder_name)  # 保存独立散点图
                plot_idx += 1  # 递增序号
    def run_all(self):
        """执行完整的数据处理、模型训练及绘图流程"""
        self.prepare_data()  # 前置数据清洗
        self.train_xgboost(self.X_train, self.y_train)  # 训练 XGBoost 模型并调参
        self.calculate_shap()  # 计算 SHAP 值与交互作用矩阵
        
        self.plot_figure_1()  # 模型预测与残差图
        self.plot_figure_2()  # 全局特征贡献与蜂群图
        self.plot_figure_3()  # 单特征偏依赖散点图
        self.plot_figure_4()  # 主效应与交互效应对比图
        self.plot_figure_5()  # 对角线特征交互复合矩阵图
        self.plot_figure_6()  # 成对交互作用散点图
        
        
        print(f"全部分析与绘图流程执行完毕。图表已保存至：{CONFIG['output_dir']}")  

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":  
    analyzer = XGBoostXAIAnalyzer()  
    analyzer.run_all()  
    print("程序运行结束。")
