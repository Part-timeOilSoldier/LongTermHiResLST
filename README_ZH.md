# LongTermHiResLST

[English](README.md) | 中文

![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)
![RasterIO](https://img.shields.io/badge/rasterio-%F0%9F%8C%90-green)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Status](https://img.shields.io/badge/status-active-success)

🛰️ 高分辨率地表温度（LST）预测流水线：融合 MODIS 与 Landsat 产品，通过机器学习增强空间细节，并输出可直接用于发表的图表与指标。

---

## 🛰️ 项目总览
- **目标**：利用 NDVI、土地利用类型与逐月回归模型，将 990 m MODIS 日 LST 提升到 30 m 分辨率，并保持全年连续性。
- **范围**：涵盖卫星影像预处理、QA 云掩膜、逐月随机森林训练/预测、CatBoost/XGBoost/Transformer 等扩展实验、验证及可视化。
- **成果**：生成 30 m 月度预测栅格、时序统计、精度评估、土地利用误差剖析，以及流程图、12 个月预报面板和分辨率对比图等可视化作品。

![Processing Flow](Visualization/flowchart.png)

## 📦 仓库结构
| 目录 | 作用 |
| --- | --- |
| `Preprocessing/` | MODIS/Landsat 数据导入、裁剪、重采样、QA 过滤、统计与绘图脚本，如 `FullYearPro.py`、`LandsatPreprocessing.py`。 |
| `Preprocessing/ValidatlyProcessing/` | 面向独立区域的验证脚本（`Validate.py`、`DataProcessing.py`）。 |
| `LST_Tools/` | 通用工具库：栅格 IO、投影、NDVI、QA 预测、精度图、KDE 缓存等。 |
| `Training/` | 模型训练入口（随机森林、CatBoost、XGBoost、Transformer 超分原型）。 |
| `Visualization/` | 高层可视化脚本及 README 引用的图片。 |
| `LICENSE` | Apache-2.0 许可证及安徽农业大学专利声明。 |

## 🔁 端到端流程
1. **采集与整理**：下载 MODIS HDF、Landsat B4/B5/B10 及土地利用数据，放入 `F:/MyProjects/MachineLearning/Data`。
2. **预处理**：运行 `Preprocessing/LandsatPreprocessing.py`、`ModisPreprocessing.py`、`FullYearPro.py`，获得 NDVI/LST 栈、QA 掩膜与逐月切片。
3. **训练模型**：使用 `Training/Training_Usable.py` 训练 12 个随机森林模型，或改用 `CatBoost.py`、`XGBoost.py`、`Transformer.py`。
4. **预测与验证**：通过 `Preprocessing/Predicting.py`、`Preprocessing/ValidatlyProcessing/Validate.py` 生成月度预测并在独立区域计算 R²。
5. **分析与可视化**：借助 `StatisticalAnalysis.py`、`StabilityAnaly.py`、`Visualization1.py` 等脚本制作 KPI 报告。

## 🧰 核心模块
- **`LST_Tools/Tool1.py`**：千行级工具集合，负责 HDF 提取、投影/拼接、裁剪、重采样、QA 处理（`preprocessing_QA_folder`）、NDVI 计算、预测写出（`predict_and_save`）、精度绘图（`model_accuracy`、`plot_mae/mse`）、KDE 存储等。
- **`LST_Tools/Tool2.py`**：封装路径配置、QA 过滤、数据加载与训练/推理拆分，确保脚本间使用一致的数据切片。
- **预处理脚本**：如 `FullYearPro.py`（全年 MODIS 处理与按月分类）、`StabilityAnaly.py`（云量 vs. R²）、`StatisticalAnalysis.py`（每日云量与均温）、`To_shp.py`（矢量边界）、`YearTemAnalysis.py`（气象站 vs. LST 对比）以及众多绘图模板（`Drawing*.py`、`Printing*.py`、`test*.py`）。
- **训练脚本**：`Training/Training_Usable.py` 提供 QA 掩膜后的月度随机森林训练；`CatBoost.py`、`XGBoost.py`、`Transformer.py` 用于 GPU/深度学习实验。
- **可视化脚本**：`Visualization` 目录负责生成分辨率对比、土地利用统计、相关热力图等出版级图件。

## 🚀 快速开始
### 依赖
- Python 3.11+、GDAL 3.x、rasterio、numpy、pandas、matplotlib/seaborn/plotly、shapely、scikit-learn、catboost、xgboost、cupy(CUDA)、torch（Transformer）、meteostat、Pillow。
- 先安装系统级 GDAL/PROJ，再 `pip install rasterio`。
- XGBoost/Transformer 依赖 GPU，请确保 CUDA 与 PyTorch 版本匹配。

### 环境配置
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt  # 可自行整理依赖
```

### 数据路径
- 代码默认使用 `F:/MyProjects/MachineLearning/Data/...` 目录，可在 `LST_Tools/Tool1.py`、`Tool2.py` 中修改或参数化。
- QA 栅格需命名为 `_QA_Usable.tif` 才能被过滤逻辑识别。

## 🧪 流水线运行
| 阶段 | 命令 | 说明 |
| --- | --- | --- |
| Landsat 预处理 | `python Preprocessing/LandsatPreprocessing.py` | 分组 B4/B5、计算 NDVI、裁剪 AOI、生成 30 m LST。 |
| MODIS 预处理 | `python Preprocessing/ModisPreprocessing.py` | 提取 MODIS LST、拼接、重投影、裁剪到 30 m 代理。 |
| 全年 MODIS | `python Preprocessing/FullYearPro.py` | 自动完成全年导入→拼接→重投影→重采样→逐月分类。 |
| 特征构建 | `python Preprocessing/FullYearPre.py` | 汇总 NDVI/MODIS/TYPE/QA，输出预测输入栅格。 |
| 月度训练 | `python Training/Training_Usable.py` | 训练/加载 12 个随机森林模型并生成月度预测。 |
| 其他模型 | `python Training/CatBoost.py` 等 | 可替换为 CatBoost、XGBoost、Transformer。 |
| 批量预测 | `python Preprocessing/Predicting.py` | 使用已有模型重新推理，自动跳过已存在输出。 |
| 区域验证 | `python Preprocessing/ValidatlyProcessing/Validate.py` | 在江苏区域运行预测并打印 R²。 |
| 可视化 | `python Visualization/Visualization1.py` 等 | 生成对比面板、散点审计、统计图。 |

### 月度预测长图
![12-month forecast results](Visualization/12monthforecastresults.png)

12 张拼图展示全年 LST 演变，便于在 `Training/Training_Usable.py` 运行完成后快速检查季节特征。

## ✅ 质量与验证
- **QA 掩膜**：`Tool1.predict_and_save`、`Tool2.filter_data_based_on_qa` 保证训练/预测过程只使用 QA=1 的像元。
- **精度指标**：`Tool1.model_accuracy`、`Tool1.print_model_R2`、`Preprocessing/StabilityAnaly.py`、`LCErrorAnalysis.py` 输出 MAE/MSE/R² 与土地利用误差箱线图。
- **气象对比**：`YearTemAnalysis.py` 结合 Meteostat 日均气温与预测 LST 绘制折线+散点，附带回归 R²。
- **密度诊断**：`get_density_data`、`simple_gpu_kde` 等函数缓存 KDE 结果，加速多模型绘图。

### 分辨率增强示例
![Resolution enhancement showcase](Visualization/Imageshowingimprovedresolution.png)

该图展示训练后的模型如何将 990 m MODIS 输入上采样为 30 m 细节，从视觉上验证 QA 过滤与回归流程的价值。

## ⚙️ 配置提示
- 在 `LST_Tools/Tool2.py` 中自定义 `predict_path_set`、`path_set` 以适配本地 NDVI/LST/Type 目录。
- 迁移至 Linux 时，需替换绝对路径并确保 GDAL 可以找到 PROJ 数据。
- 每次新增 Landsat 影像后运行 `Tool1.group_landsat_files` 保持文件夹结构一致。
- `Preprocessing/` 下的 `test*.py` 是可复用的绘图模板，按需复制修改即可。

## 📅 后续工作
1. 使用 `.env` 或 YAML 参数化硬编码路径。
2. 补充 `requirements.txt`/`environment.yml`，方便复现环境。
3. 将预处理和训练脚本整合为统一 CLI（例如 `python -m pipelines.run all`）。
4. 扩展验证区域并在 README 中附上汇总表。

## 🛡️ 许可证
- 代码遵循 Apache License 2.0，并受 `LICENSE` 中的安徽农业大学专利声明约束。商业再分发需获得书面许可。
- 复用图件或模型时请引用 `LICENSE` 中的署名说明。

---

🧭 **快速体验**：按顺序运行预处理脚本、执行 `Training/Training_Usable.py`，再打开 `Visualization/12monthforecastresults.png`，即可验证整条流水线。
