# LongTermHiResLST
> Long-Term High-Resolution Land Surface Temperature Reconstruction & Analysis Platform  
> 长期高分辨率地表温度重建与分析平台

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

## Table of Contents
- [Overview](#overview)
- [Highlights](#highlights)
- [Repository Layout](#repository-layout)
- [Data Prerequisites](#data-prerequisites)
- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [Workflow](#workflow)
- [Validation & Diagnostics](#validation--diagnostics)
- [Visualization & Reporting](#visualization--reporting)
- [Outputs](#outputs)
- [Troubleshooting & Tips](#troubleshooting--tips)
- [Roadmap](#roadmap)
- [License](#license)

## Overview
LongTermHiResLST（Long-Term High-Resolution Land Surface Temperature）融合 MODIS、Landsat、NDVI 和土地利用类型数据，通过随机森林等机器学习模型重建 30 m LST，并提供年度趋势、气象对比与多维可视化分析。

```mermaid
flowchart TD
  A[MOD11A1 HDF] -->|Tool1.process_hdf_files| B[LST 1 km TIFF]
  B -->|Reproject + Crop| C[LST 990 m]
  D[Landsat B4/B5/B10] -->|Tool1.process_all_landsat| E[NDVI & LST 30 m]
  F[Landcover] -->|Tool1.crop_type_data| G[Type 30 m]
  H[Landsat QA] -->|Tool1.preprocessing_QA_folder| I[QA 30 m]
  C & E & G -- Tool2.load_data_usable --> J[Training Samples]
  J --> K[ML Models (RF/XGB/CatBoost/Transformer)]
  K --> L[FullYearPre / FullYearPro]
  L --> M[Diagnostics & Visualization]
```

## Highlights
- Multi-source preprocessing toolkit (`LST_Tools/Tool1.py`) covering HDF extraction, reprojection, resampling, QA masking, and feature stacking.
- Pluggable ML stack (`Training/`) with classic tree ensembles plus a PyTorch Transformer super-resolution prototype.
- Monthly QA-aware training/prediction loop for long-term mosaics (`Training/Training_Usable.py`, `Preprocessing/FullYearPre.py`).
- Validation suite: in situ QA filtering, Meteostat weather comparison, land-cover specific error analysis, cloud coverage studies.
- Publication-ready visualization scripts (temperature mosaics, scatter fits, cloud–LST joint plots, land-cover boxplots).

## Repository Layout
```
LongTermHiResLST/
├── LST_Tools/
│   ├── Tool1.py            # Raster utilities, HDF handling, QA processing, metrics plotting
│   ├── Tool2.py            # Path helpers, dataset loaders, QA filtering and splitting
├── Preprocessing/
│   ├── ModisPreprocessing.py
│   ├── LandsatPreprocessing.py
│   ├── TypeProcessing.py
│   ├── CloudCoverageProcessing.py
│   ├── FullYearPro.py      # Full-year MODIS tiling, reprojection, resampling
│   ├── FullYearPre.py      # Monthly prediction orchestrator
│   ├── ValidatlyProcessing/ # Validation-specific scripts
│   ├── ...others (CorrelationAnalysis, LCErrorAnalysis, YearTemAnalysis, etc.)
├── Training/
│   ├── RandomForest.py / Training_Usable.py / CatBoost.py / XGBoost.py / Transformer.py
├── Visualization/
│   ├── Visualization1.py / Visualization2.py / temperature_image.png
├── LICENSE
├── README.md (this file)
```

## Data Prerequisites
| Dataset | Source | Native Resolution | Expected Location (default) | Notes |
| --- | --- | --- | --- | --- |
| MODIS MOD11A1 LST + QC | NASA LP DAAC | 1 km | `F:/MyProjects/MachineLearning/Data/Raw_Data/MOD11A1/` | Daily tiles; Tool1 extracts and mosaics. |
| Landsat 8 B4/B5/B10 | USGS | 30 m | `.../Raw_Data/LandsatB4B5` & `.../LandsatB10` | Used for NDVI and reference LST. |
| Land-use / LCD30 | e.g., GlobeLand30 | 30 m | `.../Raw_Data/Type2022` | Cropped to AOI. |
| Landsat QA | Landsat QA band | 30 m | `.../Raw_Data/LandsatQA` | Used to mask clouds. |
| Optional meteorological data | Meteostat API | Daily | fetched on demand | Needed for `YearTemAnalysis.py`. |

> 所有路径在代码中默认为 Windows 盘符；如使用其他环境，请先调整配置。

## Environment Setup
1. **Python**: 3.9+ (GDAL wheels are most stable on CPython/conda).  
2. **Core deps**: `numpy`, `pandas`, `rasterio`, `gdal`, `scikit-learn`, `seaborn`, `matplotlib`, `joblib`, `meteostat`, `plotly`, `cupy` (optional GPU KDE), `torch` (for Transformer).
3. Example conda env:
   ```bash
   conda create -n lst python=3.10 gdal rasterio
   conda activate lst
   pip install numpy pandas scikit-learn seaborn matplotlib joblib meteostat plotly cupy-cuda12x torch torchvision torchaudio
   ```
4. Install GDAL-compatible Microsoft Visual C++ redistributables on Windows if needed.

## Configuration
- Update `LST_Tools/Tool2.py` (`path_set`, `predict_path_set`) to match your folder layout.
- In preprocessing scripts (e.g., `ModisPreprocessing.py`, `LandsatPreprocessing.py`), adjust input/output dirs, `utm_coords`, and `size` to your AOI.
- `FullYearPro.py` assumes tile naming conventions like `LSTYYYYDDD`. Confirm before batch runs.
- GPU-only routines (Transformer, `cupy` KDE) require CUDA-capable hardware; guard them if running on CPU.

## Workflow

### 1. Preprocess Base Layers
Run the scripts in the order below (modify paths first):

```bash
python Preprocessing/ModisPreprocessing.py        # MODIS LST extraction → 990 m → 30 m
python Preprocessing/LandsatPreprocessing.py      # NDVI & 30 m reference LST
python Preprocessing/TypeProcessing.py            # Land-cover clip
python Preprocessing/CloudCoverageProcessing.py   # QA masks
```

Key operations handled inside `LST_Tools/Tool1.py`:
- HDF extraction (`process_hdf_files`, `extract_tiff`)
- Radiometric scaling & reprojection (`process_modis_lst_data`, `reproject_folder`)
- Resampling/cropping (`crop_raster`, `resample_lst_to_30m`, `crop_type_data`)
- QA normalization (`preprocessing_QA_folder`)

### 2. Train Models
Choose a modeling script:

| Script | Description |
| --- | --- |
| `Training/RandomForest.py` | Single RF experiment on stacked NDVI/LST/type features. |
| `Training/Training_Usable.py` | Month-wise RF training with QA masking + automatic prediction. |
| `Training/CatBoost.py`, `Training/XGBoost.py`, `Training/CatBoost.py` | Alternative ensemble baselines. |
| `Training/Transformer.py` | PyTorch super-resolution prototype (requires GPU). |

Typical run:
```bash
python Training/Training_Usable.py
```
Models are saved to `Data/Final_Data/Models/<year>/random_forest_model_MM.joblib`.

### 3. Long-Term Reconstruction
1. **Prepare monthly stacks**:
   ```bash
   python Preprocessing/FullYearPro.py
   ```
   - Extracts MODIS LST/QC from full-year HDF
   - Reprojects, crops, rescales to 30 m
   - Groups files per month (`TempData/LSTmouth`).

2. **Predict per month**:
   ```bash
   python Preprocessing/FullYearPre.py
   ```
   - Loads `<month>` model, stacks NDVI + monthly MODIS + land-cover
   - Applies QA masks, fills predicted rasters in `FullYearPre/FinalData`.

### 4. Validation & Diagnostics

| Script | Purpose |
| --- | --- |
| `Preprocessing/ValidatlyProcessing/Validate.py` | Applies a trained model to validation tiles and reports R² vs. Landsat truth. |
| `Preprocessing/CorrelationAnalysis.py` | Computes NDVI/LST/type correlation heatmaps (Plotly). |
| `Preprocessing/LCErrorAnalysis.py` | Land-cover-specific error boxplots. |
| `Preprocessing/YearTemAnalysis.py` | Compares reconstructed LST with Meteostat daily averages, includes regression scatterplots. |
| `Preprocessing/CloudCoverageProcessing.py` + `coveranalys.py` | QA/cloud coverage statistics vs. LST trends. |
| `Preprocessing/StabilityAnaly.py`, `StatisticalAnalysis.py` | Additional metrics and time-series summaries.

### 5. Visualization & Reporting
Use the scripts under `Visualization/` for publication-ready figures:

- `Visualization/Visualization1.py`: multi-panel maps, scatter fit with R², land-use summaries.
- `Visualization/Visualization2.py`: land-cover histogram/percentages.
- Generated assets (e.g., `temperature_image.png`) are stored alongside scripts or inside `Data/Final_Data/Pictures`.

## Validation & Diagnostics
- **QA Filtering**: `Tool2.filter_data_based_on_qa` removes cloudy pixels before model training/prediction.
- **Statistical Metrics**: `Tool1.model_accuracy`, `Tool1.plot_mae/mse`, GPU-accelerated KDE for residual density.
- **External Validation**: `YearTemAnalysis.py` aligns LST with station temperature; linear regression results (slope, R²) exported as figures.
- **Cloud Impact**: `coveranalys.py` visualizes how cloud coverage affects available samples and predicted LST per month.

## Outputs
- `Data/Final_Data/Models/<year>/` — serialized ML models.
- `Data/Final_Data/Pictures/` — performance plots, cloud coverage charts, scatterplots.
- `Data/Final_Data/Predict_Data/` & `FullYearPre/FinalData/` — GeoTIFF predictions (`Predict_LST_LSTYYYYMMDD.tif` or `Predict_LST_MM.tif`).
- Validation rasters reside in `Validation_Data/Usable`.

Document key metadata (CRS, resolution, QA thresholds) when sharing outputs to ensure reproducibility.

## Troubleshooting & Tips
- **Large Raster IO**: Enable GDAL VSI caching or process month-by-month to avoid memory spikes.
- **Path Lengths**: Windows paths can exceed 260 chars—enable long path support or shorten base directories.
- **GPU Optionality**: Guard `cupy`/CUDA imports if deploying on CPU-only machines.
- **Missing Tiles**: `FullYearPro.py` logs when MODIS granules are absent; inspect `TempData` before proceeding.
- **Data Consistency**: NDVI, MODIS, land-cover, and QA rasters must share projection and pixel grid after preprocessing.

## Roadmap
1. Parameterize file paths via `.env` or YAML instead of hard-coded strings.
2. Wrap workflows into CLI commands (e.g., `python -m lst.cli preprocess`).
3. Add unit tests for preprocessing helpers and synthetic QA masks.
4. Extend Transformer training to mini-batches of tiles with on-the-fly augmentation.

## License
This project is released under the [MIT License](./LICENSE).
