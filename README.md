# Spectral-Aware Precipitation Nowcasting with Multi-Bias Fourier Neural Operators and Local Convolutional Attention

This repository contains the full codebase for the paper:

> **Spectral-Aware Precipitation Nowcasting with Multi-Bias Fourier Neural Operators and Local Convolutional Attention**  

The project implements:

1. **Our proposed model**  
   - Multi-Scale Encoder + Multi-Bias Fourier Neural Operator architecture (m-AFNO)

2. **Baseline models** for spatiotemporal radar forecasting:  
   - ConvLSTM  
   - PredRNN  
   - SimVP  
   - TAU  
   - Earthformer  
   - PastNet  
   - EarthFarseer  
   - AFNO  

---

## 📂 Repository Structure

```

├── config/                  # YAML config files for each model
│   ├── cikm/                # Config files for the CIKM dataset
│   ├── meteonet/            # Config files for the MeteoNet dataset
│   └── sevir/               # Config files for the SEVIR dataset
├── data_index/              # Dataset indexing
├── evaluation/              # Code for model evaluation and metric computation
├── model/                   # Model implementations
├── module/                  # Core building blocks used across different models
├── util/                    # Utility functions
└── README.md                # This file

````

---


## 📥 Dataset

We use **CIKM** and **SEVIR-LR** dataset for training and evaluation:

1. **CIKM dataset**:
   * **Download**:
      Visit [https://drive.google.com/drive/folders/1IqQyI8hTtsBbrZRRht3Es9eES_S4Qv2Y](https://drive.google.com/drive/folders/1IqQyI8hTtsBbrZRRht3Es9eES_S4Qv2Y)
   
   * **Directory layout**:
      Download and extract into `data/CIKM/` so that you have:
       ```
       data/CIKM/
       ├── train/
       ├── val/
       └── test/
       ```

2. **MeteoNet dataset**:
   * **Download**:
      Visit [https://meteonet.umr-cnrm.fr/dataset/data/NW/radar/reflectivity_old_product/](https://meteonet.umr-cnrm.fr/dataset/data/NW/radar/reflectivity_old_product/)
   
   * **Directory layout**:
      Download and extract into `data/meteonet/` so that you have:
       ```
       data/meteonet/
       ├── 2016/
       ├── 2017/
       └── 2018/
       ```
     
3. **SEVIR-LR dataset**:
   * **Download**:
      Visit [https://deep-earth.s3.amazonaws.com/datasets/sevir_lr.zip](https://deep-earth.s3.amazonaws.com/datasets/sevir_lr.zip)
   
   * **Processing**:
      We provide two helper scripts to convert the raw HDF5 file into NumPy arrays and to split out individual precipitation events
        ```
        # 1) Convert the raw .h5 file to .npy array
             python process_sevir.py
        
        # 2) Split each precipitation event into a single .npy file
             python save_sevir.py
        ```
   * **Directory layout after processing**:
     ```
     data/SEVIR/data/vil_single/
     ├── random/
     └── storm/
     ```

---

## 🏃‍ Quick Start

### Train a model

1. **For CIKM dataset**:
    ```
    python train_cikm.py \
      --model m_afno \
      --batchsize 16 \
      --epoch 80 \
      --lr 1e-3 \
      --gpus 0
    ```

2. **For SEVIR-LR dataset**:
    ```
    python train_sevir.py \
      --model m_afno \
      --batchsize 16 \
      --epoch 100 \
      --lr 1e-3 \
      --gpus 0
    ```
   
3. **For MeteoNet dataset**:
    ```
    to be done
    ```
