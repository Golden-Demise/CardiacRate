# 環境重建指南

專案內從未提交過 `requirements.txt` 或 `environment.yml`,git 歷史查無版本鎖定紀錄。以下 `requirements.txt` 是由現有程式碼的 import 反推整理,版本為建議值,非原始鎖定版本。

## 1. 安裝 Python 3.10.7

先確認電腦上是否已裝 3.10.7:

```
py -0
```

若沒有,到 https://www.python.org/downloads/release/python-3107/ 下載安裝(勾選 Add to PATH)。

## 2. 建立 venv

在專案根目錄(`D:\CardiacRate`)執行,依 readme.md 原本的慣例把 venv 建在專案內、命名為 `CardiacRate`(此名稱已被 `.gitignore` 排除):

```
cd D:\CardiacRate
py -3.10 -m venv CardiacRate
CardiacRate\Scripts\activate.bat
```

啟動後,提示字元前應出現 `(CardiacRate)`。

## 3. 確認 CUDA 版本(GPU 訓練必要)

```
nvidia-smi
```

畫面右上角 `CUDA Version` 是你顯卡驅動支援的**最高**版本,不是要裝的版本。依此對照下表選擇 torch 安裝指令。

## 4. 安裝 torch(先裝,依 CUDA 版本挑一行)

```
# CUDA 12.4 及以上驅動,較新顯卡建議
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8(較舊顯卡 / 驅動)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 純 CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

不確定就先裝 cu121,再用下一步驗證。

## 5. 安裝其餘套件

```
pip install -r requirements.txt
```

## 6. 驗證安裝

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` 應回傳 `True`(GPU 版)。

## 附註

- `monailabel` 官方主要支援到 Python 3.9/3.10,若安裝失敗可能需要降級或改用官方指定版本,屆時再依錯誤訊息調整。
- `acdc_random_forest.py`、`Segmentation/inferer.py`、`Segmentation/networks/`、`make_facts.py` 等是專案內部模組,不在 requirements.txt 內,無需另外安裝。
- 若之後又找到舊環境的 `pip freeze` 記錄或安裝紀錄,可用它取代這份反推版本,以確保與訓練/推論結果完全一致。
