# FinRL 台股強化學習改版實作

本專案基於 [FinRL](https://github.com/AI4Finance-Foundation/FinRL) 框架，針對台灣股票市場進行客製化調整與策略訓練。整合 Smart Money Concept (SMC) 指標、延伸自定義環境 `env_stocktrading.py`，並使用 Soft Actor-Critic (SAC) 強化學習演算法進行訓練與回測。

## 📁 檔案說明

| 檔案名稱 | 說明 |
|----------|------|
| `AIoT_Research.py` | 主程式，負責資料載入、環境建構、訓練模型、驗證與儲存。主要使用 SAC 演算法。 |
| `env_stocktrading.py` | 自定義強化學習交易環境，擴展自 FinRL 的 `StockTradingEnv`，加入 SMC 技術指標支援與策略評估擴充。|

## 📈 特色功能

- ✅ 支援 Smart Money Concept 指標集成  
- ✅ 使用台股上市公司資料進行訓練與驗證  
- ✅ 加入 Early Stopping 與 EvalCallback 模組，強化訓練穩定性  
- ✅ 自定義 reward scaling、交易成本與技術指標  
- ✅ 儲存並匯出訓練後模型、評估日誌與交易紀錄  

## 🏗 使用說明

### 1. 安裝套件

```bash
pip install swig wrds pyportfolioopt
pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
```

### 2. 準備資料

將以下兩個 CSV 檔放入對應目錄，並依照主程式中路徑修改：

- `上市資料整合_訓練集.csv`
- `上市資料整合_驗證集.csv`

這兩份資料需包含欄位：

- `tic`: 股票代號  
- `date`: 日期（格式為 `YYYY-MM-DD`）  
- `close`: 收盤價  
- 以及其他你定義的技術指標欄位（見下方 `INDICATORS`）

### 3. 執行訓練

```bash
python AIoT_Research.py
```

訓練過程將產出：

- 儲存模型：`trained_models/agent_sac`
- 評估日誌：`results/eval_results/`
- TensorBoard 記錄檔案

## 🔧 自定義參數

在 `AIoT_Research.py` 中可調整以下訓練參數：

```python
SAC_PARAMS = {
    "batch_size": 256,
    "buffer_size": 120000,
    "learning_rate": 0.000015,
    "train_freq": 4,
    "tau": 0.002,
    "learning_starts": 25000,
    "ent_coef": 0.01,
    "gradient_steps": 1
}
```

## 📊 技術指標一覽

在 `env_kwargs` 中自定義以下 Smart Money Concept 相關欄位：

```python
INDICATORS = [
    "close", "short_term_high", "short_term_low",
    "short_bull_fvg", "short_bear_fvg", ...
]
```

## 🔍 Todo / 待辦事項

- [ ] 加入測試集分析與績效視覺化  
- [ ] 輸出交易紀錄分析報表  
- [ ] 整合 Backtesting 模組  

## 🧠 參考資源

- [FinRL 官方 GitHub](https://github.com/AI4Finance-Foundation/FinRL)  
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)

---

> 本專案由 [使用者](https://github.com/你的帳號) 自行修改 FinRL 框架以符合台股環境，歡迎參考、改進與交流。
