# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:37:20 2024

@author: User
"""
# # 安裝必要的套件
# !pip install swig wrds pyportfolioopt
# !pip install git+https://github.com/AI4Finance-Foundation/FinRL.git

# 匯入必要的套件
# import os
import pandas as pd
from finrl.meta.custom_env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
# from finrl import config_tickers
from finrl.main import check_and_make_directories
from finrl.config import TRAINED_MODEL_DIR, RESULTS_DIR
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
# import numpy as np
# import matplotlib.pyplot as plt

# def get_exponential_lr_schedule(initial_lr=0.001, end_lr=0.0001, decay_steps=1000000):
#     decay_rate = (end_lr / initial_lr) ** (1 / decay_steps)
    
#     def schedule(progress_remaining):
#         current_step = (1 - progress_remaining) * decay_steps
#         return initial_lr * (decay_rate ** current_step)
    
#     return schedule

# 建立必要的目錄
check_and_make_directories([TRAINED_MODEL_DIR])

# 載入並處理訓練資料

train = pd.read_csv("C:\\Users\\User\\Desktop\\2024_SMC\\上市資料整合_訓練集.csv",engine='python', encoding = "cp950",index_col=0)
validation = pd.read_csv("C:\\Users\\User\\Desktop\\2024_SMC\\上市資料整合_驗證集.csv",engine='python', encoding = "cp950",index_col=0)

INDICATORS = ["close", "short_term_high", "short_term_low", "short_bull_fvg", "short_bull_fvg_bottom", "short_bull_fvg_top", "short_bear_fvg", "short_bear_fvg_bottom", "short_bear_fvg_top", "mid_bull_fvg", "mid_bull_fvg_top", "mid_bull_fvg_bottom", "mid_bear_fvg", "mid_bear_fvg_top", "mid_bear_fvg_bottom", "mid_term_high", "mid_term_low", "fvg_center", "dist_to_nearest_fvg_center", "dist_to_nearest_fvg_bottom", "dist_to_nearest_fvg_top", "is_in_fvg_range", "dist_to_nearest_high", "dist_to_nearest_low"]

# 設定環境參數
stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
buy_cost_list = sell_cost_list = [0.004425] * stock_dimension
num_stock_shares = [0] * stock_dimension
env_kwargs = {
    "hmax": 5,
    "initial_amount": 10000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 0.001#1e-4
}

# 建立訓練環境
e_train_gym = StockTradingEnv(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()

# 建立評估環境（使用部分訓練數據作為驗證集）
e_eval_gym = StockTradingEnv(df=validation, **env_kwargs)  # 使用最後20%的訓練數據作為驗證集
env_eval, _ = e_eval_gym.get_sb_env()

# 初始化 DRL agent
agent = DRLAgent(env=env_train)

# 設定使用的演算法（A2C, DDPG, PPO, TD3, SAC）
# if_using_a2c = True
# if_using_ddpg = True
# if_using_ppo = True
# if_using_td3 = True
if_using_sac = True

# 訓練 SAC agent
if if_using_sac:

    # 設置參數
   # initial_lr = 0.001
   # end_lr = 0.0001
   # decay_steps = 1200000  # 總訓練步數    

   # lr_schedule = get_exponential_lr_schedule( initial_lr=initial_lr, end_lr=end_lr, decay_steps=decay_steps)    
    
    # initial_lr, end_lr = 0.001, 0.0001
   
   # 創建線性學習率函數
   # lr_schedule = get_linear_fn(start=initial_lr, end=end_lr, end_fraction=1.0)  # 在整個訓練過程中完成衰減

   SAC_PARAMS = {"batch_size": 256, "buffer_size": 120000, "learning_rate": 0.000015,"train_freq": 4,"tau": 0.002, "learning_starts": 25000, "ent_coef": 0.01,"gradient_steps": 1} #
   model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)
   # model_sac = model_sac.load(r"C:\Users\User\Desktop\AIoT_Research\trained_models\agent_sac")
   tmp_path = RESULTS_DIR + '/sac'
   new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
   model_sac.set_logger(new_logger_sac)
   
    # 設定 Early Stopping 參數
   stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20,  min_evals=30, verbose=1)
    
    # 設定評估回調
   eval_callback = EvalCallback(
        env_eval,
        best_model_save_path=TRAINED_MODEL_DIR + "/best_sac",  # 儲存最佳模型的路徑
        log_path=RESULTS_DIR + "/eval_results",               # 評估結果的日誌路徑
        eval_freq=7000,                                      # 每10000步評估一次
        n_eval_episodes=20,                                   # 每次評估進行10個回合
        deterministic=True,                                   # 評估時使用確定性動作
        callback_after_eval=stop_train_callback,              # 加入 early stopping
        verbose=1
    )
    
   print(f"Training SAC: Model is running on {model_sac.policy.device}")
   # trained_sac = agent.train_model(model=model_sac, tb_log_name='sac', total_timesteps=1600000,callback=eval_callback)
   model_sac = model_sac.learn(total_timesteps=800000, callback=eval_callback, tb_log_name='sac')
   # trained_sac.save(TRAINED_MODEL_DIR + "/agent_sac")
   model_sac.save(TRAINED_MODEL_DIR + "/agent_sac")
