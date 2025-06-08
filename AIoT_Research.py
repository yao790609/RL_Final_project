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


# train = pd.concat([xls_df_2013,xls_df_2014,xls_df_2015,xls_df_2016,xls_df_2017,xls_df_2018,xls_df_2019], ignore_index=False) #,,xls_df_2015,xls_df_2016,xls_df_2017,xls_df_2018,xls_df_2019,xls_df_2020
# validation = pd.concat([xls_df_2020,xls_df_2021], ignore_index=False)
# test_df = pd.concat([xls_df_2022,xls_df_2023,xls_df_2024], ignore_index=False)

# train = train.set_index(train.columns[0])
# validation = validation.set_index(validation.columns[0])
# test_df = test_df.set_index(test_df.columns[0])
# train.index.names = ['']
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







# stock_dimension = len(test_df.tic.unique())
# state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
# buy_cost_list = sell_cost_list = [0.004425] * stock_dimension
# num_stock_shares = [0] * stock_dimension

# env_kwargs = {
#     "hmax": 5,
#     "initial_amount": 1000000,
#     "num_stock_shares": num_stock_shares,
#     "buy_cost_pct": buy_cost_list,
#     "sell_cost_pct": sell_cost_list,
#     "state_space": state_space,
#     "stock_dim": stock_dimension,
#     "tech_indicator_list": INDICATORS,
#     "action_space": stock_dimension,
#     "reward_scaling": 1
# }

# # 建立測試環境
# test_env = StockTradingEnv(df=test_df, **env_kwargs)
# env_test, _ = test_env.get_sb_env()

# agent = DRLAgent(env=env_test)

# # 設定 SAC 參數（需要與訓練時相同）
# SAC_PARAMS = {
#     "batch_size": 256,
#     "buffer_size": 150000,
#     "learning_rate": 0.0001,
#     "learning_starts": 30000,
#     "ent_coef": "auto_0.2"
# }

# # 獲取模型並載入訓練好的權重
# model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)
# model_path = TRAINED_MODEL_DIR + "/agent_sac_20241229"  # 或使用 "/best_sac/best_model" 如果要用最佳模型
# loaded_model = model_sac.load(model_path)

# # 進行測試
# state = env_test.reset()
# done = False
# rewards = []
# actions_list = []

# step_count = 0
# while not done :
#     step_count += 1
#     if step_count % 1000 == 0:
#         print(f"已處理 {step_count} 步")

#     action, _states = loaded_model.predict(state, deterministic=False)
#     next_state, reward, done, info = env_test.step(action)
    
#     rewards.append(reward)
#     actions_list.append(action)
#     state = next_state
    
#     # 如果是最後一步，確保保存剩餘的交易日誌
#     if done:
#         # 保存剩餘的交易日誌
#         if len(test_env.logger.trade_log) > 0:
#             test_env.logger.trade_log.to_excel(test_env.logger.log_filename)
#             print(f"交易日誌已保存到: {test_env.logger.log_filename}")

#         stats = test_env.logger.get_statistics()
        
#         print("\n進階交易統計：")
#         print(f"每日平均交易量: {stats['每日平均交易量']:.2f} 張")
#         print(f"平均持倉時間: {stats['平均持倉時間(小時)']:.2f} 小時")
#         print(f"交易勝率: {stats['勝率']*100:.2f}%")
#         print(f"單筆最大獲利: ${stats['單筆最大獲利']:.2f}")
#         print(f"單筆最大虧損: ${stats['單筆最大虧損']:.2f}")
#         print(f"總交易成本: ${stats['總交易成本']:.2f}")
#         print(f"平均交易成本: ${stats['平均交易成本']:.2f}")
        
# # 從保存的交易日誌文件中讀取數據
# log_filename = test_env.logger.log_filename
# trade_log_df = pd.read_excel(log_filename)
# portfolio_values = trade_log_df['總資產價值'].tolist()
# initial_amount = env_kwargs["initial_amount"]

# # ... [後面的分析和繪圖代碼保持不變] ...

# # 輸出交易統計
# print("\n交易統計：")
# print(f"總交易筆數: {len(trade_log_df)}")
# print(f"買入交易筆數: {len(trade_log_df[trade_log_df['交易類型'] == '買入'])}")
# print(f"賣出交易筆數: {len(trade_log_df[trade_log_df['交易類型'] == '賣出'])}")

# # 顯示每支股票的交易統計
# print("\n各股票交易統計：")
# stock_stats = trade_log_df.groupby('股票名稱').agg({
#     '交易數量': 'sum',
#     '交易金額': 'sum',
#     '交易類型': 'count'
# }).rename(columns={'交易類型': '交易次數'})
# print(stock_stats)








# # 訓練 TD3 agent
# if if_using_td3:
#     TD3_PARAMS = {"batch_size": 256, "buffer_size": 500000, "learning_rate": 0.001}
#     model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)
#     # model_td3.policy.to("cuda")
#     tmp_path = RESULTS_DIR + '/td3'
#     new_logger_td3 = configure(tmp_path, ["stdout", "csv", "tensorboard"])
#     model_td3.set_logger(new_logger_td3) 
#     print(f"Training TD3: Model is running on {model_td3.policy.device}")
#     trained_td3 = agent.train_model(model=model_td3, tb_log_name='td3', total_timesteps=200000)
#     trained_td3.save(TRAINED_MODEL_DIR + "/agent_td3")

# # 訓練 PPO agent
# if if_using_ppo:
#     PPO_PARAMS = {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128}
#     model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
#     # model_ppo.policy.to("cuda")
#     tmp_path = RESULTS_DIR + '/ppo'
#     new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
#     model_ppo.set_logger(new_logger_ppo)
#     print(f"Training PPO: Model is running on {model_ppo.policy.device}")
#     trained_ppo = agent.train_model(model=model_ppo, tb_log_name='ppo', total_timesteps=1000000)
#     trained_ppo.save(TRAINED_MODEL_DIR + "/agent_ppo")
    
# # 訓練 A2C agent
# if if_using_a2c:
#     model_a2c = agent.get_model("a2c")
#     # model_a2c.policy.to("cuda")
#     tmp_path = RESULTS_DIR + '/a2c'
#     new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
#     model_a2c.set_logger(new_logger_a2c)
#     print(f"Training A2C: Model is running on {model_a2c.policy.device}")
#     trained_a2c = agent.train_model(model=model_a2c, tb_log_name='a2c', total_timesteps=50000)
#     trained_a2c.save(TRAINED_MODEL_DIR + "/agent_a2c")
#     print(TRAINED_MODEL_DIR + "/agent_a2c")

# # 訓練 DDPG agent
# if if_using_ddpg:
#     model_ddpg = agent.get_model("ddpg")
#     # model_ddpg.policy.to("cuda")
#     tmp_path = RESULTS_DIR + '/ddpg'
#     new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
#     model_ddpg.set_logger(new_logger_ddpg)
#     print(f"Training DDPG: Model is running on {model_ddpg.policy.device}")
#     trained_ddpg = agent.train_model(model=model_ddpg, tb_log_name='ddpg', total_timesteps=50000)
#     trained_ddpg.save(TRAINED_MODEL_DIR + "/agent_ddpg")
