from __future__ import annotations

import math

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime
import os

matplotlib.use("Agg")

class TradingLogger:
    def __init__(self, log_dir="trading_logs"):
        """
        初始化交易記錄器
        """
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.trade_log = pd.DataFrame(columns=['時間','股票名稱','交易類型','交易價格','交易數量','交易金額','持倉數量','當前資金','總資產價值','SAC動作值'])        
        
        # 統計數據
        self.daily_volume = {}  # 每日交易量
        self.position_times = []  # 持倉時間記錄
        self.profit_loss = []  # 獲利/虧損記錄
        self.trade_costs = []  # 交易成本
        self.current_positions = {}  # 當前持倉 {股票: [進場時間, 價格]}

        self.log_filename = os.path.join(log_dir,f"trade_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

    def log_trade(self, timestamp, stock_code, action_type, price, quantity, holdings, cash_balance, total_value, sac_action):
        """
        記錄單次交易信息
        """
        trade_record = {'時間': timestamp,'股票名稱': stock_code,'交易類型': '買入' if sac_action > 0 else '賣出','交易價格': price,'交易數量': quantity,'交易金額': abs(price * quantity * 1000),'持倉數量': holdings,'當前資金': cash_balance,'總資產價值': total_value,'SAC動作值': sac_action }

        try:
            date = pd.to_datetime(timestamp).date()
            self.daily_volume[date] = self.daily_volume.get(date, 0) + abs(quantity)
            
            if action_type == 'buy':
                self.current_positions[stock_code] = [pd.to_datetime(timestamp), price]
            elif action_type == 'sell' and stock_code in self.current_positions:
                entry_time = self.current_positions[stock_code][0]
                entry_price = self.current_positions[stock_code][1]
                hold_time = (pd.to_datetime(timestamp) - entry_time).total_seconds() / 3600
                self.position_times.append(hold_time)
                pl = (price - entry_price) * quantity * 1000
                self.profit_loss.append(pl)
                del self.current_positions[stock_code]
            
            self.trade_costs.append(price * quantity * 1000 * 0.001425)
        except Exception as e:
            print(f"統計數據更新錯誤: {e}")

        self.trade_log = pd.concat([self.trade_log, pd.DataFrame([trade_record])], ignore_index=True)  
        if len(self.trade_log) >= 100000 :
            self.trade_log.to_excel(self.log_filename)
            self.trade_log = pd.DataFrame(columns=['時間','股票名稱','交易類型','交易價格','交易數量','交易金額','持倉數量','當前資金','總資產價值','SAC動作值']) 
            self.log_filename = os.path.join("trading_logs",f"trade_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

    def _parse_timestamp(self, timestamp):
        if isinstance(timestamp, str):
            return pd.to_datetime(timestamp).date()
        return timestamp.date()
    
    def _update_daily_volume(self, timestamp, quantity):
        date = self._parse_timestamp(timestamp)
        self.daily_volume[date] = self.daily_volume.get(date, 0) + quantity
    
    def _update_position_tracking(self, timestamp, stock_code, is_buy, price, quantity):
        if is_buy:
            self.current_positions[stock_code] = [timestamp, price]
        else:
            if stock_code in self.current_positions:
                entry_time = self.current_positions[stock_code][0]
                entry_price = self.current_positions[stock_code][1]
                
                # 記錄持倉時間
                hold_time = (timestamp - entry_time).total_seconds() / 3600  # 轉換為小時
                self.position_times.append(hold_time)
                
                # 記錄獲利/虧損
                pl = (price - entry_price) * quantity * 1000
                self.profit_loss.append(pl)
                
                del self.current_positions[stock_code]
    
    def _update_trade_costs(self, price, quantity):
        # 假設交易成本為 0.1425% (手續費 0.1425%)
        cost = price * quantity * 1000 * 0.001425
        self.trade_costs.append(cost)
    
    def get_statistics(self):
        stats = {
            '每日平均交易量': np.mean(list(self.daily_volume.values())) if self.daily_volume else 0,
            '平均持倉時間(小時)': np.mean(self.position_times) if self.position_times else 0,
            '勝率': len([pl for pl in self.profit_loss if pl > 0]) / len(self.profit_loss) if self.profit_loss else 0,
            '單筆最大獲利': max(self.profit_loss) if self.profit_loss else 0,
            '單筆最大虧損': min(self.profit_loss) if self.profit_loss else 0,
            '總交易成本': sum(self.trade_costs),
            '平均交易成本': np.mean(self.trade_costs) if self.trade_costs else 0
        }
        return stats
    
class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
    ):
        # 初始化交易記錄器
        self.logger = TradingLogger()
        
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = gym.spaces.Box(
            low=-self.hmax,  # 最多賣出的手數
            high=self.hmax,  # 最多買入的手數
            shape=(self.stock_dim,),
            dtype=np.int32  # 使用整數類型
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        
        hand_units = int(round(-action)) 
        
        def _do_sell_normal():
            if self.state[index + 1] > 0:

                # 計算當前持有的手數
                current_hands = self.state[index + self.stock_dim + 1] // 1000
                # 確保不超過持有的手數
                sell_hands = min(hand_units, current_hands)
                # 轉換為股數
                sell_num_shares = sell_hands * 1000
                
                sell_amount = math.ceil(self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct[index]))
                # update balance
                self.state[0] += sell_amount

                self.state[index + self.stock_dim + 1] -= sell_num_shares
                self.cost += math.ceil(self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct[index]))
                self.trades += 1

            else:
                sell_num_shares = 0

            return sell_num_shares

        # # perform sell action based on the sign of the action
        # if self.turbulence_threshold is not None:
        #     if self.turbulence >= self.turbulence_threshold:
        #         if self.state[index + 1] > 0:
        #             # Sell only if the price is > 0 (no missing data in this particular date)
        #             # if turbulence goes over threshold, just clear out all positions
        #             if self.state[index + self.stock_dim + 1] > 0:
        #                 # Sell only if current asset is > 0
        #                 sell_num_shares = self.state[index + self.stock_dim + 1]
                        
        #                 sell_num_shares = (sell_num_shares // 1000) * 1000
                        
        #                 sell_amount = (
        #                     self.state[index + 1]
        #                     * sell_num_shares
        #                     * (1 - self.sell_cost_pct[index])
        #                 )
        #                 # update balance
        #                 self.state[0] += sell_amount
        #                 self.state[index + self.stock_dim + 1] = 0
        #                 self.cost += (
        #                     self.state[index + 1]
        #                     * sell_num_shares
        #                     * self.sell_cost_pct[index]
        #                 )
        #                 self.trades += 1
        #             else:
        #                 sell_num_shares = 0
        #         else:
        #             sell_num_shares = 0
        #     else:
        #         sell_num_shares = _do_sell_normal()
        # else:
        sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        
        hand_units = int(round(action))
                
        def _do_buy():
            if self.state[index + 1] > 0:
            
                # 計算最大可買入手數（基於現有資金）
                price_per_hand = self.state[index + 1] * 1000  # 一手的價格
                # if math.ceil(price_per_hand * (1 + self.buy_cost_pct[index])) == 0:
                #     print()
                available_hands = self.state[0] // math.ceil(price_per_hand * (1 + self.buy_cost_pct[index]))
                
                # 確保不超過最大可買入手數
                buy_hands = min(hand_units, available_hands)
                # 轉換為股數
                buy_num_shares = buy_hands * 1000
                
                buy_amount = math.ceil(self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct[index]))
                
                if buy_num_shares > 0:
                    self.state[0] -= buy_amount
                    self.state[index + self.stock_dim + 1] += buy_num_shares
                    self.cost += math.ceil(self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct[index]))
                    self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        # else:
        #     if self.turbulence < self.turbulence_threshold:
        #         buy_num_shares = _do_buy()
        #     else:
        #         buy_num_shares = 0
        #         pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state[0]
                + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * np.array(
                        self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    )
                )
                - self.asset_memory[0]
            )  # initial_amount is only cash part of our initial asset
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                plt.close()

            # Add outputs to logger interface
            # logger.record("environment/portfolio_value", end_total_asset)
            # logger.record("environment/total_reward", tot_reward)
            # logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            # logger.record("environment/total_cost", self.cost)
            # logger.record("environment/total_trades", self.trades)

            return self.state, self.reward, self.terminal, False, {}

        else:
            actions = np.round(actions).astype(np.int32)
            
            # if len(self.df.tic.unique()) > 1:
            #     no_trade_mask = self.data['close'].values == 0
            # else:
            #     no_trade_mask = self.data['close'] == 0
                
            # no_trade_indices = np.where(no_trade_mask)[0]
            
            # if len(no_trade_indices) > 0:
            #     actions[no_trade_indices] = 0
    
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                old_shares = self.state[index + self.stock_dim + 1]
                price = self.state[index+1]
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                new_shares = self.state[index + self.stock_dim + 1]
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

                if old_shares != new_shares:  # 只在實際發生交易時記錄
                    self.logger.log_trade(
                        timestamp=self._get_date(),
                        stock_code=self.df.tic.unique()[index],
                        action_type='sell',
                        price=price,
                        quantity=(old_shares - new_shares)/1000,
                        holdings=new_shares/1000,
                        cash_balance=self.state[0],
                        total_value=begin_total_asset,
                        sac_action=actions[index]/1000
                    )
                    # print({"timestamp": self._get_date(),"stock_code": self.df.tic.unique()[index],"action_type": 'sell',"price": price,"quantity": (old_shares - new_shares)/1000,"holdings": new_shares/1000,"cash_balance": self.state[0],"total_value": begin_total_asset,"sac_action": actions[index]/1000})
                    
            for index in buy_index:
                old_shares = self.state[index + self.stock_dim + 1]
                price = self.state[index+1]
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])
                new_shares = self.state[index + self.stock_dim + 1]
    
                if old_shares != new_shares:  # 只在實際發生交易時記錄
                    self.logger.log_trade(
                        timestamp=self._get_date(),
                        stock_code=self.df.tic.unique()[index],
                        action_type='buy',
                        price=price,
                        quantity=(new_shares - old_shares)/1000,
                        holdings=new_shares/1000,
                        cash_balance=self.state[0],
                        total_value=begin_total_asset,
                        sac_action=actions[index]/1000
                    )
                    # print({"timestamp": self._get_date(),"stock_code": self.df.tic.unique()[index],"action_type": 'buy',"price": price,"quantity": (old_shares - new_shares)/1000,"holdings": new_shares/1000,"cash_balance": self.state[0],"total_value": begin_total_asset,"sac_action": actions[index]/1000})
                
            self.actions_memory.append(actions)

            # state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day, :]
            # if self.turbulence_threshold is not None:
            #     if len(self.df.tic.unique()) == 1:
            #         self.turbulence = self.data[self.risk_indicator_col]
            #     elif len(self.df.tic.unique()) > 1:
            #         self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling
            self.state_memory.append(
                self.state
            )  # add current state in state_recorder for each step

        return self.state, self.reward, self.terminal, False, {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        # initiate state
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + self.num_stock_shares
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )  # append initial stocks_share to initial state, instead of all zero
            else:
                # for single stock
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        return state

    # def _initiate_state(self):
    #     if self.initial:
    #         # 對於初始狀態
    #         if len(self.df.tic.unique()) > 1:
    #             # 多股票情況
    #             state = (
    #                 [self.initial_amount]
    #                 + [p if m == 1 else 0 for p, m in zip(self.data.close.values, self.data.close_mask.values)]  # 根據遮罩處理價格
    #                 + self.num_stock_shares
    #                 + sum(
    #                     (
    #                         [v if m == 1 else 0 for v, m in zip(self.data[tech].values, self.data.close_mask.values)]
    #                         for tech in self.tech_indicator_list
    #                     ),
    #                     [],
    #                 )
    #             )
    #         else:
    #             # 單股票情況
    #             state = (
    #                 [self.initial_amount]
    #                 + [self.data.close if self.data.close_mask == 1 else 0]
    #                 + [0] * self.stock_dim
    #                 + sum(
    #                     ([self.data[tech] if self.data.close_mask == 1 else 0] for tech in self.tech_indicator_list),
    #                     []
    #                 )
    #             )
    #     else:
    #         # 使用先前狀態
    #         if len(self.df.tic.unique()) > 1:
    #             state = (
    #                 [self.previous_state[0]]
    #                 + [p if m == 1 else 0 for p, m in zip(self.data.close.values, self.data.close_mask.values)]
    #                 + self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
    #                 + sum(
    #                     (
    #                         [v if m == 1 else 0 for v, m in zip(self.data[tech].values, self.data.close_mask.values)]
    #                         for tech in self.tech_indicator_list
    #                     ),
    #                     [],
    #                 )
    #             )
    #         else:
    #             state = (
    #                 [self.previous_state[0]]
    #                 + [self.data.close if self.data.close_mask == 1 else 0]
    #                 + self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
    #                 + sum(
    #                     ([self.data[tech] if self.data.close_mask == 1 else 0] for tech in self.tech_indicator_list),
    #                     []
    #                 )
    #             )
    #     return state
    
    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
            )

        else:
            # for single stock
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )

        return state

    # def _update_state(self):
    #     """更新狀態時考慮遮罩"""
    #     if len(self.df.tic.unique()) > 1:
    #         # 多股票情況
    #         state = (
    #             [self.state[0]]  # 現金部分
    #             + [p if m == 1 else 0 for p, m in zip(self.data.close.values, self.data.close_mask.values)]  # 價格
    #             + list(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])  # 持股數量
    #             + sum(
    #                 (
    #                     [v if m == 1 else 0 for v, m in zip(self.data[tech].values, self.data.close_mask.values)]
    #                     for tech in self.tech_indicator_list
    #                 ),
    #                 [],
    #             )  # 技術指標
    #         )
    #     else:
    #         # 單股票情況
    #         state = (
    #             [self.state[0]]
    #             + [self.data.close if self.data.close_mask == 1 else 0]
    #             + list(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
    #             + sum(
    #                 ([self.data[tech] if self.data.close_mask == 1 else 0] for tech in self.tech_indicator_list),
    #                 []
    #             )
    #         )
    #     return state
    
    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        # print(df_states)
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
