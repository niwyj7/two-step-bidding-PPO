import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PowerTradingEnv(gym.Env):
    """
    Multi-stage MDP Environment for Power Market Arbitrage.
    Stages: D-2 (t=0) -> DA (t=1) -> RT Settlement
    """
    def __init__(self, df):
        super(PowerTradingEnv, self).__init__()
        self.df = df
        self.max_days = len(df)
        self.current_step = 0
        self.current_day = 0
        
        # Action: Position sizing adjustments [-0.5, 0.5]
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)
        
        # State: 5 weather features + 2 D-2 signals + 2 DA signals + current_q + time_step = 11 dims
        self.observation_space = spaces.Box(low=-10000, high=10000, shape=(11,), dtype=np.float32)
        self.q_d2 = 0.0 
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_day = np.random.randint(0, self.max_days)
        self.current_step = 0
        self.q_d2 = 0.0
        return self._get_obs(), {}
        
    def _get_obs(self):
        row = self.df.iloc[self.current_day]
        weather_cols = ['win100_spd','d2','ssrd','tcc','hour']
        weather_features = row[weather_cols].values.astype(np.float32)
        
        d2_s, d2_c = row['d2_pred_spread'], row['d2_confidence']
        
        if self.current_step ==
