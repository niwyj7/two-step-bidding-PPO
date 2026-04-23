## Project Overview

This project formulates **multi-stage power trading** as a **two-step sequential decision-making problem** and solves it with **Proximal Policy Optimisation (PPO)**. The agent integrates:

* weather features,
* order-book derived market prices,
* day-ahead and real-time price forecasts,
* forecast confidence signals,

to determine trading positions across two stages and maximise risk-adjusted profit under position constraints. 

## Problem Formulation

Considering a two-stage trading process:

* **Stage 1 (D-2 stage):** the agent determines an early position $q_{d2}$,
* **Stage 2 (DA stage):** the agent determines an additional position $q_{da}$,

such that the final position remains close to a target delivery level while maximising spread-based profit. This naturally defines a finite-horizon Markov Decision Process (MDP). 


## 1. Market Price Construction

### 1.1 Mid-price from order book

From the best bid and ask prices in the order book, the reference execution price is defined as:

$$
p_t^{\text{deal}} = \frac{p_t^{\text{bid}} + p_t^{\text{ask}}}{2}
$$

This serves as the benchmark price for the early-stage trading spread. 


## 2. Spread Definitions

### 2.1 Realised spreads

Two realised spreads are constructed from actual market prices:

$$
y_t^{\text{rt-d2}} = p_t^{\text{rt}} - p_t^{\text{deal}}
$$

$$
y_t^{\text{rt-da}} = p_t^{\text{rt}} - p_t^{\text{da}}
$$

where:

* $p_t^{\text{rt}}$ is the real-time price,
* $p_t^{\text{da}}$ is the day-ahead price,
* $p_t^{\text{deal}}$ is the order-book mid-price. 

These quantities represent the realised return opportunities for the two trading stages.

### 2.2 Predicted spreads

The model further constructs predictive spread signals:

$$
\hat{s}_{d2,t} = \hat{p}_t^{\text{rt-d2}} - p_t^{\text{deal}}
$$

$$
\hat{s}_{da,t} = \hat{p}_t^{\text{rt-da}} - \hat{p}_t^{\text{da-da}}
$$

where:

* $\hat{p}_t^{\text{rt-d2}}$ is the D-2 forecast of real-time price,
* $\hat{p}_t^{\text{rt-da}}$ is the DA forecast of real-time price,
* $\hat{p}_t^{\text{da-da}}$ is the DA forecast of day-ahead price. 

These predictive spreads provide the agent with expected arbitrage signals before taking actions.


## 3. Forecast Confidence Modelling

To quantify predictive reliability, forecast confidence is defined as the inverse rolling volatility of prediction error.

### 3.1 Prediction errors

$$
e_{d2,t} = \hat{p}_t^{\text{rt-d2}} - p_t^{\text{rt}}
$$

$$
e_{da,t} = \hat{p}_t^{\text{rt-da}} - p_t^{\text{rt}}
$$

### 3.2 Rolling-error-based confidence

Using a rolling window of length (w), confidence is defined as:

$$
c_{d2,t}^{\text{raw}} = \frac{1}{\mathrm{Std}(e_{d2,t-w+1:t}) + \varepsilon}
$$

$$
c_{da,t}^{\text{raw}} = \frac{1}{\mathrm{Std}(e_{da,t-w+1:t}) + \varepsilon}
$$

where $\varepsilon$ is a small positive constant for numerical stability. In the implementation, the rolling window is set to:

$$
w = 96 \times 7
$$

corresponding to a 7-day window under 15-minute sampling frequency. 

### 3.3 Min-max normalisation

The raw confidence values are then normalised:

$$
c_{d2,t} =
\frac{
c_{d2,t}^{\text{raw}} - \min(c_{d2}^{\text{raw}})
}{
\max(c_{d2}^{\text{raw}}) - \min(c_{d2}^{\text{raw}}) + \varepsilon
}
$$

$$
c_{da,t} =
\frac{
c_{da,t}^{\text{raw}} - \min(c_{da}^{\text{raw}})
}{
\max(c_{da}^{\text{raw}}) - \min(c_{da}^{\text{raw}}) + \varepsilon
}
$$

This normalisation makes confidence features numerically stable for policy learning. 


## 4. MDP Formulation

### 4.1 State space

The state contains weather features and market-related features.

#### Weather features

$$
x_t^{\mathrm{weather}} =
\left[
\mathrm{win100spd}_t\,
d2_t\,
ssrd_t\,
tcc_t\,
hour_t
\right]
$$

#### Market features

$$
x_t^{\text{market}} =
[
\hat{s}*{d2,t},
c*{d2,t},
\hat{s}*{da,t},
c*{da,t},
q_{d2},
\text{step}
]
$$

Thus, the full state is:

$$
s_t = [x_t^{\text{weather}}, x_t^{\text{market}}]
$$

which gives an 11-dimensional observation vector in the implementation. 


### 4.2 Action space

The action is continuous:

$$
a_t \in [-0.5, 0.5]
$$

with stage-dependent interpretation:

* at step 0:
  
$$
a_0 = q_{d2}
$$

* at step 1:
  
$$
a_1 = q_{da}
$$

Hence, the task is a **two-step continuous control problem**. 


## 5. Reward Function Derivation

### 5.1 Total position

The total trading position is defined as:

$$
Q = 1 + q_{d2} + q_{da}
$$

where the constant 1 denotes the baseline delivery position, and the two actions represent adjustments around that baseline. 



### 5.2 Trading profit

The realised profit-and-loss (PnL) is:

$$
\text{PnL}=
q_{d2}y_t^{\text{rt-d2}}
+
q_{da}y_t^{\text{rt-da}}
$$

Substituting the spread definitions:

$$
\text{PnL}=
q_{d2}(p_t^{\text{rt}} - p_t^{\text{deal}})
+
q_{da}(p_t^{\text{rt}} - p_t^{\text{da}})
$$

This means:

* The D-2 position benefits from the spread between real-time price and order-book mid-price,
* The DA position benefits from the spread between the real-time price and the day-ahead price. 



### 5.3 Position constraint penalty

To keep the final position close to the target range [0.9, 1.1], a quadratic penalty is introduced:

$$
\mathcal{P}(Q)=
\begin{cases}
\lambda (Q-1.1)^2, & Q > 1.1 \\
\lambda (0.9-Q)^2, & Q < 0.9 \\
0, & 0.9 \le Q \le 1.1
\end{cases}
$$

where the implementation sets:

$$
\lambda = 500
$$

This is a standard soft-constraint formulation. 



### 5.4 Final reward

The final reward is defined as:

$$
r = \text{PnL} - \mathcal{P}(Q)
$$

Substituting previous expressions gives:

$$
r = q_{d2}(p_t^{\text{rt}} - p_t^{\text{deal}}) + q_{da}(p_t^{\text{rt}} - p_t^{\text{da}})\mathcal{P}(1 + q_{d2} + q_{da})
$$

This reward jointly optimises profitability and delivery-position feasibility. 


## 6. Sequential Decision Interpretation

The environment has two decision steps.

### Step 1

The agent observes weather information, D-2 predictive spread, and D-2 confidence, then selects:

$$
q_{d2} = \pi_\theta(s_0)
$$

No immediate reward is assigned at this stage:

$$
r_0 = 0
$$

because the quality of the first-stage decision can only be evaluated after the second-stage adjustment and final settlement price are observed. 

### Step 2

The agent then observes updated DA information and chooses:

$$
q_{da} = \pi_\theta(s_1)
$$

The terminal reward becomes:

$$
r_1 =q_{d2}(p_t^{\text{rt}} - p_t^{\text{deal}})+q_{da}(p_t^{\text{rt}} - p_t^{\text{da}})\mathcal{P}(Q)
$$

Therefore, the total return is simply:

$$
G = r_0 + r_1 = r_1
$$

This yields a delayed-reward sequential trading problem. 


## 7. PPO Optimization

The project uses PPO with an MLP policy network to solve the continuous trading control problem. The standard PPO clipped objective is:

$$
L^{\mathrm{CLIP}}(\theta)=
\mathbb{E}_t
\left[
\min \left(
r_t(\theta)\hat{A}_t\,
\mathrm{clip}\!\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right)\hat{A}_t
\right)
\right]
$$

where

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

and $\hat{A}_t$ is the advantage estimate. PPO is particularly suitable here because it stabilises policy updates in continuous-action settings. 


## 8. Main Contributions

This project can be summarised by the following methodological contributions:

1. **Multi-source feature fusion**: combining weather, order-book prices, and forecast signals into a unified trading state.
2. **Confidence-aware decision-making**: using inverse rolling forecast volatility as a confidence measure.
3. **Two-stage trading formulation**: modeling D-2 and DA decisions as sequential continuous control.
4. **Profit-risk unification**: integrating spread profit and delivery-position constraints into a single reward function.
5. **RL-based policy learning**: solving the problem with PPO rather than fixed-rule optimisation. 


## 9. Implementation Note

The observation vector is effectively **11-dimensional**:

* 5 weather features,
* 6 market-related features.

This should match the environment definition in the code. 


## EDA & Data Visualisation
Examples of  EDA & Data Visualisation: price spread, intraday profile, cross-day weather change, orderbook & slippage, residual analysis
* Price & Price Prediction
<br>
<div align="center">
  <img width="712" height="280" alt="cdbbc624-063b-4855-ab02-a673cb3f1b62" src="https://github.com/user-attachments/assets/51b882e9-92ff-4e5b-9a02-1f924b5f5cf7" /><br>
  Below shows systematic estimation error:
  <img width="712" height="280" alt="f678bb81-8b70-4e7e-ac10-dcf44a7171be" src="https://github.com/user-attachments/assets/ae710c9e-41b5-40e8-9581-9f61a4538537" /><br>
  <img width="1000" height="352" alt="361c3de7-8d20-41eb-8c76-26204e70eeb1" src="https://github.com/user-attachments/assets/6f6800cd-38b2-4b49-a235-0bd349abd2b4" /><br>
  <img width="1000" height="352" alt="6c4a0791-eada-4d02-81e9-c15b6085c64b" src="https://github.com/user-attachments/assets/db3a4beb-8e4d-4946-b3ee-e04346d23182" />
</div>  

<br>
* Weather Features
<br>
<br>
<div align="center">
  <img width="1113" height="139" alt="43141745-0335-456a-85b3-8de55bacd7dc" src="https://github.com/user-attachments/assets/c9d592d8-4399-4875-b2a9-f5863af5c684" /> <br>
  most negative bin
  <img width="867" height="198" alt="1b78ccc7-4525-4fea-a40a-a4f694118c2d" src="https://github.com/user-attachments/assets/a1333890-aef1-411f-bbb4-188e0a55580f" /><br>
  <img width="1280" height="703" alt="d045595c-2a82-4331-8017-9387c4352c4c" src="https://github.com/user-attachments/assets/5786ba2a-35fd-42fb-b7b8-6d688cd9ed0f" /><br>
  <img width="436" height="280" alt="f011aa47-90f0-453c-bdcb-993980ef539b" src="https://github.com/user-attachments/assets/6ca594a2-6814-4e35-9ef9-63ab70974f5c" />
</div>
<br>
* Price Curve Samples & Analysis
<br>
<br>
<div align="center">
  <img width="1001" height="424" alt="dd9b4e8e-2721-497f-b9e9-06c070b3bbe4" src="https://github.com/user-attachments/assets/fd450690-83fb-49b4-b17a-de6179e8e9e2" />
  <img width="1001" height="424" alt="9a24392b-6805-49d8-918c-2327edf31cd0" src="https://github.com/user-attachments/assets/abb66312-e392-41e0-8693-8b8624fa8c84" />
  <img width="1001" height="424" alt="07c6b10f-8f94-44b7-bd57-48520b1b6b1e" src="https://github.com/user-attachments/assets/867bf40d-691e-4509-862e-70cd8520b21d" />
</div> 

<br>
<br>
  
<img width="2912" height="1464" alt="ik0u2iik0u2iik0u" src="https://github.com/user-attachments/assets/35053f65-2212-4d94-b236-402d27c5af8f" />
<br>
<img width="2048" height="2048" alt="2keh242keh242keh" src="https://github.com/user-attachments/assets/a43801f2-4afa-477b-92ba-00718581a9c9" />

