import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from jax import grad
from jax.scipy.stats import norm


def blackScholes(S, K, T, r, sigma, q=0, is_call=True):
    d1 = (np.log(S/K) + (r - q + sigma **2 /2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if is_call:
        price = S * np.exp(-q * T) * norm.cdf(d1, 0, 1) - K * np.exp(-r * T ) * norm.cdf(d2, 0, 1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2, 0, 1) - S * np.exp(-q * T) * norm.cdf(-d1, 0, 1)
    
    return price

# Returns difference between actual and theoretical values
def loss(S, K, T, r, sigma_guess, price, q=0, is_call=True):
    theoretical_price = blackScholes(S, K, T, r, sigma_guess, q, is_call)
    return theoretical_price - price

loss_grad = grad(loss, argnums=4)

def solve_for_iv(S, K, T, r, q, market_price, epsilon=0.001, max_iter=20, is_call=True):
    sigma = 1.8
    
    for i in range(max_iter):
        vega_val = loss_grad(S, K, T, r, sigma, market_price, q, is_call)
        loss_val = loss(S, K, T, r, sigma, market_price, q, is_call)

        # Newton's Method
        sigma = sigma - loss_val/vega_val
        
        if abs(loss_val) < epsilon:
            break

    return sigma
        
ticker_name = input("Enter ticker name: ")
r = float(input("Enter Interest Rate: "))
q = float(input("Enter Dividend Yield: "))
ticker = yf.Ticker(ticker_name)

    
total_calls = pd.DataFrame(columns=["symbol", "bid", "ask", "lastPrice", "strike", "expiration", "dte"])

iv_lst = []
moneyness_lst = []
dte_lst = []
S = ticker.fast_info["last_price"]

for datetime_string in ticker.options:
    sorted_calls = ticker.option_chain(datetime_string).calls
    datetime_object = datetime.strptime(datetime_string, '%Y-%m-%d')
    
    for idx, call in sorted_calls.iterrows(): 
        if call["strike"] < S * 1.3 and call["strike"] > S * 0.7:
            K = call["strike"]
            T = (datetime_object - datetime.now()).days / 365
            market_price = call["lastPrice"]
            iv = call["impliedVolatility"]
            #iv = solve_for_iv(S, K, T, r, q, market_price)
            
            iv_lst.append(iv)
            moneyness_lst.append(S/K)
            dte_lst.append(T)
            
            total_calls.loc[len(total_calls)] = [ticker_name, call["bid"], call["ask"], call["lastPrice"], call["strike"], datetime_string, (datetime_object - datetime.now()).days / 365]
    
    
fig = plt.figure(figsize=(12, 8), dpi=100)
ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_trisurf(moneyness_lst, dte_lst, iv_lst, 
                          cmap='viridis',
                          linewidth=0.1,
                          antialiased=True,
                          alpha=0.8)

colour_bar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, pad=0.1)
colour_bar.set_label("Implied Volatility", rotation=270, labelpad=15)

ax.set_xlabel("Moneyness (S/K)")
ax.set_ylabel("Time to Expiration( (Years)")
ax.set_zlabel("Implied Volatility")

# Set the initial view angle (only azimuth is fixed, elevation can vary)
ax.view_init(elev=30, azim=120)


plt.title(f"Volatility Surface for {ticker_name}")
plt.show()