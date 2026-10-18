import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from DCC_class import *
from Utilities import *
import matplotlib.pyplot as plt

spx = pd.read_csv('Data/SPX.csv')
spx1 = pd.read_csv('Data/SPX1.csv')
spx['Date'] = pd.to_datetime(spx['Date'])
spx1['Date'] = pd.to_datetime(spx1['Date'])
spx = pd.concat([spx, spx1])
spx = spx.set_index('Date')
spx['Price'] = spx['Price'].str.replace(',', '', regex=True)
spx['Price'] = spx['Price'].astype(float)
spx = spx[['Price']]
spx = spx[::-1]
spx.columns = ['SPX']

ust = pd.read_excel('Data/DGS10.xlsx', sheet_name='Daily')
ust.columns = ['Date', 'UST']
ust = ust.set_index('Date')
ust.index = pd.to_datetime(ust.index)

data = pd.concat([spx, ust], axis=1)
data = data.dropna()
data = data.resample('ME').last()
data['SPX'] = np.log(data['SPX']).diff() * 100
data['UST'] = data['UST'].diff() * 100
data = data.dropna()

reg = OLS(data['SPX'], sm.add_constant(data['UST'])).fit()
print(reg.summary())

dcc = DCC(data, vol_asymmetry=True).fit()
print(dcc.summary())

conditional_corr = dcc.conditional_cov()['Conditional correlation']
corr = []
for i in range(len(conditional_corr)):
    corr.append(conditional_corr[i][1, 0])

corr = pd.DataFrame(corr)
corr.index = data.iloc[:-20].index

plt.plot(corr)
plt.axhline(np.corrcoef(data.T)[1][0], color='k')
plt.title('Correlation correlation')
plt.show()

partial_views = [[-10]]
partial_names = ['SPX']
start = pd.to_datetime('07/10/31', format='%y/%m/%d')
end = pd.to_datetime('09/12/31', format='%y/%m/%d')
prediction_GFC = SPX_UST(data, partial_views, partial_names, **{'start': start, 'end': end, 'criterion': 'Min'})
start = pd.to_datetime('22/12/31', format='%y/%m/%d')
end = pd.to_datetime('25/12/31', format='%y/%m/%d')
prediction_rates_up = SPX_UST(data, partial_views, partial_names, **{'start': start, 'end': end, 'criterion': 'Max'})

oos_results = []
for t in range(len(data) - 30, len(data)):
    # Slice the historical data strictly up to the current day 't' (prevents data leakage)
    train_slice = data.iloc[:t]
    target_date = data.index[t]

    # Instantiate your DCC class on the active historical timeline slice
    model = DCC(y=train_slice, mean=False, err_dist="Normal", vol_asymmetry=True).fit()

    # Call your fixed forecast method for a 1-day forward step
    forecast_out = model.forecast(step=1)

    # Extract the 1-step-ahead Covariance Matrix
    cov_matrix_f = forecast_out["Conditional covariance"][0]

    # Safely isolate the scalar elements from the forecasted matrix structure
    # Indexing assumes: [0, 0] is SPX Var, [1, 1] is US10Y Var, [0, 1] is Covariance
    var_ty_f = cov_matrix_f[1, 1]
    cov_spx_ty_f = cov_matrix_f[0, 1]

    # Pair the forward metrics alongside the actual realized outcomes for that day
    oos_results.append({
        'Date': target_date,
        'SPX_Actual': data.iloc[t].iloc[0],
        'TY_Actual': data.iloc[t].iloc[1],
        'Forecast_Var_TY': var_ty_f,
        'Forecast_Cov': cov_spx_ty_f
        })

# Convert compilation records into a structured backtest tracking dataframe
backtest_df = pd.DataFrame(oos_results).set_index('Date')

# =====================================================================
# 3. APPLY DYNAMIC HEDGING FRAMEWORK
# =====================================================================
# Minimum-Variance Hedge Ratio: Beta_{t+1|t} = Covariance_{t+1|t} / Variance_TY_{t+1|t}
backtest_df['Hedge_Ratio'] = backtest_df['Forecast_Cov'] / backtest_df['Forecast_Var_TY']

# Form the Hedged Return series: Long SPX, Short (Beta * Treasury Position)
backtest_df['Hedged_Return'] = backtest_df['SPX_Actual'] - backtest_df['Hedge_Ratio'] * backtest_df['TY_Actual']

# =====================================================================
# 4. OUT-OF-SAMPLE EVALUATION DIAGNOSTICS
# =====================================================================
var_unhedged = backtest_df['SPX_Actual'].var()
var_hedged = backtest_df['Hedged_Return'].var()
hedge_effectiveness = (1 - (var_hedged / var_unhedged)) * 100

sharpe_unhedged = (backtest_df['SPX_Actual'].mean() / backtest_df['SPX_Actual'].std()) * np.sqrt(252)
sharpe_hedged = (backtest_df['Hedged_Return'].mean() / backtest_df['Hedged_Return'].std()) * np.sqrt(252)

print("\n" + "="*55)
print("     DYNAMIC ROLLING OUT-OF-SAMPLE HEDGING RESULTS     ")
print("="*55)
print(f"Unhedged Portfolio (SPX) Variance : {var_unhedged:.6f}")
print(f"DCC Forecast Hedged Variance      : {var_hedged:.6f}")
print(f"Hedge Effectiveness Index         : {hedge_effectiveness:.2f}% Variance Reduction")
print(f"Unhedged Annualised Sharpe Ratio  : {sharpe_unhedged:.4f}")
print(f"Hedged Annualised Sharpe Ratio    : {sharpe_hedged:.4f}")
print("="*55)