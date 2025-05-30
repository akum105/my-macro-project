import statsmodels.api as sm
import pandas_datareader as pdr
import numpy as np
import matplotlib.pyplot as plt

#期間設定
start_date = '1995-01-01'
end_date = '2020-01-01'

# 1. イギリスを選んだ
# イギリスと日本の実質GDP取得
gdp_UK = pdr.DataReader('CLVMNACSCAB1GQUK', 'fred', start_date, end_date)
log_gdp_UK = np.log(gdp_UK)
gdp_JP = pdr.DataReader('JPNRGDPEXP', 'fred', start_date, end_date)
log_gdp_JP = np.log(gdp_JP)

# 2, 3 HP-filter適用
cycle_UK, trend_UK = sm.tsa.filters.hpfilter(log_gdp_UK, lamb=1600)
cycle_JP, trend_JP = sm.tsa.filters.hpfilter(log_gdp_JP, lamb=1600)

# 4.a 選んだ国および日本について循環変動成分の標準偏差を計算して比較（％）
std_UK = cycle_UK.std() * 100
std_JP = cycle_JP.std() * 100
print(f"イギリスの標準偏差: {std_UK:.3f}%")
print(f"日本の標準偏差: {std_JP:.3f}%")
# 4.b 日本とイギリスの景気循環成分の相関係数計算
cycles = pd.concat([cycle_UK.rename('UK'), cycle_JP.rename('JP')], axis=1)
corr_coef = cycles['UK'].corr(cycles['JP'])
print(f"日本とイギリスの景気循環成分の相関係数: {corr_coef:.3f}")

# 5. 両系列を結合してプロット用に整形
cycles = pd.concat([cycle_UK.rename('UK'), cycle_JP.rename('Japan')], axis=1)

# プロット
plt.figure(figsize=(10, 6))
cycles.plot()
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title('Cyclical Components: UK & Japan')
plt.xlabel('Date')
plt.ylabel('Cyclical Component')
plt.legend()
plt.tight_layout()
plt.show()