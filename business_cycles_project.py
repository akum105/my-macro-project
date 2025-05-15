import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# データ取得期間
start_date = '2000-01-01'
end_date   = '2025-01-01'

# FRED からフランス実質GDPを取得し、対数変換
gdp     = web.DataReader('CLVMNACSCAB1GQFR', 'fred', start_date, end_date)
log_gdp = np.log(gdp)

# HPフィルタの平滑化パラメータ
lambdas = [10, 100, 1600]

plt.figure(figsize=(10, 6))

# 元系列を１回だけプロット
plt.plot(log_gdp, label="Original GDP (log)", color='black', linewidth=2)

# 各λで抽出したトレンド成分を重ね書き
for lam in lambdas:
    cycle, trend = sm.tsa.filters.hpfilter(log_gdp, lamb=lam)
    plt.plot(trend, label=f"Trend (λ={lam})", linewidth=1.5)

plt.title("Original GDP and HP Filter Trends (λ = 10, 100, 1600)")
plt.xlabel("Date")
plt.ylabel("Log GDP")
plt.legend()
plt.tight_layout()
plt.show()