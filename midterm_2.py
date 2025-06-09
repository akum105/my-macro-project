import pandas as pd
import numpy as np

# 1) データ読み込み
pwt90 = pd.read_stata('https://www.rug.nl/ggdc/docs/pwt90.dta')

# 2) 22のOECD国リスト
oecd_countries = [
    "Australia","Austria","Belgium","Canada","Denmark","Finland","France","Germany",
    "Greece","Iceland","Ireland","Italy","Japan","Netherlands","New Zealand","Norway",
    "Portugal","Spain","Sweden","Switzerland","United Kingdom","United States"
]

# 3) 1960–2000年のデータをフィルタ
data = (pwt90
    .query("country in @oecd_countries and 1960 <= year <= 2000")
    .loc[:, ['country','year','rgdpna','rkna','emp','avh','labsh','rtfpna']]
    .dropna()
    .copy()
)

# 4) 必要変数を計算
data['alpha']    = 1 - data['labsh']                                  # 資本分配率
data['y_n']      = data['rgdpna'] / data['emp']                        # Y/L
data['hours']    = data['emp'] * data['avh']                           # 総労働時間
data['tfp_term'] = data['rtfpna'] ** (1 / (1 - data['alpha']))         # A^(1/(1-α))
data['cap_term'] = (data['rkna'] / data['rgdpna']) ** (data['alpha'] / (1 - data['alpha']))  # (K/Y)^(α/(1-α))

# 5) 成長率計算関数（幾何平均を使用）
def calc_growth(df):
    start = df[df.year == df.year.min()].iloc[0]
    end   = df[df.year == df.year.max()].iloc[0]
    T     = end.year - start.year

    # 幾何平均成長率（年率 %）
    g_y = ((end.y_n    / start.y_n)    ** (1/T) - 1) * 100
    g_k = ((end.cap_term / start.cap_term) ** (1/T) - 1) * 100
    g_a = ((end.tfp_term / start.tfp_term) ** (1/T) - 1) * 100

    alpha_bar = (start.alpha + end.alpha) / 2
    cap_dep   = alpha_bar * g_k
    tfp_grow  = g_a

    return {
        'Country':           start.country,
        'Growth Rate':       round(g_y, 2),
        'TFP Growth':        round(tfp_grow, 2),
        'Capital Deepening': round(cap_dep, 2),
        'TFP Share':         round(tfp_grow   / g_y, 2),
        'Capital Share':     round(cap_dep    / g_y, 2)
    }

# 6) 各国＋平均をまとめて表示
results = data.groupby('country').apply(calc_growth).tolist()
df_res  = pd.DataFrame(results)

avg = {
    'Country': 'Average',
    **{col: round(df_res[col].mean(), 2)
       for col in df_res.columns if col!='Country'}
}
df_res = pd.concat([df_res, pd.DataFrame([avg])], ignore_index=True)

print("Growth Accounting in OECD Countries: 1960–2000")
print("="*80)
print(df_res.to_string(index=False))
