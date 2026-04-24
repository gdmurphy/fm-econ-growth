"""
Plot open-weights model parameter counts vs. minimum inference price on OpenRouter.

Price metric: 3 * prompt_price + completion_price (per token), scaled to per 1M tokens.
Period: July 2025 and later.

Outputs:
  params_vs_price.png          — raw scatter, labelled
  params_vs_price_binscatter.png — GPU-price-adjusted, binscatter + OLS line
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

project_dir = Path(__file__).parent.parent
data_dir = project_dir / 'data'
out_dir = Path(__file__).parent

# SemiAnalysis H100 SXM 80G 1-year price (USD/hr) — most recent observation
GPU_PRICE = 2.35


# ---------------------------------------------------------------------------
# Mapping: Epoch model name -> one or more OpenRouter IDs
# ---------------------------------------------------------------------------
MAPPING = {
    'Kimi K2 Thinking':                    ['moonshotai/kimi-k2-thinking-20251106'],
    'Kimi K2':                             ['moonshotai/kimi-k2'],
    'Ling-1T':                             ['inclusionai/ling-1t'],
    'DeepSeek-R1 (May 2025)':              ['deepseek/deepseek-r1-0528'],
    'DeepSeek-V3 (Mar 2025)':              ['deepseek/deepseek-chat-v3-0324'],
    'DeepSeek-R1':                         ['deepseek/deepseek-r1'],
    'DeepSeek-V3':                         ['deepseek/deepseek-chat-v3'],
    'LongCat-Flash':                       ['meituan/longcat-flash-chat'],
    'Qwen3-Coder-480B-A35B':               ['qwen/qwen3-coder-480b-a35b-07-25'],
    'ERNIE-4.5-VL-424B-A47B (文心大模型4.5)': ['baidu/ernie-4.5-vl-424b-a47b'],
    'Llama 3.1-405B':                      ['meta-llama/llama-3.1-405b-instruct'],
    'Llama 4 Maverick':                    ['meta-llama/llama-4-maverick-17b-128e-instruct'],
    'Jamba 1.5-Large':                     ['ai21/jamba-1-5-large'],
    'GLM-4.7':                             ['z-ai/glm-4.7-20251222'],
    'GLM-4.6':                             ['z-ai/glm-4.6', 'z-ai/glm-4.6-20251208'],
    'GLM-4.5':                             ['z-ai/glm-4.5'],
    'Nemotron-4 340B':                     ['nvidia/nemotron-4-340b-instruct'],
    'Qwen3-235B-A22B-Thinking (Jul 2025)': ['qwen/qwen3-235b-a22b-thinking-2507'],
    'Qwen3-235B-A22B (Jul 2025)':          ['qwen/qwen3-235b-a22b-07-25'],
    'Qwen3-235B-A22B':                     ['qwen/qwen3-235b-a22b-04-28'],
    'MiniMax-M2.1':                        ['minimax/minimax-m2.1'],
    'MiniMax-M2':                          ['minimax/minimax-m2'],
    'DBRX':                                ['databricks/dbrx-instruct'],
    'Pixtral Large':                       ['mistralai/pixtral-large-2411'],
    'Mistral Large 2':                     ['mistralai/mistral-large-2411'],
    'gpt-oss-120b':                        ['openai/gpt-oss-120b'],
    'Llama 4 Scout':                       ['meta-llama/llama-4-scout-17b-16e-instruct'],
    'Qwen2-72B':                           ['qwen/qwen-2-72b-instruct'],
    'Qwen2.5-72B':                         ['qwen/qwen-2.5-72b-instruct'],
    'Llama 2-70B':                         ['meta-llama/llama-2-70b-chat'],
    'Llama 3-70B':                         ['meta-llama/llama-3-70b-instruct'],
    'Llama 3.3 70B':                       ['meta-llama/llama-3.3-70b-instruct'],
    'Mixtral 8x7B':                        ['mistralai/mixtral-8x7b-instruct'],
    'Yi-34B':                              ['01-ai/yi-34b-chat'],
    'Qwen3-Omni-30B-A3B':                  ['qwen/qwen3-30b-a3b-instruct-2507'],
    'Qwen2.5-32B':                         ['qwen/qwen2.5-32b-instruct'],
    'QwQ-32B':                             ['qwen/qwq-32b'],
    'Olmo 3':                              ['allenai/olmo-3-32b-think-20251121',
                                            'allenai/olmo-3.1-32b-instruct-20251215'],
    'Tongyi DeepResearch':                 ['alibaba/tongyi-deepresearch-30b-a3b'],
    'gpt-oss-20b':                         ['openai/gpt-oss-20b'],
    'Llama 3.2 11B':                       ['meta-llama/llama-3.2-11b-vision-instruct'],
}

SHORT_LABELS = {
    'Kimi K2 Thinking':                    'Kimi K2 Thinking',
    'Kimi K2':                             'Kimi K2',
    'Ling-1T':                             'Ling-1T',
    'DeepSeek-R1 (May 2025)':              'DS-R1 (May25)',
    'DeepSeek-V3 (Mar 2025)':              'DS-V3 (Mar25)',
    'DeepSeek-R1':                         'DS-R1',
    'DeepSeek-V3':                         'DS-V3',
    'LongCat-Flash':                       'LongCat-Flash',
    'Qwen3-Coder-480B-A35B':               'Qwen3-Coder-480B',
    'ERNIE-4.5-VL-424B-A47B (文心大模型4.5)': 'ERNIE-4.5-VL-424B',
    'Llama 3.1-405B':                      'Llama 3.1-405B',
    'Llama 4 Maverick':                    'Llama 4 Maverick',
    'Jamba 1.5-Large':                     'Jamba 1.5-L',
    'GLM-4.7':                             'GLM-4.7',
    'GLM-4.6':                             'GLM-4.6',
    'GLM-4.5':                             'GLM-4.5',
    'Nemotron-4 340B':                     'Nemotron-4 340B',
    'Qwen3-235B-A22B-Thinking (Jul 2025)': 'Qwen3-235B-Think',
    'Qwen3-235B-A22B (Jul 2025)':          'Qwen3-235B (Jul)',
    'Qwen3-235B-A22B':                     'Qwen3-235B',
    'MiniMax-M2.1':                        'MiniMax-M2.1',
    'MiniMax-M2':                          'MiniMax-M2',
    'DBRX':                                'DBRX',
    'Pixtral Large':                       'Pixtral Large',
    'Mistral Large 2':                     'Mistral Large 2',
    'gpt-oss-120b':                        'GPT-OSS-120B',
    'Llama 4 Scout':                       'Llama 4 Scout',
    'Qwen2-72B':                           'Qwen2-72B',
    'Qwen2.5-72B':                         'Qwen2.5-72B',
    'Llama 2-70B':                         'Llama 2-70B',
    'Llama 3-70B':                         'Llama 3-70B',
    'Llama 3.3 70B':                       'Llama 3.3-70B',
    'Mixtral 8x7B':                        'Mixtral 8x7B',
    'Yi-34B':                              'Yi-34B',
    'Qwen3-Omni-30B-A3B':                  'Qwen3-30B',
    'Qwen2.5-32B':                         'Qwen2.5-32B',
    'QwQ-32B':                             'QwQ-32B',
    'Olmo 3':                              'OLMo 3',
    'Tongyi DeepResearch':                 'Tongyi DR-30B',
    'gpt-oss-20b':                         'GPT-OSS-20B',
    'Llama 3.2 11B':                       'Llama 3.2-11B',
}

# ---------------------------------------------------------------------------
# Load Epoch data
# ---------------------------------------------------------------------------
epoch = pd.read_csv(data_dir / 'Epoch Database - Notable Models.csv')
open_weights_mask = (
    epoch['Model accessibility'].isin([
        'Open weights (restricted use)',
        'Open weights (unrestricted)',
        'Open weights (non-commercial)',
    ])
    & epoch['Parameters'].notna()
    & (epoch['Parameters'] != '')
)
epoch_ow = epoch.loc[open_weights_mask, ['Model', 'Parameters']].copy()
epoch_ow['params_num'] = pd.to_numeric(epoch_ow['Parameters'], errors='coerce')
epoch_ow = epoch_ow.dropna(subset=['params_num'])

# ---------------------------------------------------------------------------
# Load OpenRouter panel, filter to July 2025+, compute price
# ---------------------------------------------------------------------------
panel = pd.read_parquet(data_dir / 'openrouter_panel.parquet')
panel['date'] = pd.to_datetime(panel['date'])

panel = panel[~panel['openrouter_id'].str.endswith(':free', na=False)]
panel = panel[
    panel['prompt_price'].notna()
    & panel['completion_price'].notna()
    & (panel['prompt_price'] > 0)
    & (panel['completion_price'] > 0)
]
panel['price'] = (3 * panel['prompt_price'] + panel['completion_price']) * 1e6

min_price = panel.groupby('openrouter_id')['price'].min()

# Reverse mapping: openrouter_id -> params_num (for expanding to all timestamps)
id_to_params = {}
for epoch_name, or_ids in MAPPING.items():
    params_rows = epoch_ow[epoch_ow['Model'] == epoch_name]
    if params_rows.empty:
        continue
    params = params_rows.iloc[0]['params_num']
    for oid in or_ids:
        id_to_params[oid] = params

# ---------------------------------------------------------------------------
# Apply mapping: find minimum price per Epoch model (used for Plot 1)
# ---------------------------------------------------------------------------
rows = []
for epoch_name, or_ids in MAPPING.items():
    params_rows = epoch_ow[epoch_ow['Model'] == epoch_name]
    if params_rows.empty:
        print(f'  [WARN] Not found in Epoch DB: {epoch_name}')
        continue
    params = params_rows.iloc[0]['params_num']

    prices = [min_price[oid] for oid in or_ids if oid in min_price.index]
    if not prices:
        print(f'  [SKIP] No pricing data: {epoch_name}  (tried: {or_ids})')
        continue

    best_price = min(prices)
    rows.append({
        'name': epoch_name,
        'label': SHORT_LABELS.get(epoch_name, epoch_name),
        'params': params,
        'price': best_price,
        'adj_price': best_price * GPU_PRICE,
    })

df = pd.DataFrame(rows).sort_values('params', ascending=False)
print(f'\nMatched {len(df)} models with pricing data:\n')
print(df[['name', 'params', 'price', 'adj_price']].to_string(index=False))

# ---------------------------------------------------------------------------
# Full panel for Plot 2: all timestamps for mapped models
# ---------------------------------------------------------------------------
panel_full = panel[panel['openrouter_id'].isin(id_to_params)].copy()
panel_full['params'] = panel_full['openrouter_id'].map(id_to_params)
panel_full['adj_price'] = panel_full['price'] * GPU_PRICE
print(f'\nFull panel for binscatter: {len(panel_full)} observations across {panel_full["openrouter_id"].nunique()} model IDs')

# ---------------------------------------------------------------------------
# Shared axis formatter
# ---------------------------------------------------------------------------
def fmt_params(x, _):
    if x >= 1e12:
        return f'{x/1e12:.0f}T'
    if x >= 1e9:
        return f'{x/1e9:.0f}B'
    if x >= 1e6:
        return f'{x/1e6:.0f}M'
    return str(int(x))


# ---------------------------------------------------------------------------
# Plot 1: raw scatter with labels
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(df['params'], df['price'], color='steelblue', s=50, zorder=3, alpha=0.85)
for _, row in df.iterrows():
    ax.annotate(
        row['label'],
        xy=(row['params'], row['price']),
        xytext=(4, 4),
        textcoords='offset points',
        fontsize=6.5,
        color='#333333',
    )
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Total parameters', fontsize=12)
ax.set_ylabel('Min price: 3M prompt + 1M completion (USD)', fontsize=12)
ax.set_title(
    'Open-weights model inference price vs. parameter count\n'
    'OpenRouter, minimum price Jul 2025 onwards',
    fontsize=13,
)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
ax.tick_params(labelsize=9)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_params))
fig.tight_layout()
fig.savefig(out_dir / 'params_vs_price.png', dpi=150)
plt.close(fig)
print(f'\nPlot 1 saved to {out_dir / "params_vs_price.png"}')

# ---------------------------------------------------------------------------
# Plot 2: GPU-price-adjusted binscatter with OLS line + regression table
# ---------------------------------------------------------------------------

# Prepare regression variables
panel_full['log_params'] = np.log10(panel_full['params'])
panel_full['log_adj_price'] = np.log10(panel_full['adj_price'])
panel_full['week'] = panel_full['date'].dt.to_period('W').astype(str)
panel_full['author'] = panel_full['openrouter_id'].str.split('/').str[0]

y_reg = panel_full['log_adj_price'].values
log_params_reg = panel_full['log_params'].values
n_obs = len(y_reg)

# Weight each observation by 1 / (number of observations for its model)
obs_counts = panel_full.groupby('openrouter_id')['log_adj_price'].transform('count')
weights = (1.0 / obs_counts).values
cluster_ids = panel_full['openrouter_id'].values

week_dummies = pd.get_dummies(panel_full['week'], drop_first=True).astype(float).values
author_dummies = pd.get_dummies(panel_full['author'], drop_first=True).astype(float).values

ones = np.ones(n_obs)
X1 = np.column_stack([ones, log_params_reg])
X2 = np.column_stack([ones, log_params_reg, week_dummies])
X3 = np.column_stack([ones, log_params_reg, week_dummies, author_dummies])


import statsmodels.api as sm


def wls_clustered(X, y, w, clusters):
    """WLS with standard errors clustered by openrouter_id using statsmodels."""
    return sm.WLS(y, X, weights=w).fit(
        cov_type='cluster', cov_kwds={'groups': clusters}
    )


r1 = wls_clustered(X1, y_reg, weights, cluster_ids)
r2 = wls_clustered(X2, y_reg, weights, cluster_ids)
r3 = wls_clustered(X3, y_reg, weights, cluster_ids)

# log_params coefficient is at index 1 in all three models
slope = r1.params[1]  # use model 1 slope for the OLS line on the plot


def sig_stars(t_val):
    a = abs(t_val)
    if a > 2.576: return '***'
    if a > 1.960: return '**'
    if a > 1.645: return '*'
    return ''


print('\nRegression summary (dep. var: log10(adj_price)):')
for i, res in enumerate([r1, r2, r3], 1):
    print(f'  ({i}) log(params) = {res.params[1]:.3f}{sig_stars(res.tvalues[1])}  SE={res.bse[1]:.3f}  R²={res.rsquared:.3f}')

# Binscatter
x = panel_full['params'].values
y = panel_full['adj_price'].values
log_x = panel_full['log_params'].values
log_y = panel_full['log_adj_price'].values

df2 = pd.DataFrame({'log_params': log_x, 'log_adj_price': log_y})
df2['bin'] = pd.qcut(df2['log_params'], q=10, labels=False, duplicates='drop')
bins = (
    df2.groupby('bin', observed=True)
    .agg(log_x_mean=('log_params', 'mean'), log_y_mean=('log_adj_price', 'mean'))
    .reset_index()
)
bins['x_mean'] = 10 ** bins['log_x_mean']
bins['y_mean'] = 10 ** bins['log_y_mean']

x_line = np.logspace(log_x.min(), log_x.max(), 200)
y_line = 10 ** (r1.params[0] + slope * np.log10(x_line))

fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(x, y, color='steelblue', s=15, alpha=0.06, zorder=2)
ax.scatter(bins['x_mean'], bins['y_mean'],
           color='darkorange', s=80, zorder=4, edgecolors='white', linewidths=0.8)
ax.plot(x_line, y_line, color='firebrick', linewidth=1.8, zorder=3,
        label=f'OLS (1): slope = {slope:.2f}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Total parameters', fontsize=12)
ax.set_ylabel('Adj. price × H100 1-yr rate  (USD²/hr · 1M tok)', fontsize=12)
ax.set_title(
    'Open-weights inference price (GPU-adjusted) vs. parameter count\n'
    'Binscatter (10 bins), OLS fit — all OpenRouter price observations',
    fontsize=13,
)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
ax.tick_params(labelsize=9)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_params))
ax.legend(fontsize=10)

fig.tight_layout()
fig.savefig(out_dir / 'params_vs_price_binscatter.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Plot 2 saved to {out_dir / "params_vs_price_binscatter.png"}')

# ---------------------------------------------------------------------------
# Regression table (terminal)
# ---------------------------------------------------------------------------
w = 14
header = f'{"": <12}  {"(1) Baseline":>{w}}  {"(2) + Week FE":>{w}}  {"(3) + Author FE":>{w}}'
sep = '-' * len(header)
print(f'\n{sep}')
print(header)
print(sep)
print(f'{"log(params)": <12}  {f"{r1.params[1]:.3f}{sig_stars(r1.tvalues[1])}":>{w}}  {f"{r2.params[1]:.3f}{sig_stars(r2.tvalues[1])}":>{w}}  {f"{r3.params[1]:.3f}{sig_stars(r3.tvalues[1])}":>{w}}')
print(f'{"": <12}  {f"({r1.bse[1]:.3f})":>{w}}  {f"({r2.bse[1]:.3f})":>{w}}  {f"({r3.bse[1]:.3f})":>{w}}')
print(sep)
print(f'{"Week FE": <12}  {"No":>{w}}  {"Yes":>{w}}  {"Yes":>{w}}')
print(f'{"Author FE": <12}  {"No":>{w}}  {"No":>{w}}  {"Yes":>{w}}')
print(sep)
print(f'{"R²": <12}  {f"{r1.rsquared:.3f}":>{w}}  {f"{r2.rsquared:.3f}":>{w}}  {f"{r3.rsquared:.3f}":>{w}}')
print(f'{"N": <12}  {f"{n_obs:,}":>{w}}  {f"{n_obs:,}":>{w}}  {f"{n_obs:,}":>{w}}')
print(sep)
print('Clustered SEs (by openrouter_id) in parentheses. * p<0.10  ** p<0.05  *** p<0.01')
