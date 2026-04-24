import pandas as pd
import matplotlib.pyplot as plt
from set_globals import *

df = pd.read_parquet(data_dir / 'openrouter_panel.parquet')
df['year'] = df['date'].dt.year

# De-overlap rolling 7-day token windows.
# Each snapshot S(D) covers [D-7, D-1]. Weight = min(gap_to_prev, 7) / 7.
# First observation per model gets weight 1.0 (full window attributed, no prior to compare).
# Gaps > 7 days: windows are disjoint, cap at 7 → weight 1.0.
_obs = (
    df[df['total_prompt_tokens'].notna()]
    .sort_values(['openrouter_id', 'date'])[['openrouter_id', 'date']]
    .copy()
)
_obs['_prev_date'] = _obs.groupby('openrouter_id')['date'].shift(1)
_obs['deoverlap_weight'] = (
    (_obs['date'] - _obs['_prev_date'])
    .dt.days.fillna(7).clip(upper=7).div(7.0)
)
df = df.merge(_obs[['openrouter_id', 'date', 'deoverlap_weight']], on=['openrouter_id', 'date'], how='left')
df['attributed_prompt_tokens'] = df['total_prompt_tokens'] * df['deoverlap_weight']
df['attributed_completion_tokens'] = df['total_completion_tokens'] * df['deoverlap_weight']
df['attributed_reasoning_tokens'] = df['total_native_tokens_reasoning'] * df['deoverlap_weight']

# Forward-fill prices within each model so every attributed-token row has a price
df = df.sort_values(['openrouter_id', 'date'])
df['prompt_price_ffill'] = df.groupby('openrouter_id')['prompt_price'].ffill()
df['completion_price_ffill'] = df.groupby('openrouter_id')['completion_price'].ffill()
df['prompt_revenue'] = df['attributed_prompt_tokens'] * df['prompt_price_ffill']
df['completion_revenue'] = df['attributed_completion_tokens'] * df['completion_price_ffill']
df['attributed_revenue'] = df['prompt_revenue'].fillna(0) + df['completion_revenue'].fillna(0)
df['date_notz'] = df['date'].dt.tz_localize(None)

output_dir = Path(__file__).parent
plots_dir = output_dir / 'plots'
plots_dir.mkdir(exist_ok=True)

# OpenRouter's assumed share of the North American AI market
OPENROUTER_NA_SHARE = 0.003
OPENROUTER_NA_SHARE_SUP = 0.0106

price_cols = ['prompt_price', 'completion_price', 'internal_reasoning_price']
cols = [
    'revenue',
    'attributed_prompt_tokens',
    'attributed_completion_tokens',
    'prompt_completion_ratio',
    'prompt_completion_reasoning_ratio',
    'attributed_reasoning_tokens',
    'prompt_price',
    'completion_price',
    'internal_reasoning_price',
]
int_cols = [c for c in cols if c not in price_cols]

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.6f}'.format)


def calc_total_revenue(data):
    m = (
        data.groupby('openrouter_id')[['attributed_prompt_tokens', 'attributed_completion_tokens']]
        .sum()
        .join(data.groupby('openrouter_id')[['prompt_price', 'completion_price']].last())
    )
    m['revenue'] = m['attributed_prompt_tokens'] * m['prompt_price'] + m['attributed_completion_tokens'] * m['completion_price']
    return m['revenue'].sum()


def build_top10(data):
    t = (
        data.groupby('openrouter_id')[['attributed_prompt_tokens', 'attributed_completion_tokens', 'attributed_reasoning_tokens']]
        .sum()
        .join(data.groupby('openrouter_id')[['prompt_price', 'completion_price', 'internal_reasoning_price']].last())
        .reset_index()
    )
    t['revenue'] = t['attributed_prompt_tokens'] * t['prompt_price'] + t['attributed_completion_tokens'] * t['completion_price']
    t['prompt_completion_ratio'] = t['attributed_prompt_tokens'] / t['attributed_completion_tokens']
    t['prompt_completion_reasoning_ratio'] = t['attributed_prompt_tokens'] / (t['attributed_completion_tokens'] + t['attributed_reasoning_tokens'])
    t = t.sort_values('revenue', ascending=False).head(10).reset_index(drop=True)
    out = t[['openrouter_id'] + cols].copy()
    out[int_cols] = out[int_cols].round(0).astype('Int64')
    return out


def build_pie(data, title, filename, total_revenue):
    m = (
        data.groupby('openrouter_id')[['attributed_prompt_tokens', 'attributed_completion_tokens']]
        .sum()
        .join(data.groupby('openrouter_id')[['prompt_price', 'completion_price', 'openrouter_author']].last())
    )
    m['revenue'] = m['attributed_prompt_tokens'] * m['prompt_price'] + m['attributed_completion_tokens'] * m['completion_price']
    author_rev = m.groupby('openrouter_author')['revenue'].sum().sort_values(ascending=False)
    top = author_rev[author_rev / author_rev.sum() >= 0.01]
    other = author_rev[author_rev / author_rev.sum() < 0.01].sum()
    pie_data = pd.concat([top, pd.Series({'other': other})])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
    ax.set_title(f'{title}\nTotal Revenue: ${total_revenue:,.0f}')
    plt.tight_layout()
    path = plots_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved chart to {path}')


def table_section(label, data, chart_filename, total_revenue):
    out = build_top10(data)
    print(f'\n--- {label} (Total Revenue: ${total_revenue:,.0f}) ---')
    print(out.to_string(index=False))
    md = f'\n## Top 10 Models by Revenue — {label}\n\n'
    md += f'**Total estimated revenue across all models: ${total_revenue:,.0f}**\n\n'
    md += out.to_markdown(index=False, floatfmt='.6f')
    md += f'\n\n### Revenue Market Share by Author — {label}\n\n'
    md += 'Authors accounting for less than 1% of total revenue are grouped into "Other".\n\n'
    md += f'![{label} Revenue Market Share](plots/{chart_filename})\n'
    return md


# Compute total revenues
date_min = df['date'].min().strftime('%B %Y')
date_max = df['date'].max().strftime('%B %Y')
total_rev_all = calc_total_revenue(df)

year_revenues = {}
for year in [2023, 2024, 2025, 2026]:
    df_year = df[df['year'] == year]
    rev = calc_total_revenue(df_year)
    if not df_year.empty and rev > 0:
        year_revenues[year] = rev

# Build markdown
md = f"""\
# OpenRouter Usage & Revenue

Revenue is estimated as total prompt tokens × prompt price + total completion tokens × completion price,
using each model's most recently observed prices within the period. Prices are in USD per token.
Token counts are cumulative across all dates in each period.

"""

md += table_section(f'All Time ({date_min}–{date_max})', df, 'market_share_all.png', total_rev_all)
build_pie(df, 'OpenRouter Revenue Market Share (All Time)', 'market_share_all.png', total_rev_all)

for year in [2023, 2024, 2025, 2026]:
    if year not in year_revenues:
        continue
    df_year = df[df['year'] == year]
    rev = year_revenues[year]
    md += table_section(str(year), df_year, f'market_share_{year}.png', rev)
    build_pie(df_year, f'OpenRouter Revenue Market Share ({year})', f'market_share_{year}.png', rev)

# Compute annualized revenues
import calendar
has_usage = df['attributed_prompt_tokens'].notna() & (df['attributed_prompt_tokens'] > 0)
year_annualized = {}
year_coverage_days = {}
for year, grp in df[has_usage].groupby('year'):
    if year not in year_revenues:
        continue
    year_start = pd.Timestamp(f'{year}-01-01', tz='UTC')
    year_end   = pd.Timestamp(f'{year}-12-31', tz='UTC')
    coverage_start = max(year_start, grp['date'].min() - pd.Timedelta(days=7))
    coverage_end   = min(year_end,   grp['date'].max() - pd.Timedelta(days=1))
    effective_days = (coverage_end - coverage_start).days + 1
    days_in_year = 366 if calendar.isleap(year) else 365
    year_coverage_days[year] = (effective_days, days_in_year)
    year_annualized[year] = year_revenues[year] * days_in_year / effective_days

# Annual revenue bar chart (raw)
fig, ax = plt.subplots(figsize=(8, 5))
years = list(year_revenues.keys())
revs = [year_revenues[y] for y in years]
bars = ax.bar([str(y) for y in years], revs, color='steelblue')
ax.bar_label(bars, labels=[f'${r:,.0f}' for r in revs], padding=4, fontsize=9)
ax.set_title('OpenRouter Estimated Annual Revenue')
ax.set_xlabel('Year')
ax.set_ylabel('Revenue (USD)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.tight_layout()
bar_path = plots_dir / 'annual_revenue.png'
fig.savefig(bar_path, dpi=150)
plt.close(fig)
print(f'Saved chart to {bar_path}')

# Annualized revenue bar chart
fig, ax = plt.subplots(figsize=(8, 5))
years_ann = list(year_annualized.keys())
revs_ann = [year_annualized[y] for y in years_ann]
bar_labels = [f'${r:,.0f}\n({year_coverage_days[y][0]}/{year_coverage_days[y][1]} days)' for y, r in zip(years_ann, revs_ann)]
bars = ax.bar([str(y) for y in years_ann], revs_ann, color='steelblue')
ax.bar_label(bars, labels=bar_labels, padding=4, fontsize=8)
ax.set_title('OpenRouter Estimated Annualized Revenue\n(scaled to full year based on days with data)')
ax.set_xlabel('Year')
ax.set_ylabel('Annualized Revenue (USD)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_ylim(0, max(revs_ann) * 1.25)
plt.tight_layout()
ann_bar_path = plots_dir / 'annual_revenue_annualized.png'
fig.savefig(ann_bar_path, dpi=150)
plt.close(fig)
print(f'Saved chart to {ann_bar_path}')

md += """\

## Annual Revenue

![OpenRouter Annual Revenue](plots/annual_revenue.png)

## Annualized Revenue

The chart below scales each year's revenue to a full year based on the calendar days actually covered by the data
(from the start of the first 7-day window to the end of the last 7-day window, clipped to the year).

![OpenRouter Annualized Revenue](plots/annual_revenue_annualized.png)
"""

# --- Regional revenue time series ---
regions = pd.read_csv(data_dir / 'spend_volumes_by_region_extracted.csv', parse_dates=['Week'])
regions = regions.set_index('Week').sort_index()

# Interpolate weekly shares to daily, then average by calendar month
regions_daily = regions.resample('D').interpolate('linear')
regions_monthly = regions_daily.resample('MS').mean()
# Normalise to fractions (rows should already sum to ~100)
regions_monthly = regions_monthly.div(regions_monthly.sum(axis=1), axis=0)

monthly_total = df.groupby(df['date_notz'].dt.to_period('M'))['attributed_revenue'].sum()
monthly_total.index = monthly_total.index.to_timestamp()

common_months = monthly_total.index.intersection(regions_monthly.index)
regional_rev = regions_monthly.loc[common_months].multiply(monthly_total.loc[common_months], axis=0)

fig, ax = plt.subplots(figsize=(12, 6))
for col in regional_rev.columns:
    ax.plot(regional_rev.index, regional_rev[col], marker='o', markersize=3, label=col)
ax.set_title('OpenRouter Estimated Monthly Revenue by Region')
ax.set_xlabel('Month')
ax.set_ylabel('Revenue (USD)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
regional_path = plots_dir / 'regional_revenue.png'
fig.savefig(regional_path, dpi=150)
plt.close(fig)
print(f'Saved chart to {regional_path}')

md += """\

## Monthly Revenue by Region

Regional shares are from weekly survey data, linearly interpolated to monthly.
Total monthly revenue is split by region using these shares.

![OpenRouter Monthly Revenue by Region](plots/regional_revenue.png)
"""

# --- Implied total NA AI market time series ---
monthly_na_rev = regional_rev['North America']  # already in USD
implied_low  = monthly_na_rev / OPENROUTER_NA_SHARE       * 12  # annualized
implied_high = monthly_na_rev / OPENROUTER_NA_SHARE_SUP   * 12

# Subscription-anchor line: derive global annualized subscription revenue in Jul 2025
# from OpenAI subs data, scale over time using OpenRouter revenue growth, apply NA share.
OPENAI_SUBS            = 35e6
OPENAI_PREMIUM_SHARE   = 0.058
OPENAI_PREMIUM_PRICE   = 200   # $/month
OPENAI_STANDARD_PRICE  = 20    # $/month

ramp = pd.read_csv(data_dir / 'ramp_market_share_normalized.csv', parse_dates=['Date'])
openai_share_jul2025 = ramp.loc[ramp['Date'] == '2025-07-01', 'OpenAI'].values[0] / 100.0

openai_monthly_rev = OPENAI_SUBS * (
    OPENAI_PREMIUM_SHARE * OPENAI_PREMIUM_PRICE +
    (1 - OPENAI_PREMIUM_SHARE) * OPENAI_STANDARD_PRICE
)
global_sub_rev_annual_jul2025 = openai_monthly_rev / openai_share_jul2025 * 12  # ~18.6B

# Scale anchor to each month using OpenRouter revenue as growth proxy, then apply NA share
jul2025 = pd.Timestamp('2025-07-01')
openrouter_monthly_total = monthly_total.copy()
openrouter_jul2025 = openrouter_monthly_total.get(jul2025, float('nan'))

sub_anchor_na = (
    global_sub_rev_annual_jul2025
    * (openrouter_monthly_total / openrouter_jul2025)
    * regions_monthly.loc[openrouter_monthly_total.index.intersection(regions_monthly.index), 'North America']
)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(implied_low.index,  implied_low.values,  marker='o', markersize=4,
        label=f'OpenRouter NA share = {OPENROUTER_NA_SHARE:.1%}')
ax.plot(implied_high.index, implied_high.values, marker='o', markersize=4,
        label=f'OpenRouter NA share = {OPENROUTER_NA_SHARE_SUP:.2%}')
ax.plot(sub_anchor_na.index, sub_anchor_na.values, marker='s', markersize=4,
        linestyle='--', color='seagreen',
        label=f'Subscription anchor (${global_sub_rev_annual_jul2025/1e9:.1f}B global Jul 2025)')
ax.set_title('Implied Total NA AI Market (Annualized)\nDerived from OpenRouter Monthly NA Revenue')
ax.set_xlabel('Month')
ax.set_ylabel('Annualized Market Size (USD)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v/1e9:.1f}B'))
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
implied_path = plots_dir / 'implied_na_market.png'
fig.savefig(implied_path, dpi=150)
plt.close(fig)
print(f'Saved chart to {implied_path}')

md += f"""\

## Implied Total NA AI Market Over Time

Monthly NA revenue = total OpenRouter revenue × NA regional share.
Annualized market = (monthly NA revenue / assumed OpenRouter NA share) × 12.
Two scenarios: {OPENROUTER_NA_SHARE:.1%} (base) and {OPENROUTER_NA_SHARE_SUP:.2%} (upper bound).
Subscription anchor: ${global_sub_rev_annual_jul2025/1e9:.1f}B global annualized revenue in July 2025,
derived from OpenAI ({OPENAI_SUBS/1e6:.0f}M subs, {OPENAI_PREMIUM_SHARE:.1%} at ${OPENAI_PREMIUM_PRICE}/mo)
at {openai_share_jul2025:.1%} market share, scaled by OpenRouter revenue growth and NA regional share.

![Implied NA AI Market](plots/implied_na_market.png)
"""

# --- Model-level market share CSV ---
model_stats = df.groupby(['openrouter_id', 'openrouter_author']).agg(
    attributed_prompt_tokens=('attributed_prompt_tokens', 'sum'),
    attributed_completion_tokens=('attributed_completion_tokens', 'sum'),
    prompt_revenue=('prompt_revenue', 'sum'),
    completion_revenue=('completion_revenue', 'sum'),
).reset_index()
model_stats['revenue'] = model_stats['prompt_revenue'] + model_stats['completion_revenue']
model_stats['market_share'] = model_stats['revenue'] / model_stats['revenue'].sum()
model_stats['prompt_price_qwavg'] = model_stats['prompt_revenue'] / model_stats['attributed_prompt_tokens']
model_stats['completion_price_qwavg'] = model_stats['completion_revenue'] / model_stats['attributed_completion_tokens']
model_stats['prompt_completion_ratio'] = model_stats['attributed_prompt_tokens'] / model_stats['attributed_completion_tokens']
model_stats = model_stats.sort_values('market_share', ascending=False).reset_index(drop=True)
model_csv = model_stats[[
    'openrouter_id', 'openrouter_author', 'market_share',
    'attributed_prompt_tokens', 'attributed_completion_tokens', 'prompt_completion_ratio',
    'prompt_price_qwavg', 'completion_price_qwavg', 'revenue',
]]
model_csv_path = output_dir / 'openrouter_model_market_share.csv'
model_csv.to_csv(model_csv_path, index=False)
print(f'Saved model market share to {model_csv_path}')

# --- Author-level market share CSV (aggregated from model stats) ---
author_stats = model_stats.groupby('openrouter_author').agg(
    attributed_prompt_tokens=('attributed_prompt_tokens', 'sum'),
    attributed_completion_tokens=('attributed_completion_tokens', 'sum'),
    prompt_revenue=('prompt_revenue', 'sum'),
    completion_revenue=('completion_revenue', 'sum'),
).reset_index()
author_stats['revenue'] = author_stats['prompt_revenue'] + author_stats['completion_revenue']
author_stats['market_share'] = author_stats['revenue'] / author_stats['revenue'].sum()
author_stats['prompt_price_qwavg'] = author_stats['prompt_revenue'] / author_stats['attributed_prompt_tokens']
author_stats['completion_price_qwavg'] = author_stats['completion_revenue'] / author_stats['attributed_completion_tokens']
author_stats['prompt_completion_ratio'] = author_stats['attributed_prompt_tokens'] / author_stats['attributed_completion_tokens']
author_stats = author_stats.sort_values('market_share', ascending=False).reset_index(drop=True)
author_csv = author_stats[[
    'openrouter_author', 'market_share',
    'attributed_prompt_tokens', 'attributed_completion_tokens', 'prompt_completion_ratio',
    'prompt_price_qwavg', 'completion_price_qwavg', 'revenue',
]]
author_csv_path = output_dir / 'openrouter_author_market_share.csv'
author_csv.to_csv(author_csv_path, index=False)
print(f'Saved author market share to {author_csv_path}')

# --- Weekly author market share CSV ---
df['week'] = df['date_notz'].dt.to_period('W').dt.start_time
weekly_author = df.groupby(['week', 'openrouter_author'])[['prompt_revenue', 'completion_revenue']].sum()
weekly_author['revenue'] = weekly_author['prompt_revenue'] + weekly_author['completion_revenue']
weekly_author = weekly_author.drop(columns=['prompt_revenue', 'completion_revenue']).reset_index()
weekly_total = weekly_author.groupby('week')['revenue'].transform('sum')
weekly_author['market_share'] = weekly_author['revenue'] / weekly_total
weekly_author = weekly_author.sort_values(['week', 'market_share'], ascending=[True, False]).reset_index(drop=True)
weekly_author_path = data_dir / 'weekly_author_market_share.csv'
weekly_author.to_csv(weekly_author_path, index=False)
print(f'Saved weekly author market share to {weekly_author_path}')

# --- Prompt vs completion tokens scatterplot ---
import numpy as np

scatter_data = model_stats[
    model_stats['attributed_prompt_tokens'].notna() &
    model_stats['attributed_completion_tokens'].notna() &
    (model_stats['attributed_prompt_tokens'] > 0) &
    (model_stats['attributed_completion_tokens'] > 0)
]
x = scatter_data['attributed_completion_tokens'].values
y = scatter_data['attributed_prompt_tokens'].values

m_fit, b_fit = np.polyfit(x, y, 1)

top_prompt = set(scatter_data.nlargest(3, 'attributed_prompt_tokens')['openrouter_id'])
top_completion = set(scatter_data.nlargest(3, 'attributed_completion_tokens')['openrouter_id'])
label_ids = top_prompt | top_completion

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(x, y, alpha=0.5, s=20, color='steelblue')
x_line = np.array([x.min(), x.max()])
ax.plot(x_line, m_fit * x_line + b_fit, color='tomato', linewidth=1.5,
        label=f'y = {m_fit:.2f}x + {b_fit:.2e}')
for _, row in scatter_data[scatter_data['openrouter_id'].isin(label_ids)].iterrows():
    ax.annotate(
        row['openrouter_id'].split('/')[-1],
        xy=(row['attributed_completion_tokens'], row['attributed_prompt_tokens']),
        xytext=(6, 3), textcoords='offset points', fontsize=7, alpha=0.85,
    )
ax.set_xlabel('Attributed Completion Tokens (billions)')
ax.set_ylabel('Attributed Prompt Tokens (billions)')
ax.set_title('Prompt vs Completion Token Usage by Model (All Time)')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1e9:.1f}B'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1e9:.1f}B'))
ax.legend()
plt.tight_layout()
scatter_path = plots_dir / 'prompt_vs_completion_scatter.png'
fig.savefig(scatter_path, dpi=150)
plt.close(fig)
print(f'Saved chart to {scatter_path}')

md += f"""\

## Prompt vs Completion Token Usage

Each point is one model. Line of best fit: y = {m_fit:.2f}x + {b_fit:.2e}.

![Prompt vs Completion Tokens](plots/prompt_vs_completion_scatter.png)
"""

# --- North America market size estimates by author ---
# OpenRouter NA revenue is assumed to be 0.3% of total NA AI market.

regions = pd.read_csv(data_dir / 'spend_volumes_by_region_extracted.csv', parse_dates=['Week'])

halves = {
    'H1 2025': (pd.Timestamp('2025-01-01'), pd.Timestamp('2025-06-30')),
    'H2 2025': (pd.Timestamp('2025-07-01'), pd.Timestamp('2025-12-31')),
}

md += '\n## North America Market Size Estimates by Author\n\n'
md += ('Revenue share assumed to equal OpenRouter usage share. '
       'NA revenue = author OpenRouter revenue × avg NA share. '
       'Total market estimate = NA revenue / 0.3% (assumed OpenRouter share of NA market).\n')

for half_label, (h_start, h_end) in halves.items():
    # Average NA share for this period from weekly survey data
    mask = (regions['Week'] >= h_start) & (regions['Week'] <= h_end)
    avg_na_share = regions.loc[mask, 'North America'].mean() / 100.0

    # Revenue per author for this period
    df_half = df[(df['date_notz'] >= h_start) & (df['date_notz'] <= h_end)]
    half_author = df_half.groupby('openrouter_author')[['prompt_revenue', 'completion_revenue']].sum()
    half_author['openrouter_revenue'] = half_author['prompt_revenue'] + half_author['completion_revenue']
    half_author = half_author[half_author['openrouter_revenue'] > 0].copy()

    half_author['na_revenue'] = half_author['openrouter_revenue'] * avg_na_share
    half_author['total_market_estimate'] = half_author['na_revenue'] / OPENROUTER_NA_SHARE
    half_author['market_share'] = half_author['openrouter_revenue'] / half_author['openrouter_revenue'].sum()
    half_author = half_author.sort_values('total_market_estimate', ascending=False).reset_index()

    # CSV
    csv_cols = ['openrouter_author', 'openrouter_revenue', 'market_share', 'na_revenue', 'total_market_estimate']
    csv_path = data_dir / f'na_market_estimate_{half_label.replace(" ", "_").lower()}.csv'
    half_author[csv_cols].to_csv(csv_path, index=False)
    print(f'Saved {csv_path}')

    # Pie chart — collapse <1% authors into Other
    total = half_author['total_market_estimate'].sum()
    top_mask = half_author['total_market_estimate'] / total >= 0.01
    pie_top = half_author[top_mask].set_index('openrouter_author')['total_market_estimate']
    other_val = half_author[~top_mask]['total_market_estimate'].sum()
    if other_val > 0:
        pie_top = pd.concat([pie_top, pd.Series({'Other': other_val})])

    def _pie_label(pct, values):
        val = pct / 100.0 * values.sum()
        return f'${val:,.0f}\n({pct:.1f}%)'

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.pie(
        pie_top,
        labels=pie_top.index,
        autopct=lambda pct: _pie_label(pct, pie_top),
        startangle=140,
        textprops={'fontsize': 8},
    )
    ax.set_title(
        f'Estimated Total AI Market by Author — {half_label}\n'
        f'(avg NA share: {avg_na_share:.1%}, OpenRouter NA share assumption: {OPENROUTER_NA_SHARE:.1%})\n'
        f'Total estimated market: ${total:,.0f}'
    )
    plt.tight_layout()
    chart_filename = f'na_market_estimate_{half_label.replace(" ", "_").lower()}.png'
    chart_path = plots_dir / chart_filename
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f'Saved chart to {chart_path}')

    md += f'\n### {half_label} (avg NA share: {avg_na_share:.1%})\n\n'
    md += f'**Total estimated NA AI market: ${total:,.0f}**\n\n'
    md += f'![NA Market Estimate {half_label}](plots/{chart_filename})\n'

md_path = output_dir / 'top10_openrouter_usage.md'
md_path.write_text(md)
print(f'\nSaved markdown to {md_path}')
