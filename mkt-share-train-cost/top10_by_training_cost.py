import csv
import os
import re

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def parse_number(value):
    """Parse numeric values, handling empty strings, $ prefixes, and commas."""
    if not value or value.strip() == "":
        return None
    cleaned = value.strip().lstrip("$").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def format_params(params):
    if params >= 1e12:
        return f"{params / 1e12:.1f}T"
    elif params >= 1e9:
        return f"{params / 1e9:.1f}B"
    elif params >= 1e6:
        return f"{params / 1e6:.1f}M"
    return str(params)


def format_tokens(tokens):
    if tokens >= 1e12:
        return f"{tokens / 1e12:.1f}T"
    elif tokens >= 1e9:
        return f"{tokens / 1e9:.1f}B"
    elif tokens >= 1e6:
        return f"{tokens / 1e6:.1f}M"
    return str(tokens)


def format_cost(cost):
    if cost >= 1e9:
        return f"${cost / 1e9:.2f}B"
    elif cost >= 1e6:
        return f"${cost / 1e6:.1f}M"
    elif cost >= 1e3:
        return f"${cost / 1e3:.1f}K"
    return f"${cost:.0f}"


def fmt_share(val):
    return val if val is not None else "—"


# ---------------------------------------------------------------------------
# Analysis: Ramp market share normalization
# ---------------------------------------------------------------------------

RAMP_COMPANIES = {
    "Google":    "overall_google_user_share",
    "OpenAI":    "overall_openai_user_share",
    "Anthropic": "overall_anthropic_user_share",
    "DeepSeek":  "overall_deepseek_user_share",
    "xAI":       "overall_x_user_share",
}


def load_ramp_market_shares(filepath):
    """Read Ramp CSV and return a DataFrame with normalized market shares per month."""
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])

    # Strip % and convert to float
    for col in df.columns:
        if col != "Date":
            df[col] = df[col].str.rstrip("%").astype(float)

    company_cols = list(RAMP_COMPANIES.values())
    row_totals = df[company_cols].sum(axis=1)

    shares = pd.DataFrame({"Date": df["Date"]})
    for company, col in RAMP_COMPANIES.items():
        shares[company] = df[col] / row_totals * 100

    return shares


# ---------------------------------------------------------------------------
# Analysis: Menlo Ventures annual survey → monthly step series
# ---------------------------------------------------------------------------

MENLO_ANNUAL = {
    2023: {"OpenAI": 50, "Anthropic": 12, "Google":  7, "Meta": 16},
    2024: {"OpenAI": 34, "Anthropic": 24, "Google": 12, "Meta": 16},
    2025: {"OpenAI": 27, "Anthropic": 40, "Google": 21, "Meta":  8},
}


def build_menlo_series(dates):
    """Expand annual Menlo Ventures survey values to a monthly step-function DataFrame."""
    records = []
    for date in dates:
        row = {"Date": date}
        year_data = MENLO_ANNUAL.get(date.year, {})
        for company, pct in year_data.items():
            row[company] = pct
        records.append(row)
    return pd.DataFrame(records).set_index("Date")


# ---------------------------------------------------------------------------
# Plotting: Ramp normalized market share time series
# ---------------------------------------------------------------------------

COMPANY_COLORS = {
    "OpenAI":    "black",
    "Anthropic": "orange",
    "Google":    "purple",
    "xAI":       "yellow",
    "DeepSeek":  "red",
    "Meta":      "blue",
}


def find_model_events(parquet_path):
    """Return dict of {label: (company, earliest_date)} from openrouter panel."""
    import polars as pl

    MODEL_SEARCHES = [
        ("GPT-4",          "gpt-4",          "OpenAI"),
        ("GPT-5",          "gpt-5",          "OpenAI"),
        ("Grok 3",         "grok-3",         "xAI"),
        ("Claude 3 Sonnet","claude-3-sonnet", "Anthropic"),
        ("Claude Sonnet 4","claude-sonnet-4", "Anthropic"),
        ("Gemini 2.5 Pro", "gemini-2.5-pro",  "Google"),
        ("Gemini 1.0 Pro", "gemini-pro([^-]|$)", "Google"),
    ]

    df = pl.read_parquet(parquet_path).with_columns(
        pl.col("date").cast(pl.Date)
    )
    events = {}
    for label, search, company in MODEL_SEARCHES:
        match = df.filter(
            pl.col("openrouter_id").str.to_lowercase().str.contains(search.lower())
        )
        if len(match) == 0:
            print(f"  [warn] no rows found for '{search}'")
            continue
        earliest = match["date"].min()
        events[label] = (company, pd.Timestamp(earliest))
        print(f"  {label}: {earliest}")
    return events


def plot_ramp_market_shares(shares, menlo_series, output_path, model_events=None):
    """Plot normalized Ramp market share time series with Menlo overlay and save to output_path."""
    import matplotlib.lines as mlines

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for company in RAMP_COMPANIES:
        ax.plot(
            shares["Date"],
            shares[company],
            color=COMPANY_COLORS[company],
            linewidth=3,
            marker="o",
            markersize=3,
        )

    for company in menlo_series.columns:
        col = menlo_series[company].dropna()
        ax.plot(
            col.index,
            col.values,
            color=COMPANY_COLORS[company],
            linewidth=3,
            linestyle="--",
        )

    # Model release markers on each company's Ramp line
    if model_events:
        for label, (company, event_date) in model_events.items():
            if company not in shares.columns:
                continue
            # Interpolate y-value at event_date from Ramp series
            row = shares[shares["Date"] == event_date]
            if row.empty:
                # Find nearest date
                idx = (shares["Date"] - event_date).abs().idxmin()
                row = shares.loc[[idx]]
            y = row[company].values[0]
            ax.scatter(
                event_date, y,
                color=COMPANY_COLORS[company],
                s=80, zorder=5, edgecolors="white", linewidths=1,
            )
            ax.annotate(
                label, xy=(event_date, y),
                xytext=(6, 4), textcoords="offset points",
                fontsize=7, color=COMPANY_COLORS[company],
            )

    # Legend: one entry per company (color), plus style guide
    company_handles = [
        mlines.Line2D([], [], color=COMPANY_COLORS[c], linewidth=3, label=c)
        for c in list(RAMP_COMPANIES) + [c for c in menlo_series.columns if c not in RAMP_COMPANIES]
    ]
    style_handles = [
        mlines.Line2D([], [], color="gray", linewidth=3, linestyle="-",  label="Ramp AI Index (normalized)"),
        mlines.Line2D([], [], color="gray", linewidth=3, linestyle="--", label="Menlo Ventures Survey (annual)"),
    ]
    ax.legend(handles=company_handles + style_handles, loc="upper left", framealpha=0.9, fontsize=9)

    ax.set_title("AI Company Market Share Over Time", fontsize=13)
    ax.set_ylabel("Market Share (%)")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved chart to {output_path}")


# ---------------------------------------------------------------------------
# Epoch database: top 10 by training cost
# ---------------------------------------------------------------------------

with open("data/Epoch Database - Notable Models.csv", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

filtered = []
for row in rows:
    model = re.sub(r"\s+", " ", row["Model"]).strip()
    params = parse_number(row["Parameters"])
    dataset_size = parse_number(row["Training dataset size (total)"])
    cost = parse_number(row["Training compute cost (2023 USD)"])

    if cost is not None:
        pub_date_str = row["Publication date"].strip()
        try:
            pub_date = pd.to_datetime(pub_date_str)
        except Exception:
            pub_date = None
        filtered.append(
            {
                "model": model,
                "organization": row["Organization"].strip(),
                "publication_date": pub_date_str,
                "pub_date": pub_date,
                "parameters": params,
                "dataset_size": dataset_size,
                "cost": cost,
            }
        )

top10 = sorted(filtered, key=lambda x: x["cost"], reverse=True)[:10]

# ---------------------------------------------------------------------------
# Ramp analysis + plot
# ---------------------------------------------------------------------------

ramp_shares = load_ramp_market_shares("data/ramp-data-Dq5pU.csv")
menlo_series = build_menlo_series(ramp_shares["Date"])
model_events = find_model_events("data/openrouter_panel.parquet")
plot_ramp_market_shares(ramp_shares, menlo_series, "mkt-share-train-cost/ramp_market_share.png", model_events=model_events)

# ---------------------------------------------------------------------------
# Build markdown
# ---------------------------------------------------------------------------

def build_top10_table(models):
    """Return markdown lines for a top-10-by-cost table given a list of model dicts."""
    rows = []
    rows.append("| Rank | Model | Organization | Publication Date | Parameters | Training Dataset Size | Training Cost (2023 USD) |")
    rows.append("|------|-------|--------------|-----------------|------------|----------------------|--------------------------|")
    for i, m in enumerate(sorted(models, key=lambda x: x["cost"], reverse=True)[:10], 1):
        params_str = format_params(m["parameters"]) if m["parameters"] is not None else "-"
        dataset_str = format_tokens(m["dataset_size"]) if m["dataset_size"] is not None else "-"
        rows.append(
            f"| {i} | {m['model']} | {m['organization']} | {m['publication_date']} "
            f"| {params_str} | {dataset_str} | {format_cost(m['cost'])} |"
        )
    return rows


lines = [
    "# Top 10 AI Models by Training Cost",
    "",
    "Models ranked by training compute cost (2023 USD), filtered to those with a known training cost.",
    "Parameters and Training Dataset Size are shown where available.",
    "",
]
lines += build_top10_table(filtered)
lines.append("")
lines.append(f"*Data source: Epoch Database – Notable Models. {len(filtered)} models had a known training cost.*")

# Artificial Analysis inference API survey — raw adoption rates, normalized to sum to 100%
_aa_raw = {
    "OpenAI": 84, "Google": 80, "Anthropic": 67, "DeepSeek": 53, "Meta": 42,
    "xAI": 31, "Alibaba": 25, "Mistral": 22, "Perplexity": 20, "Microsoft": 14,
    "Amazon": 9, "NVIDIA": 7, "Cohere": 7, "Reka": 3, "AI21": 2, "Other": 2,
}
_aa_total = sum(_aa_raw.values())
_aa_normalized = {co: v / _aa_total * 100 for co, v in _aa_raw.items()}

def aa_share(company):
    """Return formatted normalized Artificial Analysis share, or None if not in survey."""
    v = _aa_normalized.get(company)
    return f"{v:.1f}%" if v is not None else None

# Market share snapshot table
# (Company, Menlo Ventures, Demirer et al. OpenRouter, Ramp AI Index, a16z CIO Survey, Artificial Analysis)
market_share = [
    ("Anthropic", "40%",   "4.84%",  "37.3%", "17%"),
    ("OpenAI",    "27%",   "7.37%",  "52.5%", "56%"),
    ("Google",    "21%",   "25.76%", "7.18%", "16%"),
    ("Meta",      "8%",    None,     None,    "5%"),
    ("xAI",       None,    "32.22%", "2.9%",  None),
    ("DeepSeek",  None,    None,     "0.2%",  None),
]

lines += [
    "",
    "## AI Company Market Share by Survey Source",
    "",
    "| Company | Menlo Ventures Survey (Nov 2025) | Demirer et al. OpenRouter tokens (Dec 2025) | Ramp AI Index normalized (Jan 2026) | a16z CIO Survey LLM spend (Jan 2026) | Artificial Analysis normalized (H1 2025) |",
    "|---------|----------------------------------|---------------------------------------------|-------------------------------------|--------------------------------------|------------------------------------------|",
]

for company, menlo, openrouter, ramp, a16z in market_share:
    lines.append(
        f"| {company} | {fmt_share(menlo)} | {fmt_share(openrouter)} | {fmt_share(ramp)} "
        f"| {fmt_share(a16z)} | {fmt_share(aa_share(company))} |"
    )

lines += [
    "",
    "*Sources: Menlo Ventures Survey; Demirer et al. OpenRouter token-based data; "
    "Ramp AI Index (shares normalized from summed adoption across reported companies); "
    "Andreessen Horowitz CIO Survey (LLM spend); "
    "Artificial Analysis inference API survey (adoption rates normalized to sum to 100% across all reported providers).*",
]

# Ramp time series section
lines += [
    "",
    "## Ramp AI Index: Normalized Market Share Over Time",
    "",
    "Each company's monthly share of AI spend on the Ramp platform, normalized so that the five "
    "tracked companies (Google, OpenAI, Anthropic, DeepSeek, xAI) sum to 100% in each month.",
    "",
    "![Ramp normalized market share time series](ramp_market_share.png)",
    "",
    "*Source: Ramp AI Index (`ramp-data-Dq5pU.csv`). Normalization: each company's raw user-share "
    "divided by the sum of all five companies' raw shares for that month.*",
]

for year in [2023, 2024, 2025]:
    year_models = [m for m in filtered if m["pub_date"] is not None and m["pub_date"].year == year]
    lines += [
        "",
        f"## Top 10 AI Models by Training Cost — {year}",
        "",
        f"*{len(year_models)} models released in {year} had a known training cost.*",
        "",
    ]
    lines += build_top10_table(year_models)

output = "\n".join(lines)

with open("mkt-share-train-cost/top10_training_cost.md", "w", encoding="utf-8") as f:
    f.write(output)

print("Wrote top10_training_cost.md")

# ---------------------------------------------------------------------------
# Export CSVs to data/
# ---------------------------------------------------------------------------

os.makedirs("data", exist_ok=True)

def models_to_df(models):
    top = sorted(models, key=lambda x: x["cost"], reverse=True)[:10]
    return pd.DataFrame([
        {
            "Rank": i,
            "Model": m["model"],
            "Organization": m["organization"],
            "Publication Date": m["publication_date"],
            "Parameters": format_params(m["parameters"]) if m["parameters"] is not None else "",
            "Training Dataset Size": format_tokens(m["dataset_size"]) if m["dataset_size"] is not None else "",
            "Training Cost (2023 USD)": format_cost(m["cost"]),
        }
        for i, m in enumerate(top, 1)
    ])

models_to_df(filtered).to_csv("data/top10_overall.csv", index=False)
print("Wrote data/top10_overall.csv")

for year in [2023, 2024, 2025]:
    year_models = [m for m in filtered if m["pub_date"] is not None and m["pub_date"].year == year]
    models_to_df(year_models).to_csv(f"data/top10_{year}.csv", index=False)
    print(f"Wrote data/top10_{year}.csv")

# Market share snapshot
market_share_rows = []
for company, menlo, openrouter, ramp, a16z in market_share:
    market_share_rows.append({
        "Company": company,
        "Menlo Ventures Survey (Nov 2025)": fmt_share(menlo),
        "Demirer et al. OpenRouter tokens (Dec 2025)": fmt_share(openrouter),
        "Ramp AI Index normalized (Jan 2026)": fmt_share(ramp),
        "a16z CIO Survey LLM spend (Jan 2026)": fmt_share(a16z),
        "Artificial Analysis normalized (H1 2025)": fmt_share(aa_share(company)),
    })
pd.DataFrame(market_share_rows).to_csv("data/market_share_snapshot.csv", index=False)
print("Wrote data/market_share_snapshot.csv")

# Ramp normalized time series
ramp_shares.to_csv("data/ramp_market_share_normalized.csv", index=False)
print("Wrote data/ramp_market_share_normalized.csv")
