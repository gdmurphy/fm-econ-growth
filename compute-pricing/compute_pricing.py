import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PRODUCT_SLUG = "n-h100-sxm-80g"
COUNTRY = "US"
PRICE_TYPES = [12, 24, 36]
PRICE_TYPE_LABELS = {12: "1 year", 24: "2 year", 36: "3 year"}
PROVIDER_SOURCE_TYPE = "Average"

FILES = [
    "compute-pricing/tier2_2024_rental.csv",
    "compute-pricing/tier2_2025_rental.csv",
]


def load_and_filter(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])

    # 2024 file predates provider_source_type column; treat all rows as Average
    if "provider_source_type" not in df.columns:
        df["provider_source_type"] = PROVIDER_SOURCE_TYPE

    mask = (
        (df["product_slug"] == PRODUCT_SLUG)
        & (df["country"] == COUNTRY)
        & (df["price_type"].isin(PRICE_TYPES))
        & (df["provider_source_type"] == PROVIDER_SOURCE_TYPE)
    )
    return df.loc[mask, ["date", "price_type", "price"]].copy()


frames = [load_and_filter(f) for f in FILES]
data = pd.concat(frames, ignore_index=True)

# Check for duplicates on (date, price_type)
dupes = data[data.duplicated(subset=["date", "price_type"], keep=False)]
if not dupes.empty:
    raise ValueError(
        f"Duplicate (date, price_type) rows found:\n{dupes.to_string(index=False)}"
    )

# Warn if any requested price type has no data
for pt in PRICE_TYPES:
    if data[data["price_type"] == pt].empty:
        print(f"Warning: no data found for price_type={pt} ({PRICE_TYPE_LABELS[pt]})")

# SemiAnalysis H100 SXM 80G 1-year prices
# H1/H2 2023 are half-year estimates; quarterly thereafter until 2025-07; monthly from 2025-07
SEMI_DATES = [
    "2023-01-01", "2023-07-01",                          # H1 / H2 2023
    "2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01",  # Q1–Q4 2024
    "2025-01-01", "2025-04-01",                           # Q1–Q2 2025
    "2025-07-01", "2025-08-01", "2025-09-01", "2025-10-01",  # monthly 2025-07 …
    "2025-11-01", "2025-12-01",
    "2026-01-01", "2026-02-01", "2026-03-01",             # … 2026-03
]
SEMI_PRICES = [
    3.05, 2.97,
    2.80, 2.35, 2.30, 2.00,
    1.95, 1.95,
    1.85, 1.75, 1.75, 1.70,
    1.73, 1.73,
    1.77, 2.08, 2.35,
]
semi_dates = pd.to_datetime(SEMI_DATES)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

has_any_data = False
for pt in PRICE_TYPES:
    subset = data[data["price_type"] == pt].sort_values("date")
    if subset.empty:
        continue
    has_any_data = True
    ax.plot(subset["date"], subset["price"], marker=".", linewidth=1.5,
            label=f"price_type {pt} ({PRICE_TYPE_LABELS[pt]})")

ax.plot(semi_dates, SEMI_PRICES, marker="s", linewidth=1.5, linestyle="--",
        color="tab:red", label="SemiAnalysis 12-month")

if not has_any_data:
    ax.text(0.5, 0.5, "No data available for requested price types",
            ha="center", va="center", transform=ax.transAxes, fontsize=13)

ax.set_title(f"{PRODUCT_SLUG} — {COUNTRY} — Reserved pricing over time\n"
             f"(provider_source_type: {PROVIDER_SOURCE_TYPE})")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD/hr)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45, ha="right")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("compute_pricing.png", dpi=150)
plt.show()
print("Saved compute_pricing.png")
