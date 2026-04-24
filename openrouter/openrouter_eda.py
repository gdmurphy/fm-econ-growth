import polars as pl

df = pl.read_parquet("data/openrouter_panel.parquet")

# ── Basic shape ──────────────────────────────────────────────────────────────
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"Date range: {df['date'].min()} → {df['date'].max()}")
print(f"Unique models (openrouter_id): {df['openrouter_id'].n_unique():,}")
print(f"Unique authors:                {df['openrouter_author'].n_unique():,}")
print(f"Unique endpoint providers:     {df['endpoint_provider'].n_unique():,}")
print()

# ── Null rates for key columns ────────────────────────────────────────────────
key_cols = [
    "net_revenue", "volume", "count",
    "total_prompt_tokens", "total_completion_tokens",
    "prompt_price", "completion_price",
    "huggingface_id",
]
print("Null rates (key columns):")
print(
    df.select([pl.col(c).is_null().mean().alias(c) for c in key_cols])
    .unpivot(variable_name="column", value_name="null_rate")
    .sort("null_rate", descending=True)
)
print()

# ── Usage coverage ────────────────────────────────────────────────────────────
has_volume = df.filter(pl.col("volume").is_not_null())
print(f"Rows with volume data: {len(has_volume):,} ({len(has_volume)/len(df):.1%})")
print(f"Total volume (tokens): {has_volume['volume'].sum():,.0f}")
print(f"Total net_revenue ($): {has_volume['net_revenue'].sum():,.2f}")
print()

# ── Top 15 models by total tokens ────────────────────────────────────────────
print("Top 15 models by total tokens:")
print(
    has_volume
    .group_by("openrouter_id")
    .agg(
        pl.col("total_tokens").sum().alias("total_tokens"),
        pl.col("volume").sum().alias("total_volume"),
        pl.col("net_revenue").sum().alias("total_revenue"),
        pl.col("date").n_unique().alias("days_active"),
        pl.col("total_prompt_tokens").sum().alias("total_prompt_tokens"),
        pl.col("total_completion_tokens").sum().alias("total_completion_tokens"),
        pl.col("prompt_price").drop_nulls().last().alias("prompt_price"),
        pl.col("completion_price").drop_nulls().last().alias("completion_price"),
    )
    .sort("total_tokens", descending=True)
    .head(15)
)
print()

# ── Top 10 authors by volume & revenue ───────────────────────────────────────
print("Top 10 authors by total volume:")
print(
    has_volume
    .group_by("openrouter_author")
    .agg(
        pl.col("volume").sum().alias("total_volume"),
        pl.col("net_revenue").sum().alias("total_revenue"),
        pl.col("openrouter_id").n_unique().alias("num_models"),
    )
    .sort("total_volume", descending=True)
    .head(10)
)
print()

# ── Monthly aggregate ─────────────────────────────────────────────────────────
print("Monthly totals:")
print(
    has_volume
    .with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
    .group_by("month")
    .agg(
        pl.col("volume").sum().alias("total_volume"),
        pl.col("net_revenue").sum().alias("total_revenue"),
        pl.col("openrouter_id").n_unique().alias("active_models"),
    )
    .sort("month")
)
print()

# ── Price distribution ────────────────────────────────────────────────────────
print("Price distribution ($/token, non-zero rows):")
print(
    df.filter(pl.col("prompt_price").is_not_null() & (pl.col("prompt_price") > 0))
    .select("prompt_price", "completion_price")
    .describe()
)
print()

# ── Top endpoint providers ────────────────────────────────────────────────────
print("Top 15 endpoint providers by row count:")
print(
    df.group_by("endpoint_provider")
    .agg(pl.len().alias("rows"))
    .sort("rows", descending=True)
    .head(15)
)
print()

# ── Token distribution check ──────────────────────────────────────────────────
token_cols = ["volume", "total_prompt_tokens", "total_completion_tokens", "total_native_tokens_reasoning"]

print("Token column coverage:")
print(
    df.select([pl.col(c).is_not_null().sum().alias(c) for c in token_cols])
    .unpivot(variable_name="column", value_name="non_null_rows")
)
print()

# ── Correlation: volume vs 3*prompt + 1*completion ───────────────────────────
complete = df.filter(
    pl.col("volume").is_not_null()
    & pl.col("total_prompt_tokens").is_not_null()
    & pl.col("total_completion_tokens").is_not_null()
)
print(f"Rows with volume, prompt, and completion tokens: {len(complete):,}")
corr = (
    complete
    .with_columns(
        (pl.col("total_prompt_tokens") + pl.col("total_completion_tokens")).alias("weighted_tokens")
    )
    .select(pl.corr("volume", "weighted_tokens"))
)
print("Correlation between volume and prompt_tokens + completion_tokens:")
print(corr)
print()
