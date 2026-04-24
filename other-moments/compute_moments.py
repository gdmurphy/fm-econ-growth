import pandas as pd

# Load data
automated = pd.read_csv("other-moments/job_exposure.csv")
occ_level = pd.read_csv("other-moments/occ_level.csv")
bls = pd.read_excel("other-moments/national_M2024_dl.xlsx", sheet_name="national_M2024_dl")
task_statements = pd.read_excel("other-moments/Task Statements.xlsx", sheet_name="Task Statements")
ap = pd.read_excel("other-moments/automation_potential (LLM time saving more than 10%) MAY 7 2025.xlsx", sheet_name="Sheet1")

# Filter to detailed occupations and normalize key column
bls_detailed = bls[bls["O_GROUP"] == "detailed"].copy()
bls_detailed["OCC_CODE"] = bls_detailed["OCC_CODE"].str.strip()
bls_detailed["TOT_EMP"] = pd.to_numeric(bls_detailed["TOT_EMP"], errors="coerce")

bls_codes = set(bls_detailed["OCC_CODE"])

# --- Fraction of tasks automated by AI (Anthropic observed exposure metric) ---
automated["occ_code"] = automated["occ_code"].str.strip()

unmatched_auto = set(automated["occ_code"]) - bls_codes
if unmatched_auto:
    raise ValueError(f"Automated fraction codes with no BLS match: {sorted(unmatched_auto)}")

merged_auto = bls_detailed[["OCC_CODE", "TOT_EMP"]].merge(
    automated[["occ_code", "observed_exposure"]],
    left_on="OCC_CODE", right_on="occ_code",
    how="left"
).dropna(subset=["TOT_EMP"])
merged_auto["observed_exposure"] = merged_auto["observed_exposure"].fillna(0)

auto_unmatched_bls = merged_auto["occ_code"].isna().sum()
print(f"[Automated fraction]    BLS detailed: {len(bls_detailed)}, Matched: {len(merged_auto) - auto_unmatched_bls}, BLS codes set to 0: {auto_unmatched_bls}")

auto_weighted   = (merged_auto["TOT_EMP"] * merged_auto["observed_exposure"]).sum() / merged_auto["TOT_EMP"].sum()
auto_unweighted = merged_auto["observed_exposure"].mean()

# --- Fraction of tasks not completable by AI (Elondou et al.) ---
# dv_rating_gamma = fraction of tasks completable by AI; 1 - gamma = fraction not completable.
# O*NET codes are XX-XXXX.XX; strip to XX-XXXX to match BLS.
# Average gamma across O*NET sub-codes sharing the same BLS code before inverting.
occ_level["bls_code"] = occ_level["O*NET-SOC Code"].str[:7]
not_completable = (
    occ_level.groupby("bls_code")["dv_rating_gamma"]
    .mean()
    .reset_index()
)
not_completable["not_completable"] = 1 - not_completable["dv_rating_gamma"]

unmatched_nc = set(not_completable["bls_code"]) - bls_codes
if unmatched_nc:
    print(
        f"[Not completable fraction] NOTE: {len(unmatched_nc)} O*NET-derived codes have no match in BLS 2024 "
        f"detailed occupations (likely deprecated/renamed) and are excluded from both metrics: {sorted(unmatched_nc)}"
    )

merged_nc = bls_detailed[["OCC_CODE", "TOT_EMP"]].merge(
    not_completable[["bls_code", "not_completable"]],
    left_on="OCC_CODE", right_on="bls_code",
    how="left"
).dropna(subset=["TOT_EMP"])
merged_nc["not_completable"] = merged_nc["not_completable"].fillna(0)

nc_unmatched_bls = merged_nc["bls_code"].isna().sum()
print(f"[Not completable fraction] BLS detailed: {len(bls_detailed)}, Matched: {len(merged_nc) - nc_unmatched_bls}, BLS codes set to 0: {nc_unmatched_bls}")

nc_weighted   = (merged_nc["TOT_EMP"] * merged_nc["not_completable"]).sum() / merged_nc["TOT_EMP"].sum()
nc_unweighted = merged_nc["not_completable"].mean()

# --- Fraction of tasks not completable by AI (LLME0: absent from automation_potential) ---
# Tasks not in automation_potential are LLME0 (not completable). Core/blank weight=2, Supplemental weight=1.
# Compute a weighted fraction of LLME0 tasks per O*NET occupation, then average sub-codes to BLS level.
ap_task_ids = set(ap["Task ID"])
task_statements["is_llme0"] = (~task_statements["Task ID"].isin(ap_task_ids)).astype(int)
task_statements["weight"] = task_statements["Task Type"].apply(
    lambda t: 1 if str(t).strip() == "Supplemental" else 2
)

occ_scores = (
    task_statements.groupby("O*NET-SOC Code")
    .apply(lambda g: (g["weight"] * g["is_llme0"]).sum() / g["weight"].sum(), include_groups=False)
    .reset_index(name="llme0_score")
)
occ_scores["bls_code"] = occ_scores["O*NET-SOC Code"].str[:7]
llme0_by_bls = occ_scores.groupby("bls_code")["llme0_score"].mean().reset_index()

unmatched_llme0 = set(llme0_by_bls["bls_code"]) - bls_codes
if unmatched_llme0:
    print(
        f"[LLME0 fraction] NOTE: {len(unmatched_llme0)} O*NET-derived codes have no match in BLS 2024 "
        f"detailed occupations and are excluded: {sorted(unmatched_llme0)}"
    )

merged_llme0 = bls_detailed[["OCC_CODE", "TOT_EMP"]].merge(
    llme0_by_bls, left_on="OCC_CODE", right_on="bls_code", how="left"
).dropna(subset=["TOT_EMP"])
merged_llme0["llme0_score"] = merged_llme0["llme0_score"].fillna(0)

llme0_unmatched_bls = merged_llme0["bls_code"].isna().sum()
print(f"[LLME0 fraction]           BLS detailed: {len(bls_detailed)}, Matched: {len(merged_llme0) - llme0_unmatched_bls}, BLS codes set to 0: {llme0_unmatched_bls}")

llme0_weighted   = (merged_llme0["TOT_EMP"] * merged_llme0["llme0_score"]).sum() / merged_llme0["TOT_EMP"].sum()
llme0_unweighted = merged_llme0["llme0_score"].mean()

# --- Summary: exposure moments ---
col1 = "Fraction automated by AI"
col2 = "Frac. not completable (Elondou)"
col3 = "Frac. not completable (LLME0)"
print(f"\n{'Metric':<35} {col1:>26} {col2:>33} {col3:>31}")
print("-" * 128)
print(f"{'Employment-weighted':<35} {auto_weighted:>26.6f} {nc_weighted:>33.6f} {llme0_weighted:>31.6f}")
print(f"{'Unweighted mean':<35} {auto_unweighted:>26.6f} {nc_unweighted:>33.6f} {llme0_unweighted:>31.6f}")

# --- Ratio of AI inference spend to wage bill ---

# Wage bill (FRED labor share × BEA GDP)
gdp             = 30.76e12   # BEA
labor_share     = 0.519      # FRED
wage_bill       = gdp * labor_share

# Method 1: Extrapolate from OpenAI ARR + OpenRouter revenue share
openai_arr         = 25e9
openai_api_rev     = 0.20 * openai_arr          # 20% of ARR is API revenue
openrouter_openai_2mo = 1e6                     # OpenRouter revenue from OpenAI models over 2 months
openrouter_openai_ann = openrouter_openai_2mo * 6
openrouter_share_m1   = openrouter_openai_ann / openai_api_rev  # OpenRouter's share of OpenAI API market
openrouter_annual_rev = 182e6
global_api_market_m1  = openrouter_annual_rev / openrouter_share_m1
us_share              = 0.227                   # Cloudflare web traffic to GenAI sites
us_inference_m1       = global_api_market_m1 * us_share
ratio_m1              = us_inference_m1 / wage_bill

# Method 2: Extrapolate from Google token volume + OpenRouter token share
google_tokens_2mo       = 1.46e15              # Google total API tokens over 2 months
openrouter_google_2mo   = 6.2e12               # OpenRouter Google tokens over 2 months
openrouter_share_m2     = openrouter_google_2mo / google_tokens_2mo
global_api_market_m2    = openrouter_annual_rev / openrouter_share_m2
us_inference_m2         = global_api_market_m2 * us_share
ratio_m2                = us_inference_m2 / wage_bill

print(f"\n--- AI Inference Spend / Wage Bill ---")
print(f"Wage bill (FRED labor share {labor_share:.1%} × BEA GDP ${gdp/1e12:.2f}T): ${wage_bill/1e12:.2f}T")
print()
print(f"Method 1 (OpenAI ARR extrapolation):")
print(f"  OpenAI API revenue:              ${openai_api_rev/1e9:.1f}B/yr")
print(f"  OpenRouter share of OpenAI API:  {openrouter_share_m1:.4%}")
print(f"  Global API market:               ${global_api_market_m1/1e9:.1f}B/yr")
print(f"  US inference spend (×{us_share:.1%}):    ${us_inference_m1/1e9:.1f}B/yr")
print(f"  Inference / wage bill:           {ratio_m1:.4%}")
print()
print(f"Method 2 (Google token volume extrapolation):")
print(f"  OpenRouter share of Google API:  {openrouter_share_m2:.4%}")
print(f"  Global API market:               ${global_api_market_m2/1e9:.1f}B/yr")
print(f"  US inference spend (×{us_share:.1%}):    ${us_inference_m2/1e9:.1f}B/yr")
print(f"  Inference / wage bill:           {ratio_m2:.4%}")
