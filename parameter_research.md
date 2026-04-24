# Growth Model Parameter Research

## $r$ - rental rate of compute

- NVIDIA H100 (SXM 80GB) looks to be the most common hardware for large model runs according to [Epoch Data](https://epoch.ai/data/ai-models). On vast.ai, where competition is strong and decentralized, they are available for [$1.53/hr](https://vast.ai/pricing/gpu/H100-SXM). GCP also has a cost calculator for GPUs. Epoch estimates FLOPS/$ for H100s at [2.2e10 with a $44,000 upfront cost](https://epoch.ai/data-insights/price-performance-hardware).

An estimation for calculating the FLOPs for D tokens and M parameters, the forward pass is 2MD for the forward pass and 4MD for the backward pass, making [inference cost 2MD](https://arxiv.org/pdf/2211.05102) whereas training cost 6MD. A nice envelope calculation of this is [here](https://jax-ml.github.io/scaling-book/transformers/). These numbers are also noted as estimates in Kaplan et al. MoE models can mess this calculation up.

### $\xi^f$(training)

- Epoch also has estimates for various models on training FLOP and training costs for frontier model runs. Here is some day on $/FLOP for the largest models by training cost:


| Model                      | Organization | Training Compute (FLOP) | Training Cost (2023 USD) | $/FLOP   |
| -------------------------- | ------------ | ----------------------- | ------------------------ | -------- |
| Grok 4                     | xAI          | 5.00e+26                | 387,842,678              | 7.76e-19 |
| GPT-4.5                    | OpenAI       | 3.80e+26                | 339,957,480              | 8.95e-19 |
| Grok 3                     | xAI          | 3.50e+26                | 217,835,546              | 6.22e-19 |
| Llama 3.1-405B             | Meta AI      | 3.80e+25                | 52,885,434               | 1.39e-18 |
| Llama 4 Behemoth (preview) | Meta AI      | 5.18e+25                | 44,588,964               | 8.60e-19 |


DeepSeek V3 also reports 2.788M H800 GPU-hours and assumed $2/H800-hour in their paper which using Epoch's 3.41e24 FLOP number for them gives 1.44e-18 lining up well with these numbers.

Cost numbers are derived from their [cost paper](https://epoch.ai/blog/how-much-does-it-cost-to-train-frontier-ai-models) which has a method of disentangling into price per chip-hour from cloud providers and training chip-hours. Training chip-hours are calculated for each model using FLOP / (FLOP utilization * FLOPS) for the model. Their repo has hardware data on FLOPS and utilization for different GPUs. This method should bake in cumulative resource costs pretty well. Looks like their FLOP number is drawn from different sources but often the rule of thumb of $C=6MD$.

### $\xi^b$ (inference)

I was thinking of just looking at cost estimates of running large open-weight models by their parameter counts and fitting a curve to extrapolate to closed source models via Epoch estimates of their parameter counts. There is data for inference costs on [Artificial Analysis](https://artificialanalysis.ai/).
It is important to note that the price of inference (and training though we don't really care if we're measuring in FLOP) [depends on the speed which you serve at](https://epoch.ai/blog/inference-economics-of-language-models). In raw hardware, there is a [Google paper](https://arxiv.org/pdf/2211.05102) which details chip-seconds per token (for TPU v4) for different sized PaLM models and finds the cost to be proportional to parameter count. In both papers there looks to be convergence of inference cost as latency goes to infinity. NVIDIA has similar [data for tokens/sec](https://docs.nvidia.com/nim/benchmarking/llm/latest/performance.html) on 70B and 405B Llama models for different GPUs, precisions, and I/O sequence lengths. Argonne National Lab also has [a paper](https://arxiv.org/abs/2411.00136) where they benchmark tokens/sec across a variety of different open models, hardware, batch size, degrees of parallelization, and I/O lengths.

## $\bar{h}$ - Loss needed for most complex economic task

### Moment - Fraction of tasks not completable by AI

I think this is equivalent to the fraction of tasks with zero "exposure." In [GPTs are GPTs](https://arxiv.org/pdf/2303.10130#page=11.69), this is in Table 3 as $1 - \zeta = 0.44$.   
The ILO also has [an AI exposure measure](https://www.ilo.org/publications/generative-ai-and-jobs-refined-global-index-occupational-exposure) for ISCO-08 tasks.
[MIT Iceberg Index](https://iceberg.mit.edu/report.pdf) has an exposure measure on an occupational level which “measures the wage value of skills that AI systems can perform within each occupation.” They do say their analysis of BLS skill taxonomies show AI systems can perform 16% of classified labor tasks.

## $Z$ - Scale parameter for worker productivity

### Moment - Fraction of tasks automated by AI

WEF has [bi-annual survey data](https://www.weforum.org/publications/the-future-of-jobs-report-2025/) for automation more generally.
The [Anthropic Economic Index](https://www.anthropic.com/economic-index#job-explorer) has strong data for each O*NET task on "mostly automated" and "mostly augmented" tasks. The new [labor markets paper](https://www.anthropic.com/research/labor-market-impacts) they put out has an "observed exposure" measure based upon actual Claude usage which is supposed to measure which tasks are actually seeing automated usage. The results are initial and don't provide data on their statistic for every O*NET task, but maybe we can use the coarse data that's already there?
[GDPval](https://arxiv.org/abs/2510.04374v1) provides data on the ability of models to outperform workers on various tasks, but it doesn't account for whether it is cost-effective to do so.
In [MIT Iceberg Index](https://iceberg.mit.edu/report.pdf) they say their analysis of BLS skill taxonomies show AI systems can perform 16% of classified labor tasks.

## $K$ - Number of foundational model providers

### Moment - market shares of inference

Menlo Ventures has [survey data](https://menlovc.com/wp-content/uploads/2025/12/menlo_ventures_enterprise_ai_report-2025-123125.pdf) on AI usage for major companies showing OpenAI, Anthropic, and Google concentrate 77% of the API market share.
[Ramp AI Index](https://ramp.com/velocity/ai-index-january-2026) shows most of the market share among the top 3 players, but it's hard to use one number since adopting companies may be using multiple providers.
[Similarweb](https://www.similarweb.com/blog/marketing/seo/most-used-ai/), which provides website analytics, shows DeepSeek having more traffic, but I would assume this is mostly free users then.
a16z surveyed CIOs from Global 2000 companies and found [89% of spend](https://a16z.com/leaders-gainers-and-unexpected-winners-in-the-enterprise-ai-arms-race/) was on the top 3.

## tokens/task

[this paper constructed](https://arxiv.org/pdf/2412.14161v2#page=7.36) a lot of onet tasks and fed them to models. There's a codebase, but I haven't found task-level results so we may need to run ourselves? Also there results are given in dollars and LLM "steps" (i.e. model calls) rather than pure tokens
[this paper](https://openreview.net/pdf?id=1bUeVB3fov#page=21.10) looked at inference costs for SWE-bench tasks and gives the average tokens for completing one of those, so I guess we could just interpret those as a singular task and use that number

## $\theta$

### Moment - relationship between model improvements in loss and feasability of automation

A standardized view of loss via cross-entropy isn't very common with modern LLMs, but there is [some work](https://www.morpheus.systems/blog/norm_perplex) showing how it tracks for various open models. In terms of how automation improves with it, [GDPval](https://arxiv.org/abs/2510.04374v1) tracks how often AI can perform a task better than a human for different models.

## $\bar{d}$

### Moment - Ratio of AI inference spend to wage bill

There is surprisingly little info here, but revenue from foundation model providers might be an accurate idea of what we want to calculate. In [this MIT paper](https://www.astrid-online.it/static/upload/ssrn/ssrn-5767103.pdf#page=43.09), they use estimates of OpenAI API revenue from Epoch combined with the fraction of OpenRouter traffic OpenAI has and OpenRouter total revenue to back out a number of $60B for the total AI API inference market. They have a few other methods with similar numbers there representing a tweet from Demis Hassabis and the Menlo Ventures data from the $K$ estimate, but this seems like probably the best. We can do similar calculations using updated OpenAI revenue estimates. Using OpenRouter estimates will create a large amount of selection that we should be aware of.
If we want to filter to U.S. labor and AI usage, we can use [OpenRouter data](https://arxiv.org/pdf/2601.10088v1#page=25.09) on regional distribution though it looks like they only restrict to North America with estimates of over 40%. [Anthropic data](https://www.anthropic.com/research/economic-index-geography) has more specific U.S. estimates of 21.6%. [Cloudflare](https://blog.cloudflare.com/global-expansion-in-generative-ai-a-year-of-growth-newcomers-and-attacks/) and [similarweb](https://www.similarweb.com/website/chatgpt.com/#geography) provide similar estimates in terms of internet traffic to ChatGPT. This [Microsoft broad AI usage data](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/10/AI-Usage-Technical-Report.pdf) is a little different and required me to make some external estimates of working-age population but shows only 8.8% in North America.

For calculating the wage bill, we can just take GDP / labor share. FRED estimates labor share to be [51.9%](https://fred.stlouisfed.org/series/A4002E1A156NBEA). GDP is at 30.76T according to new [BEA data](https://www.bea.gov/data/gdp/gross-domestic-product) giving a labor bill of $15.97T. Thus, our total ratio here is probably something like 0.08% if we're looking at just API inference. If we go off of total OpenAI revenue rather than using an expected 20% on API, then we have 0.4%.

dig deeply into menlo data, filtration into U.S.
fraction of tasks that are green, green+blue in anthropic index and check with Elondou measure and weight by U.S. employment measures - do this for Elondou as well if not done already
work on fraction of tasks automated more
survey paper for more data
Python script formulating all measures from existing data - especially those discussed above
