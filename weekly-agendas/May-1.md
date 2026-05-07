# Weekly Action Items Summary

## 1. Market Share Data Sources

**Update:**  
We now have several sources of market share data available:

- Previous survey-based data
- Previously collected Ramp-based data
- New Ramp data with spending shares
- API spending data from the CBA Australian bank environment

**Key issue:**  
There is substantial inconsistency across the different sources. These discrepancies will need to be reconciled or clearly documented before drawing strong conclusions from the combined evidence.


## 2. Ramp Business Size Methodology

**Status:** Clarified

**Update:**  
Ramp business sizes are determined based on the number of employees.

The current size categories are:

| Category | Number of Employees |
|---|---:|
| Small | 1-24 |
| Medium | 25-199 |
| Large | 200+ |


## 3. Firm Characteristics from CBA Data

**Status:** Partially available; better measures expected later

**Update:**  
The CBA data provides good measures of adoption, but the current measures of firm size, expenses, and revenue are weak.

In Neil's previous presentation, we used a CBA-defined categorization of wages as a proxy. However, we do not have high confidence in that measure.

**Key issue:**  
The wage-based proxy may not reliably capture the firm characteristics we care about.

**Next steps:**  

- Treat existing CBA firm-size and financial-characteristic measures cautiously.
- Avoid overinterpreting results based on the wage proxy.
- Revisit this once improved measures become available in the coming weeks and months.


## 4. Advertising and Marketing Data for Foundation Model Vendors

**Status:** Limited data found

**Update:**  
There appears to be little comprehensive public data on advertising and marketing spend by foundation model vendors. However, I found several useful data points:

- A reported leak suggesting OpenAI spent approximately **$2 billion on sales and marketing in the first half of 2025**.
- MediaRadar estimates for advertising spend from **October 2024 to October 2025**:
  - OpenAI spent approximately **$30 million** advertising ChatGPT.
  - Anthropic spent approximately **$14.8 million** advertising Claude.
  - Google spent approximately **$17 million** advertising Gemini.

**Key issue:**  
These sources are incomplete and may not be directly comparable. Sales and marketing spend is broader than advertising spend, and vendor-specific definitions may differ.


## 5. OpenRouter Price Consistency

**Status:** Initial anomalies identified

**Update:**  
Some models in the OpenRouter data show spikes in pricing. These pricing changes may be useful for measuring elasticity.

For validation, Hans has historical Artificial Analysis pricing data, but it is somewhat inconsistent with OpenRouter and less fine-grained. The Artificial Analysis data is available at monthly frequency at most, while OpenRouter data is more granular. I can perform further research on this though beyond the major sources which I've looked at it as I'd imagine this data is available somewhere. If we really care to get good data on this, I can look at archived webpages for model providers of interest and parse them each individually into historical pricing data.

**Key issue:**  
There is not yet strong external validation for the OpenRouter pricing spikes.

**Next steps:**  

- Compare OpenRouter price spikes against Hans's Artificial Analysis pricing data where possible.
