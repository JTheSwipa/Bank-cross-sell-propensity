# Propensity to Buy: Expected Value-Driven Credit Card Cross-Sell Model

## Business Problem
Traditional marketing campaigns often rely on arbitrary percentile targeting (e.g., "email the top 10% most likely to buy"). This approach is financially flawed. It ignores the actual customer lifetime value (LTV) of the product and the operational cost of the marketing outreach, resulting in wasted budget on low-probability prospects or missed opportunities on high-value segments.

## The Solution
This repository implements an **Expected Value (EV) framework** for product cross-selling. Instead of optimizing for pure statistical accuracy, the pipeline translates machine learning probabilities directly into projected dollar amounts. 

By calculating `EV = (Probability of Conversion * Expected LTV) - Cost of Contact`, the model isolates a strictly profitable segment of the customer base. Based on extensive temporal backtesting, the mathematically optimal configuration generates a projected **$2,166 ROI per 1,000 scored customers**, capturing 62.4% of all available buyers in the market.

## Key Technical Methodologies
* **Optimal Temporal Windowing (3->4):** Conducted normalized sweeps across 16 different time-window configurations to identify that a 3-month observation window paired with a 4-month performance horizon yields the highest financial return.
* **Strict Out-Of-Time (OOT) Validation:** Prevented data leakage by anchoring training, validation, and testing sets in entirely distinct, chronologically sequential timeframes to simulate true production environments.
* **The "Stability Rule" Target Generation:** Engineered a custom target definition that ensures only clients with 100% stable product holdings throughout the observation period are flagged as eligible cross-sell prospects.
* **Probability Calibration for Business Logic:** Utilized `CalibratedClassifierCV` (Sigmoid) on top of the base CatBoost classifier. Because the final EV math relies heavily on accurate probability magnitudes rather than just rank-ordering, true empirical calibration was prioritized over synthetic data sampling methods like SMOTE.

## Repository Structure
├── data/                      # Raw and processed datasets (Ignored in git)
├── notebooks/
│   └── Draftv1.0.ipynb        # Core pipeline: EDA, Feature Engineering, Model Training, Sweeps
├── artifacts/
│   ├── output_39_0.png        # Feature Importance Visualization
│   ├── Target_List_EV.csv     # Final scored deployment list for OOT evaluation
│   └── Active_Base_201903.csv # Production-ready inference list
├── README.md                  # Project overview and methodology
└── requirements.txt           # Python dependencies
