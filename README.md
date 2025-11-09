# Increasing WellCo Member Retention with Vi Engage
## By *Vi Labs* 
**Version:** v1 | **Date:** Nov 11, 2025 | **Authors**: Vi Labs Team (Or Yair, Allon Hammer, Gil Caspi)

## 1. Getting Started 

### 1.1 Install Python 3.11
https://www.python.org/downloads/release/python-31114/

### 1.2 Create Virtual Environment (Recommended)
```bash
    python3.11 -m venv venv
    source venv/bin/activate
```

### 1.3 Install Required Packages
```bash
    pip install -r requirements.txt
```

### 1.4 Run the analysis pipeline: 
1. **Optional** - Run features generation: 
```bash
    python runners/run_features_generation.py
```    

2. **Optional** - Data exploration notebook: 
```bash
    jupyter lab data_exploration.ipynb
```

3. Train and evaluate models: 
```bash
    python runners/run_training_and_evaluation.py
```

4. Report notebook with train and evaluation results: 
```bash
    jupyter lab final_report_notebook.ipynb
```

5.Artifacts and results are saved in the `artifacts/` folder.
    5.1. Table with top-N members for outreach (ranked and sorted): `artifacts/predictions/latest_outreach_suggestion.csv`


## 2. Executive Summary
WellCo experienced increased members churn rates recently and seek Vi Labs assistance in reducing it. 
Vi Labs seamlessly integrated with WellCo's data systems and developed an AI-driven outreach strategy to optimize retention efforts.
**Vi Engage increased WellCo's expected member retention uplift from 1.32% to 3.94% - using the same outreach budget.**

## 3. Objective & Problem Definition
WellCo seeks to reduce member churn through data-driven, personalized online interventions.  
The main objective is:
- **Maximize Retention Uplift:** Identify members most likely to benefit from targeted interventions and maximize overall retention impact.

Output should include a ranked list of members for outreach, along with prioritization scores and rank. 

## 4. Data Overview 
1. WellCo datasets: 
   1. churn labels:
      1. member_id 
      2. signup_date
      3. churn 
      4. outreach

   2. claims (icd10 codes): 
      1. member_id 
      2. icd_code 
      3. diagnosis_date

2. Vi Labs datasets: 
   1. web visits:
      1. member_id 
      2. url 
      3. title 
      4. description 
      5. timestamp

## 5. Feature Engineering

### 5.1 Claims-Based Features
Derived from ICD-10 diagnosis codes:
- **has_diabetes (E11.9)** – binary flag  
- **has_hypertension (I10)** – binary flag  
- **has_dietary_counseling (Z71.3)** – binary flag  
- **in_cohort** – flag for members with any of the above

### 5.2 App Usage Features
Aggregated from the mobile app activity logs:
- `total_app_sessions` – total sessions per member  
- `unique_app_active_days` – distinct active days  
- `average_sessions_per_day` – mean engagement intensity  
- `std_sessions_per_day` – engagement variability  
- `max_sessions_per_day` – peak daily activity  
- `app_usage_duration_days` – active period (first–last session)  
- `days_from_last_app_use` – inactivity gap prior to churn observation

### 5.3 Web Visit Features
Extracted from web interaction data:
- `total_web_visits` – overall visits  
- `unique_urls` – number of unique visited URLs  
- `total_wellco_visits` – visits to WellCo-related domains  
- `wellco_visits_ratio` – share of WellCo visits among total  
- `unique_wellco_active_days` – active days on WellCo web content  
- `average_wellco_visits_per_day` – normalized engagement metric

Semantic web pages categories features (added in feature version v2): 
- Generate 10 clusters using K-Means on TF-IDF embeddings of page content (domain + title + description).
- For each cluster, create features:  
  - Category visit counts – total visits in category (cluster)
  - Visit ratios – share of visits in category with respect to total visits
  - Binary flags – whether member visited category at least once
  - Category diversity (entropy) - diversity of visited categories
  - Time-based metrics - 
    - Days since last visit in category 
    - Number of active days in category
    - Average visits per active day in category
  
## 6. Modelling Framework

### 6.1 Baseline Churn Model
A **Logistic Regression** model (with standardized features and class balancing) was trained to predict churn probability.  

### 6.2 Uplift Modeling
To estimate **incremental impact** of outreach (the “treatment”), we implemented uplift modeling using the *Two-Model* approach:

- **Treatment model:** predicts churn probability for outreached members  
- **Control model:** predicts churn probability for non-outreached members  
- **Uplift score:**  

  $$
  \text{uplift}(\mathbf{x}) = \mathbb{E}[\text{churn} \mid \text{outreach}=0, \mathbf{X}=\mathbf{x}] - \mathbb{E}[\text{churn} \mid \text{outreach}=1, \mathbf{X}=\mathbf{x}]
  $$

  Higher scores indicate members **less likely to churn if outreached**. 

### 6.3 Matching Procedure
Since the dataset is observational, we used **propensity score matching (PSM)** to simulate a randomized design (randomized control trial).


## 7. Algorithm Flow 
### 7.1 Load and join datasets
### 7.2 Train - Test split 
Split data into train (80%) and test (20%) sets, stratified by churn and outreach labels.
### 7.3 Balance treatment and control groups 
Use propensity-score matching to balance outreach and non-outreach groups.
### 7.4 Feature generation
Explained in Section 5.
### 7.5 Model training
- Train baseline churn model: $$\text {churn risk} = \mathbb{P}[\text{churn = 1} \mid \mathbf{X}=\mathbf{x}]$$ using Logistic Regression.
- Train uplift models (two models approach - treatment and control) to estimate retention gain from outreach (both XGBoost models): 
  - Treatment model: $$\mathbb{P}[\text{churn = 1} \mid \text{outreach = 1}, \mathbf{X}=\mathbf{x}]$$
  - Control model: $$\mathbb{P}[\text{churn = 1} \mid \text{outreach = 0}, \mathbf{X}=\mathbf{x}]$$

### 7.6 Members ranking and scoring 
- Compute uplift score:
  - $$\text{uplift}(\mathbf{x}) = \mathbb{P}[\text{churn = 1} \mid \text{outreach = 0}, \mathbf{X}=\mathbf{x}] - \mathbb{P}[\text{churn = 1} \mid \text{outreach = 1}, \mathbf{X}=\mathbf{x}]$$
- Rank members by uplift score (highest to lowest).
- Select top-N members for outreach based on budget constraints.

## 8. Evaluation & Results
- Cumulative gain in retention uplift from 1.32% (current outreach policy) to 3.94% (Vi Engage uplift model) using the same outreach budget.

## 9. Cost - Value Optimization
- Given cost and lifetime value per member, optimize N to maximize ROI.

## 10. Business Insights


## 11. Deliverables Summary

## 12. Next Steps 
### 12.1 **Model Optimization**  
   - Evaluate alternative models  
   - Tune hyperparameters  
   - Test alternative uplift modeling approaches  

### 12.2 **Feature Enrichment**  
   - Improve web page classification using LLMs  

### 12.3 **Code Refactoring**  
   - Prepare Vi Engage for seamless integration with future clients  

### 12.4 **Business Alignment**  
   - Optimize expected ROI through a personalized outreach method for each member  
   - Requires access to WellCo’s outreach cost structure and average member value
