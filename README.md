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

4. Artifacts and results are saved in the `artifacts/` folder.
   1. Table with top-N members for outreach (ranked and sorted): `artifacts/predictions/latest_outreach_suggestion.csv`


## 2. Executive Summary
WellCo experienced increased members churn rates recently and seek Vi Labs assistance in reducing it. 
Vi Labs seamlessly integrated with WellCo's data systems and developed an AI-driven outreach strategy to optimize retention efforts.
**Vi Engage increased WellCo's expected member retention uplift from 3% to 21% - using the same outreach budget.**

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

## 7. Evaluation & Results

## 8. Cost - Value Optimization

## 9. Business Insights

## 10. Deliverables Summary

## 11. Next Steps 
1. **Model Optimization**  
   - Evaluate alternative models  
   - Tune hyperparameters  
   - Test alternative uplift modeling approaches  

2. **Feature Enrichment**  
   - Improve web page classification using LLMs  

3. **Code Refactoring**  
   - Prepare Vi Engage for seamless integration with future clients  

4. **Business Alignment**  
   - Optimize expected ROI through a personalized outreach method for each member  
   - Requires access to WellCo’s outreach cost structure and average member value
