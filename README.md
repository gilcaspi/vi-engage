# AI System to Optimize WellCo Outreach Strategy for Maximal ROI
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

### 
1. Run features generation: 
```bash
    python src/features/build_features.py
```    

2. Data exploration notebook: 
```bash
    jupyter lab notebooks/1_data_exploration.ipynb
```

3. Train and evaluate models: 
```bash
    python src/models/train_evaluate_models.py
```

4. Artifacts and results are saved in the `artifacts/` folder.
   1. Table with top-N members for outreach (ranked and sorted): `artifacts/predictions/latest_outreach_suggestion.csv`


## 2. Executive Summary
WellCo experienced increased members churn rates recently and seek Vi Labs assistance in reducing it. 
Vi Labs seamlessly integrated with WellCo's data systems and developed an AI-driven outreach strategy to optimize retention efforts.
**Vi Engage increased WellCo's expected member retention uplift from 3% to 21% - using the same outreach budget.**

## 3. Objectives & Problem Definition
WellCo seeks to reduce member churn through data-driven, personalized online interventions.  
The main objectives are:

- **Maximize Retention Uplift:** Identify members most likely to benefit from targeted interventions and maximize overall retention impact.  
- **Optimize Outreach Budget:** Achieve the highest possible return on investment (ROI) within the existing budget for online engagement efforts.

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

## 6. Modeling Framework 
### 6.1 Baseline Churn Model 
Trained a model to predict churn risk using historical data.


### 6.2 Uplift Modeling 
#### 6.2.1 Matching pairs 
Since our data is observational (not randomized control trial with respect to outreach as a treatment),
we needed to split our data to outreached group and control group with matching pairs. 



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
