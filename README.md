<h1 align="center">
<b>NBA Player Shot Quality Prediction</b>
</h1>
<h3 align="center">
<b>By: Anirudh Mazumder</b>
</h3>

## Project Overview
In this project, I built an expected points (xPoints) model that evaluates the quality of NBA shots using play-by-play and shot log data. The model predicts the probability of a shot being made based on contextual features and then multiplies that probability by the shot type (2PT or 3PT) to estimate expected points per possession.

## Tech Stack
- Python: Pandas, Scikit-Learn, Matplotlib, Streamlit

## Features
### 1. Data Preprocessing
I cleaned the data by removed missing values, converted time features, and dropped columns which potentially lead to data leakage during model training. Further, I encoded categorical features and kept key contextual variables.

### 2. Model Training
I trained logistic regression, random forest, and neural network models and calcualted evaluation metrics for each model, such as, AU-ROC values, accuracy metrics, and a classification report. I then saved the neural network because it performed the best.

### 3. Data Analysis and Visualization
I created a webapp using Streamlit to visualize information about the data and added a filter so users can see the performance of the model and how well a player was at shooting on a player-by-player basis. Below is an example of some of the data that is visualized on the dashboard.
<img width="1401" height="473" alt="Screenshot 2025-09-18 at 12 40 20 PM" src="https://github.com/user-attachments/assets/b73261b0-a28c-4aec-b782-bd514548fd0f" />

## Citation
Shot Logs Dataset: https://www.kaggle.com/datasets/dansbecker/nba-shot-logs/data
