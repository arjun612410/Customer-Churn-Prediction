🧠 ANN Classification — Customer Churn Prediction

An end-to-end Deep Learning project using TensorFlow and Streamlit to predict customer churn based on the Churn_Modelling.csv dataset.


📌 Project Overview

This project builds a complete pipeline — from raw data to a deployed web application — using an Artificial Neural Network (ANN). It covers feature engineering, model training, evaluation, and deployment via an interactive Streamlit web app.


🏗️ Project Architecture

Churn_Modelling.csv
        │
        ▼
Feature Transformation (sklearn)
        │
        ▼
ANN Model Training (TensorFlow/Keras)
        │
        ▼
Model Evaluation & Prediction
        │
        ▼
Streamlit Web App Integration
        │
        ▼
Deployment


🔄 Project Workflow

1. 🔧 Feature Transformation (sklearn)


Handled missing values and outliers
Applied Label Encoding for binary categorical columns (e.g., Gender)
Applied One-Hot Encoding for multi-class categorical columns (e.g., Geography)
Used StandardScaler to normalize numerical features
Split dataset into training and test sets using train_test_split


2. 🧠 ANN Model Training (TensorFlow/Keras)


Built a Sequential ANN model with optimal hidden layers and neurons
Used ReLU activation for hidden layers and Sigmoid for output layer
Configured with:

Optimizer: Adam
Loss Function: Binary Crossentropy
Metric: Accuracy



Trained with early stopping to prevent overfitting
Saved the trained model for later inference


3. 🔮 Prediction with Trained ANN


Loaded the saved trained model
Preprocessed new/unseen input data using the same sklearn transformers
Generated churn probability predictions
Applied threshold (0.5) to classify as Churned or Retained


4. 🌐 Streamlit Web App Integration


Built an interactive UI using Streamlit
Users can input customer details via form fields
Backend loads the saved ANN model and sklearn preprocessors
Displays real-time prediction results with probability score


5. 🚀 Deployment


Deployed the Streamlit web app
App accepts live user inputs and returns churn predictions instantly
Model artifacts (.h5 / .pkl files) are bundled with the app


6. 📈 ANN Regression Practical Implementation


Extended the ANN architecture for regression tasks
Replaced Sigmoid output with Linear activation
Changed loss function to Mean Squared Error (MSE)
Demonstrated use case: predicting a continuous output (e.g., estimated salary)


7. 🔍 Finding Optimal Hidden Layers & Neurons


Experimented with different architectures:

Varying number of hidden layers (1, 2, 3+)
Varying neurons per layer (16, 32, 64, 128)



Used KerasTuner / manual grid search to identify best configuration
Evaluated models using validation accuracy and loss curves
Selected the architecture with best bias-variance tradeoff



🛠️ Tech Stack

ComponentTechnologyLanguagePython 3.xDeep LearningTensorFlow / KerasFeature EngineeringScikit-learnWeb AppStreamlitData HandlingPandas, NumPyVisualizationMatplotlib, TensorBoard