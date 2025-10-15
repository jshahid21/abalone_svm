# Abalone Ring Prediction using Support Vector Regression

This project aims to predict the age of abalone by predicting the number of rings using a Support Vector Regression (SVR) model.

## Data

The dataset used is the Abalone Data Set, which contains various physical measurements of abalone. The 'Sex' column was dropped as part of the preprocessing. The target variable is 'Rings', which represents the number of rings and corresponds to the age of the abalone.

## Model

A Support Vector Regression (SVR) model with a Radial Basis Function (RBF) kernel was used for prediction. The data was split into training and testing sets (80/20 split).

## Results

The SVR model with an RBF kernel achieved a score of {{sv.score(X_test, y_test):.2f}} on the test set.

The scatter plot of Actual vs. Predicted Rings shows that the model is able to capture some of the variance in the data, but there is still a significant spread in the predictions, particularly for higher numbers of rings. This suggests that the model's ability to predict the exact number of rings is limited.

## Interpretation

The current SVR model provides a moderate level of accuracy in predicting the number of abalone rings. While the RBF kernel improved the performance compared to a linear kernel, the scatter plot indicates that the model struggles with precise predictions, especially for older abalone with more rings. This could be due to the inherent variability in abalone growth or limitations of the current model and features. Further exploration, such as feature engineering, hyperparameter tuning, or trying different models (like the classification approach discussed earlier), could potentially improve the prediction accuracy.
