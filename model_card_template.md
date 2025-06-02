# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a binary classification model that predicts whether a person's annual income is greater than \$50K or less than or equal to \$50K, based on demographic and employment-related features from the U.S. Census dataset. It is implemented using a RandomForestClassifier from scikit-learn and deployed as a REST API using FastAPI.

## Intended Use
The model is a binary classifier that predicts if the salary is >50K or <=50K.

## Training Data
Information about training data can be found https://archive.ics.uci.edu/dataset/20/census+income

## Evaluation Data
20% of the cleaned dataset was held out as test data for final model evaluation. In addition, slice-based evaluations were performed across all categorical features to assess performance fairness across different groups (e.g., gender, race, education levels).

## Metrics
The model was evaluated using:
- Precision: 0.736
- Recall: 0.613
- F1 Score: 0.669

## Ethical Considerations
The dataset contains sensitive attributes such as race, sex, and native-country, which may introduce bias in model predictions. Although slice-based evaluation is used to monitor fairness, no mitigation or bias correction has been applied. This model should not be used in decision-making systems that affect individuals’ access to resources or opportunities.

## Caveats and Recommendations
- This model is intended for instructional use only.
- Fairness evaluations should be extended with tools like Aequitas or Fairlearn if used beyond this educational scope.
- The model's performance may vary significantly across slices. Further tuning or feature engineering may be required to improve fairness or overall accuracy.
- Income prediction based on personal data may raise ethical and legal concerns — real-world applications should follow data privacy and fairness standards.