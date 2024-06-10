# Classification of the TE Superfamilies

We possess 89 genomes of the plant *Arabidopsis thaliana*. For each of these genomes, we have the transposable elements (TEs) and 69 associated features. Our goal is to develop a machine learning model that predicts the superfamily of the TEs using all the available features.

Here is an example of a Python code snippet used to achieve this:

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings

# Binary mapping of TE superfamily
combined_df['TE.superfamily_binary'] = combined_df['TE.superfamily'].map(lambda x: 1 if x == 'Helitron' else 0)

# Ignore warnings for better readability
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_numérique_df, combined_df['TE.superfamily_binary'], test_size=0.2, random_state=42)

# Define the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=300)

# Perform cross-validation with 5 folds
cv_results = cross_validate(model, X_train, y_train, cv=5, return_estimator=True)

# Retrieve the model estimators for each fold
estimators = cv_results['estimator']

# Plot the confusion matrix for each fold
plt.figure(figsize=(15, 10))
for i, estimator in enumerate(estimators):
    # Predictions on the test data
    predictions = estimator.predict(X_test)

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions)

    # Plot the confusion matrix
    plt.subplot(2, 3, i+1)
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
    plt.title(f"Confusion Matrix - Fold {i+1} Helitron")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")

plt.tight_layout()
plt.show()




```markdown
![Confusion Matrix - Helitron](Helitron.PNG)

