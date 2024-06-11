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
```

And here is the output : 

<img width="515" alt="Helitron" src="https://github.com/jeremchn/Results-of-the-internship/assets/152181344/a242cd4d-a343-4423-8017-bcc41f5e1ae6">

We can do this with all the superfamilies and the best results or obtained with the superfamilies DNA and Gypsy :
<img width="492" alt="DNA" src="https://github.com/jeremchn/Results-of-the-internship/assets/152181344/27b0acc3-d6b5-4bb3-8cc5-32dadb09cd7a">
<img width="447" alt="Gypsy" src="https://github.com/jeremchn/Results-of-the-internship/assets/152181344/d4021b02-2169-4298-9043-4c42cab40b29">

With the following codes we can plot the AUC and ROC curves which are 2 metrics to represent the recall and the precision ouf our models
```python
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Tracer les courbes ROC et les courbes précision-rappel pour chaque fold
plt.figure(figsize=(15, 10))
for i, estimator in enumerate(cv_results['estimator']):
    try:
        # Prédictions de probabilités sur les données de test
        y_prob = estimator.predict_proba(X_test)[:, 1]

        # Calculer les courbes ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        # Calculer l'aire sous la courbe ROC (AUC)
        roc_auc = auc(fpr, tpr)

        # Tracer la courbe ROC
        plt.subplot(2, 3, i+1)
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Fold {i+1}')
        plt.legend(loc="lower right")

        # Calculer les courbes précision-rappel
        precision, recall, _ = precision_recall_curve(y_test, y_prob)

        # Tracer la courbe précision-rappel
        plt.subplot(2, 3, i+1)  # Utilisation du même sous-graphique que la courbe ROC
        plt.plot(recall, precision, color='green', lw=2, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - Fold {i+1} DNA')
        plt.legend(loc="lower left")
    except Exception as e:
        print(f"Fold {i+1} a échoué avec l'erreur : {e}")

plt.tight_layout()
plt.show()
```
Here are the different curves:

<img width="506" alt="DNA curves" src="https://github.com/jeremchn/Results-of-the-internship/assets/152181344/4202eced-ee28-4d41-a6fb-6472c30481e7">
<img width="502" alt="Gypsy curves" src="https://github.com/jeremchn/Results-of-the-internship/assets/152181344/0b6b95ef-5518-44a7-b26f-dd169f96c21f">

Finally we can see the importance of the different features in our classification models : 
```python
# Entraîner à nouveau le modèle pour obtenir les valeurs d'importance des features
model.fit(X_train, y_train)

# Récupérer l'importance des features
feature_importance = model.feature_importances_

# Récupérer les noms des features
feature_names = combined_numérique_df.columns

# Tracer l'importance des features sur un barplot
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names, palette="Blues_d")
plt.xlabel('Importance des features')
plt.ylabel('Features')
plt.title('Importance des features dans RandomForestClassifier DNA')
plt.show()
```
Here are the importance features : 

<img width="498" alt="DNA importance" src="https://github.com/jeremchn/Results-of-the-internship/assets/152181344/acc9be4e-50fd-4973-befc-1a3b37cf4766">
<img width="454" alt="Gypsy Importance" src="https://github.com/jeremchn/Results-of-the-internship/assets/152181344/83d3842b-357c-4119-b2d0-711079fc885f">
