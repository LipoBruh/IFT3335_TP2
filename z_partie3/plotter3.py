import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# partie 1.1
df1 = pd.read_csv("../output/3_1_superlearner.csv")
#

# Accuracy plot for the 3 models : 
summary = pd.DataFrame({
    'Model': [
        'Super Learner : LR+SVC+RF (TF-IDF)'
    ],
    'Mean Accuracy': [
        df1['mean_test_accuracy'].iloc[0]
    ],
    'Std Accuracy': [
        df1['std_test_accuracy'].iloc[0]
    ],
    'Mean F1': [
        df1['mean_test_f1'].iloc[0]
    ],
    'Std F1': [
        df1['std_test_f1'].iloc[0]
    ],
})

print(df1)

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 8), sharex=True)

# First plot: Accuracy
ax1.bar(summary['Model'], summary['Mean Accuracy'], yerr=summary['Std Accuracy'], capsize=5, color='skyblue', edgecolor='black', width=0.3)
ax1.set_ylabel('Mean Accuracy')
ax1.set_ylim(0.95, 1.00)
ax1.set_title('Model Accuracy ± Std')


# Second plot: F1 Score
ax2.bar(summary['Model'], summary['Mean F1'], yerr=summary['Std F1'], capsize=5, color='lightcoral', edgecolor='black', width=0.3)
ax2.set_ylabel('Mean F1 Score')
ax2.set_ylim(0.85, 1.00)
ax2.set_title('Model F1 Score ± Std')

plt.xticks(rotation=20)
plt.tight_layout()
plt.show()