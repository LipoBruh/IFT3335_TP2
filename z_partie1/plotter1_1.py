import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# partie 1.1
df1 = pd.read_csv("../output/1_1_LR_BOW.csv")
df2 = pd.read_csv("../output/1_1_LR_IDF.csv")
df3 = pd.read_csv("../output/1_1_RF_BOW.csv")
df4 = pd.read_csv("../output/1_1_RF_IDF.csv")
df5 = pd.read_csv("../output/1_1_MLP_BOW.csv")
df6 = pd.read_csv("../output/1_1_MLP_IDF.csv")
#

# Accuracy plot for the 3 models : 
summary = pd.DataFrame({
    'Model': [
        'LR (BoW)',
        'LR (TF-IDF)',
        'RF (BoW)',
        'RF (TF-IDF)',
        'MLP (BoW)',
        'MLP (TF-IDF)'
    ],
    'Mean Accuracy': [
        df1['mean_test_accuracy'].iloc[0],
        df2['mean_test_accuracy'].iloc[0],
        df3['mean_test_accuracy'].iloc[0],
        df4['mean_test_accuracy'].iloc[0],
        df5['mean_test_accuracy'].iloc[0],
        df6['mean_test_accuracy'].iloc[0]
    ],
    'Std Accuracy': [
        df1['std_test_accuracy'].iloc[0],
        df2['std_test_accuracy'].iloc[0],
        df3['std_test_accuracy'].iloc[0],
        df4['std_test_accuracy'].iloc[0],
        df5['std_test_accuracy'].iloc[0],
        df6['std_test_accuracy'].iloc[0],      
    ],
    'Mean F1': [
        df1['mean_test_f1'].iloc[0],
        df2['mean_test_f1'].iloc[0],
        df3['mean_test_f1'].iloc[0],
        df4['mean_test_f1'].iloc[0],
        df5['mean_test_f1'].iloc[0],
        df6['mean_test_f1'].iloc[0]
    ],
    'Std F1': [
        df1['std_test_f1'].iloc[0],
        df2['std_test_f1'].iloc[0],
        df3['std_test_f1'].iloc[0],
        df4['std_test_f1'].iloc[0],
        df5['std_test_f1'].iloc[0],
        df6['std_test_f1'].iloc[0],
    ],
})


fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 8), sharex=True)

# First plot: Accuracy
ax1.bar(summary['Model'], summary['Mean Accuracy'], yerr=summary['Std Accuracy'], capsize=5, color='skyblue', edgecolor='black')
ax1.set_ylabel('Mean Accuracy')
ax1.set_ylim(0.95, 1.00)
ax1.set_title('Model Accuracy ± Std')

# Second plot: F1 Score
ax2.bar(summary['Model'], summary['Mean F1'], yerr=summary['Std F1'], capsize=5, color='lightcoral', edgecolor='black')
ax2.set_ylabel('Mean F1 Score')
ax2.set_ylim(0.85, 0.95)
ax2.set_title('Model F1 Score ± Std')

plt.xticks(rotation=20)
plt.tight_layout()
plt.show()