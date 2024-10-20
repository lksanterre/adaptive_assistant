import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, kstest, wasserstein_distance
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

# Load reference and test data (questions and predicted labels)
reference_data = pd.read_pickle('/Users/lancesanterre/pipeline_edu/data/processed/pipeline_and_data.pkl')
test_data = pd.read_pickle('/Users/lancesanterre/pipeline_edu/data/predictions/predictions.pkl')

# Create a DataFrame with the first 5000 questions and predicted labels
test_data = pd.DataFrame({'question': reference_data['question'][0:5000], 'labels': list(test_data)})

# Ensure the reference labels are simple class labels, not lists
reference_data['predicted_label'] = np.argmax(np.vstack(reference_data['labels']), axis=1)

# Convert probability vectors in 'labels' column to predicted labels for test data
test_data['predicted_label'] = np.argmax(np.vstack(test_data['labels']), axis=1)

# Tokenize text for reference and test sets
vectorizer = CountVectorizer()
reference_text_tokens = vectorizer.fit_transform(reference_data['question'][0:5000])
test_text_tokens = vectorizer.transform(test_data['question'])

# Drift metric: Compare word count distribution (Jensen-Shannon or Wasserstein distance)
ref_word_counts = np.sum(reference_text_tokens.toarray(), axis=0)
test_word_counts = np.sum(test_text_tokens.toarray(), axis=0)

# Calculate Wasserstein distance for token distributions
wasserstein_text = wasserstein_distance(ref_word_counts, test_word_counts)

# Drift test: KS test for token distributions
ks_test_text = kstest(ref_word_counts, test_word_counts)

# Predicted label distributions (now simple class labels, not lists)
reference_label_dist = reference_data['predicted_label'][0:5000].value_counts(normalize=True)
test_label_dist = test_data['predicted_label'].value_counts(normalize=True)

# Align indices of test_label_dist and reference_label_dist
aligned_test_label_dist = test_label_dist.reindex(reference_label_dist.index, fill_value=0)

# Drift metric: Chi-square test for predicted label distributions
chi_square_labels = chisquare(aligned_test_label_dist, reference_label_dist)

# Visualize distributions for predicted labels
plt.figure(figsize=(10, 5))
reference_label_dist.plot(kind='bar', alpha=0.5, label='Reference', color='blue')
aligned_test_label_dist.plot(kind='bar', alpha=0.5, label='Test', color='orange')
plt.title('Predicted Label Distribution')
plt.legend()
plt.show()

# If true labels available, calculate performance metric (e.g., accuracy)
if 'true_label' in test_data.columns:
    y_true = test_data['true_label']  # Assuming you have the true labels
    y_pred = test_data['predicted_label']
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")

# Log the results
print(f"Wasserstein Distance (question text): {wasserstein_text}")
print(f"KS Test Result (question text): {ks_test_text}")
print(f"Chi-square Test Result (predicted labels): {chi_square_labels}")

# Create a dictionary to store the results
drift_results = {
    'Wasserstein Distance (question text)': [wasserstein_text],
    'KS Test Statistic (question text)': [ks_test_text.statistic],
    'KS Test p-value (question text)': [ks_test_text.pvalue],
    'Chi-square Statistic (predicted labels)': [chi_square_labels.statistic],
    'Chi-square p-value (predicted labels)': [chi_square_labels.pvalue]
}

# Convert the dictionary into a DataFrame
drift_results_df = pd.DataFrame(drift_results)

# Save the DataFrame as a CSV file
output_csv_path = '/Users/lancesanterre/pipeline_edu/data/drift/drift_scores.csv'
drift_results_df.to_csv(output_csv_path, index=False)

print(f"Drift scores saved to {output_csv_path}")
