import pandas as pd
import gzip
import numpy as np

def weak_supervision_label(question):
    if not isinstance(question, str):
        #error handleing 
        return 

    question = question.lower()
    
    
    what_like = ["what", "which", "who", "where", "when"]
    how_like = ["how", "in what way", "by what means"]
    why_like = ["why", "for what reason", "how come"]
    
    
    label = np.array([0.0, 0.0, 0.0])
    
    # Define weights for the word position in the question
    weight_start = 1.5
    weight_middle = 1.0
    weight_end = 2.0
    
    # Helper function to apply weights based on position
    def apply_weights(word_list, weight):
        nonlocal label
        for word in word_list:
            if word in question:
                if question.endswith(word):
                    label += np.array([weight_end if word in what_like else 0, 
                                        weight_end if word in how_like else 0, 
                                        weight_end if word in why_like else 0])
                elif question.startswith(word):
                    label += np.array([weight_start if word in what_like else 0, 
                                        weight_start if word in how_like else 0, 
                                        weight_start if word in why_like else 0])
                else: 
                    label += np.array([weight_middle if word in what_like else 0, 
                                        weight_middle if word in how_like else 0, 
                                        weight_middle if word in why_like else 0])
    
    # Apply the weights for each category
    apply_weights(what_like, weight_middle)
    apply_weights(how_like, weight_middle)
    apply_weights(why_like, weight_middle)
    
    # Normalize the label to sum to 1
    if np.sum(label) > 0:
        label /= np.sum(label)
    else:
        label = np.array([0.33, 0.33, 0.33])  # Uncertain label
    
    return label.tolist()

def filter_data(df):
    """Filters rows based on the certainty of labels."""
    # Convert the DataFrame to a NumPy array
    data_array = df.to_numpy()

    equal_to_33 = []
    not_equal_to_33 = []

    for row in data_array:
        # Check if the labels are a list and not None
        labels = row[-1]  # Assuming 'labels' is the last column
        if labels is None or not isinstance(labels, list):
            print(f"Skipping row due to invalid labels: {labels}")
            continue
        
        # Filter based on the label values
        if all(label == 0.33 for label in labels):
            equal_to_33.append(row)
        else:
            not_equal_to_33.append(row)

    # Convert back to DataFrames
    df_equal_to_33 = pd.DataFrame(equal_to_33, columns=df.columns)
    df_not_equal_to_33 = pd.DataFrame(not_equal_to_33, columns=df.columns)

    return df_equal_to_33, df_not_equal_to_33

def process_data(path):
    """Processes the data by loading, labeling, and filtering."""
    # Read the pickle file without assuming it's gzipped
    data = pd.read_pickle(path, compression=None)

    questions = data['question']
    data_lists = []

    for question in questions:
        label = weak_supervision_label(question)
        data_lists.append([question, label])

    df = pd.DataFrame(data_lists, columns=['question', 'labels'])
    unsure, sure = filter_data(df)

    # Extract filtered questions and labels
    filtered_questions = sure['question']
    filtered_labels = sure['labels']

    return filtered_questions, filtered_labels
