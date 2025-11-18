import pandas as pd
from process_data import get_labels, separate_labels

labels_df = get_labels() # DataFrame with all relavant columns
labels = labels_df['Finding Labels'].apply(separate_labels) # Take label strings and put in multi-hot encoding

