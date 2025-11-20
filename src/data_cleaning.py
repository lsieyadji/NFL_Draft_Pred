import pandas as pd
import numpy as np
import os

# Set up file paths
data_dir = r'C:\Users\cash\coding_projects\NFL_Draft_Pred\data'
input_file = os.path.join(data_dir, 'NFL.csv')
output_file = os.path.join(data_dir, 'NFL_cleaned.csv')

# Load the data
print("Loading data...")
df = pd.read_csv(input_file)
print(f"Original data shape: {df.shape}")

# Step 1: Drop specified columns
columns_to_drop = ['School', 'Player', 'Year', 'Drafted..tm.rnd.yr.', 'Player_Type',
                   'Shuttle', 'Agility_3cone']
df_clean = df.drop(columns=columns_to_drop, errors='ignore')
print(f"After dropping columns: {df_clean.shape}")

# Step 2: One-hot encode Position and Position_Type (convert to 1/0)
print("One-hot encoding Position and Position_Type...")
position_dummies = pd.get_dummies(df_clean['Position'], prefix='Pos').astype(int)
position_type_dummies = pd.get_dummies(df_clean['Position_Type'], prefix='PosType').astype(int)

# Combine with main dataframe
df_clean = pd.concat([df_clean, position_dummies, position_type_dummies], axis=1)

# Drop the original categorical columns
df_clean = df_clean.drop(['Position', 'Position_Type'], axis=1)
print(f"After one-hot encoding: {df_clean.shape}")

# Step 3: Binary encode Drafted column
print("Encoding Drafted column...")
df_clean['Drafted'] = df_clean['Drafted'].map({'Yes': 1, 'No': 0})

# Step 4: Position-based median imputation for Combine metrics
print("Performing position-based median imputation...")
combine_metrics = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump']

# Store original position column for imputation (we'll use the one from original df)
positions = df['Position']

for metric in combine_metrics:
    print(f"  Imputing {metric}...")
    missing_before = df_clean[metric].isna().sum()

    # Calculate median by position
    position_medians = df.groupby('Position')[metric].median()

    # Impute missing values based on position
    for position in positions.unique():
        position_mask = (positions == position) & (df_clean[metric].isna())
        if position_mask.any():
            median_val = position_medians.get(position, df_clean[metric].median())
            df_clean.loc[position_mask, metric] = median_val

    missing_after = df_clean[metric].isna().sum()
    print(f"    Fixed {missing_before - missing_after} missing values")

# Step 5: Check for any remaining missing values and handle them
print("Checking for remaining missing values...")
remaining_missing = df_clean.isna().sum().sum()
if remaining_missing > 0:
    print(f"  There are {remaining_missing} remaining missing values")
    # Fill any stragglers with overall median
    for col in df_clean.columns:
        if df_clean[col].isna().any() and pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

# Step 6: Final data check
print("\nFinal data check:")
print(f"Final dataset shape: {df_clean.shape}")
print(f"Missing values: {df_clean.isna().sum().sum()}")
print(f"Drafted distribution:")
print(df_clean['Drafted'].value_counts())

# Verify that one-hot encoded columns are 1/0
print("\nSample of one-hot encoded columns (should be 1/0):")
pos_columns = [col for col in df_clean.columns if col.startswith('Pos_') or col.startswith('PosType_')]
print(df_clean[pos_columns[:5]].head())  # Show first 5 one-hot columns

# Step 7: Save the cleaned dataset
print(f"\nSaving cleaned data to: {output_file}")
df_clean.to_csv(output_file, index=False)

print("Data preprocessing complete! ✅")
print(f"Original data: {df.shape}")
print(f"Cleaned data: {df_clean.shape}")