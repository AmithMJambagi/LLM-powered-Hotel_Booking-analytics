import pandas as pd

# Load the CSV file
df = pd.read_csv("C:\\Users\\Admin\\Downloads\\project\\project\\hotel_bookings.csv")

# Display basic info about the dataset
print("Initial Data Info:")
print(df.info())
print("\nNumber of missing values per column:")
print(df.isnull().sum())

# 1. Handle Missing Values
# Drop columns with more than 30% missing values (if any)
threshold = 0.3
df = df[df.columns[df.isnull().mean() < threshold]]

# Fill missing categorical values with the mode (most frequent value)
for column in df.select_dtypes(include=['object']).columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

# Fill missing numerical values with the median
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    df[column].fillna(df[column].median(), inplace=True)

# 2. Format Inconsistencies Handling
# Convert date columns to datetime format (if any)
date_columns = ['reservation_status_date']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# 3. Remove Duplicates
df.drop_duplicates(inplace=True)

# 4. Standardize Text Columns (if applicable)
# Convert all string columns to lowercase for consistency
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].str.lower()

# 5. Save the cleaned data to a new CSV file
df.to_csv('processed_data.csv', index=False)

# Display final info about the cleaned dataset
print("\nCleaned Data Info:")
print(df.info())
