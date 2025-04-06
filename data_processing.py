import pandas as pd

# Load the CSV file
df = pd.read_csv("C:\\Users\\Admin\\Desktop\\LLMbasedhotelbookinganalytics\\project\\project\\hotel_bookings.csv")

# Show initial info
print("Initial Data Info:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())

# 1. Drop columns with >30% missing values
threshold = 0.3
df = df[df.columns[df.isnull().mean() < threshold]]

# 2. Handle missing values
# Fill categorical (object) columns with mode
for column in df.select_dtypes(include=['object']).columns:
    if df[column].isnull().any():
        mode_value = df[column].mode()[0]
        df[column] = df[column].fillna(mode_value)

# Fill numerical columns with median
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    if df[column].isnull().any():
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)

# 3. Convert date columns to datetime
date_columns = ['reservation_status_date']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Optional: create a real datetime column from year/month/day if available
if all(col in df.columns for col in ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']):
    try:
        df["arrival_date_month"] = df["arrival_date_month"].apply(
            lambda x: pd.to_datetime(str(x), format='%B').month if isinstance(x, str) else int(x)
        )
        df["arrival_date"] = pd.to_datetime(dict(
            year=df["arrival_date_year"].astype(int),
            month=df["arrival_date_month"].astype(int),
            day=df["arrival_date_day_of_month"].astype(int)
        ), errors='coerce')
    except Exception as e:
        print(f"❌ Failed to convert arrival date: {e}")

# 4. Drop duplicates
df.drop_duplicates(inplace=True)

# 5. Normalize text fields
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].astype(str).str.strip().str.lower()

# 6. Save cleaned data
processed_csv = "C:\\Users\\Admin\\Desktop\\LLMbasedhotelbookinganalytics\\project\\project\\processed_data.csv"
df.to_csv(processed_csv, index=False)

print("\n✅ Cleaned Data Info:")
print(df.info())
print(f"\n✅ Cleaned CSV saved to {processed_csv}")
