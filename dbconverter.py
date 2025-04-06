import sqlite3
import pandas as pd

# Load CSV
csv_file = 'C:\\Users\\Admin\\Desktop\\LLMbasedhotelbookinganalytics\\project\\project\\processed_data.csv'
df = pd.read_csv(csv_file)

# ------------------ CLEANING STARTS HERE ------------------

# Drop rows with missing critical booking information
df.dropna(subset=[
    "arrival_date_year", 
    "arrival_date_month", 
    "adults", 
    "children", 
    "babies"
], inplace=True)

# Convert 'arrival_date_month' (string like "July") to integer month (7)
df["arrival_date_month"] = df["arrival_date_month"].apply(
    lambda x: pd.to_datetime(x, format="%B").month if isinstance(x, str) else x
)

# Convert year to integer
df["arrival_date_year"] = df["arrival_date_year"].astype(int)

# Replace missing numerical values (like children/babies) with 0
numerical_cols = ["adults", "children", "babies", "adr", "lead_time", 
                  "stays_in_weekend_nights", "stays_in_week_nights", 
                  "booking_changes", "required_car_parking_spaces", 
                  "total_of_special_requests", "agent", "company", 
                  "days_in_waiting_list", "previous_cancellations", 
                  "previous_bookings_not_canceled", "arrival_date_day_of_month", 
                  "arrival_date_week_number"]
for col in numerical_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Ensure consistency for string fields (strip spaces)
string_cols = ["hotel", "meal", "country", "market_segment", "distribution_channel", 
               "reserved_room_type", "assigned_room_type", "deposit_type", 
               "customer_type", "reservation_status"]
for col in string_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# ------------------ SAVE TO SQLITE ------------------

# Connect to SQLite
db_file = "C:\\Users\\Admin\\Desktop\\LLMbasedhotelbookinganalytics\\project\\project\\bookings.db"
conn = sqlite3.connect(db_file)

# Save to DB (overwrite if exists)
df.to_sql('bookings', conn, if_exists='replace', index=False)
conn.close()

print(f"✅ Cleaned data saved to '{db_file}' in table 'bookings'")
