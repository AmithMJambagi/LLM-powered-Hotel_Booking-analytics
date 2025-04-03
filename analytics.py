import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

# Connect to the database
engine = create_engine("sqlite:///bookings.db")

# Load data into DataFrame
df = pd.read_sql("SELECT * FROM bookings", engine)

#  Revenue Trends Over Time
def revenue_trends():
    revenue_df = df.groupby("arrival_date")["adr"].sum().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=revenue_df, x="arrival_date", y="adr", marker="o")
    plt.title("Revenue Trends Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Revenue (ADR)")
    plt.xticks(rotation=45)
    plt.show()

#  Cancellation Rate
def cancellation_rate():
    canceled = df[df["is_canceled"] == 1].shape[0]
    total = df.shape[0]
    rate = (canceled / total) * 100
    print(f"📊 Cancellation Rate: {rate:.2f}%")

#  Geographical Distribution of Bookings
def geo_distribution():
    country_counts = df["country"].value_counts().head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=country_counts.index, y=country_counts.values, palette="viridis")
    plt.title("Top 10 Booking Countries")
    plt.xlabel("Country")
    plt.ylabel("Number of Bookings")
    plt.xticks(rotation=45)
    plt.show()

#  Booking Lead Time Distribution
def lead_time_distribution():
    plt.figure(figsize=(12, 6))
    sns.histplot(df["lead_time"], bins=50, kde=True, color="blue")
    plt.title("Booking Lead Time Distribution")
    plt.xlabel("Days Before Booking")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    revenue_trends()
    cancellation_rate()
    geo_distribution()
    lead_time_distribution()
