import sqlite3
import pandas as pd

# Load your CSV file into a pandas DataFrame
csv_file = 'C:\\Users\\Admin\\Downloads\\project\\project\\processed_data.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file)

# Connect to (or create) the SQLite database file
db_file = 'C:\\Users\\Admin\\Downloads\\project\\project\\bookings.db'  # Desired path for your database file
conn = sqlite3.connect(db_file)

# Save DataFrame to the database as a table named 'bookings'
df.to_sql('bookings', conn, if_exists='replace', index=False)

# Close the connection
conn.close()

print(f"Database file '{db_file}' successfully created and populated with the table 'bookings'.")
