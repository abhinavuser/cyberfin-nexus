import os
import pandas as pd
from utils.db_manager import get_engine

def seed_database_from_csv(csv_directory=".", num_rows=100, skip_rows=1000):
    """
    Reads a specific slice of rows from the local CSV files 
    and appends them to the PostgreSQL database.
    """
    print(f"⏳ Connecting to PostgreSQL database...")
    try:
        engine = get_engine()
        print("✅ Connected!")
    except Exception as e:
        print(f"❌ Failed to connect to the database: {e}")
        return

    print(f"\n📂 Skipping {skip_rows} rows and reading the next {num_rows} rows from your CSV files...")
    
    # We use range(1, skip_rows + 1) to skip the rows but KEEP the 0th row (the header)
    skip_range = range(1, skip_rows + 1) if skip_rows > 0 else None
    
    try:
        data_dict = {
            "accounts": pd.read_csv(os.path.join(csv_directory, "accounts.csv"), skiprows=skip_range, nrows=num_rows),
            "devices": pd.read_csv(os.path.join(csv_directory, "devices.csv"), skiprows=skip_range, nrows=num_rows),
            "acc_dev_map": pd.read_csv(os.path.join(csv_directory, "acc_dev_map.csv"), skiprows=skip_range, nrows=num_rows),
            "cyber_events": pd.read_csv(os.path.join(csv_directory, "cyber_events.csv"), skiprows=skip_range, nrows=num_rows),
            "transactions": pd.read_csv(os.path.join(csv_directory, "transactions.csv"), skiprows=skip_range, nrows=num_rows),
            "mule_rings": pd.read_csv(os.path.join(csv_directory, "mule_rings.csv"), skiprows=skip_range, nrows=num_rows)
        }
        
        # Ensure timestamp columns are parsed as datetime to match the Postgres schema expectations
        data_dict["cyber_events"]["timestamp"] = pd.to_datetime(data_dict["cyber_events"]["timestamp"])
        data_dict["transactions"]["timestamp"] = pd.to_datetime(data_dict["transactions"]["timestamp"])
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find one of the required CSV files.")
        print(f"Details: {e}")
        return
    except Exception as e:
        print(f"❌ Error parsing CSV files: {e}")
        return

    print(f"\n💾 Appending these {num_rows} rows to the database tables...")
    for table_name, df in data_dict.items():
        print(f"  -> Appending to table: {table_name} (+{len(df)} rows)")
        # if_exists="append" adds these new rows to the existing tables
        df.to_sql(table_name, engine, if_exists="append", index=False)
        
    print(f"\n✅ Successfully appended {num_rows} new rows to PostgreSQL database!")
    print("Go to your Streamlit dashboard and click the '🔄 Refresh Data' button!")

if __name__ == "__main__":
    seed_database_from_csv(num_rows=100, skip_rows=1000)
