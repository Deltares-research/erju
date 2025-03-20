import sqlite3
import pandas as pd


import os
print('this is the current working directory')
print(os.getcwd())


def connect_db(db_path):
    """Connects to the SQLite database and returns the connection."""
    try:
        conn = sqlite3.connect(db_path)
        print("Connected to database successfully.")
        return conn
    except sqlite3.Error as e:
        print("Error connecting to database:", e)
        return None


def list_tables(conn):
    """Lists all tables in the SQLite database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [table[0] for table in tables]


def get_table_schema(conn, table_name):
    query = f'PRAGMA table_info("{table_name}")'  # Enclose in double quotes
    df_schema = pd.read_sql_query(query, conn)
    return df_schema



def preview_table(conn, table_name, limit=5):
    """Prints the first few rows of a table."""
    cursor = conn.cursor()

    # Enclose table name in double quotes to handle special characters
    query = f'SELECT * FROM "{table_name}" LIMIT {limit};'

    try:
        cursor.execute(query)
        rows = cursor.fetchall()

        print(f"Table: {table_name}")
        for row in rows:
            print(row)
        print("\n")
    except sqlite3.Error as e:
        print(f"Error accessing table {table_name}: {e}")


if __name__ == "__main__":

    # Database path
    db_path = r"P:\11207352-stem\database\Wielrondheid_132887.db"

    conn = connect_db(db_path)

    if conn:
        tables = list_tables(conn)
        print("Tables found:", tables)

        # Show schema and preview data for each table
        for table in tables:
            get_table_schema(conn, table)
            preview_table(conn, table)

        conn.close()


    # Table name (properly quoted)
    holten_1 = '20240524_20240901_Holten_Meetjournal_MP8_Holten_zuid_4m_C'
    holten_2 = '20240901_20241001_Holten_Meetjournal_MP8_Holten_zuid_4m_C'

    # Open connection to the SQLite database
    conn = sqlite3.connect(db_path)

    # Read the table (ensure the name is enclosed in double quotes)
    query = f'SELECT * FROM "{holten_1}"'
    df_h1 = pd.read_sql_query(query, conn)

    # Read the table (ensure the name is enclosed in double quotes)
    query = f'SELECT * FROM "{holten_2}"'
    df_h2 = pd.read_sql_query(query, conn)

    # Read the table (ensure the name is enclosed in double quotes)
    query = f'SELECT * FROM "{'metadata'}"'
    df_metadata = pd.read_sql_query(query, conn)

    # Close the connection
    conn.close()

