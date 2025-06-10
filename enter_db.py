import pyodbc
import pandas as pd

info_excel = "/home/sagar/Master_pdfs/emp_info.xlsx"

SERVER="outsystems1.database.windows.net"
DATABASE="OUTSYSTEM_API"
UID="Galaxy"
PWD='OutSystems@123'

try:
    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={SERVER};"
        f"DATABASE={DATABASE};"
        f"UID={UID};"
        f"PWD={PWD}"
    )
    cursor = conn.cursor()
    
    print("Database connection successful")
    
    # Read Excel data
    df = pd.read_excel(info_excel)

    # Iterate over DataFrame rows and insert into DB
    for idx, row in df.iterrows():
        user_name = str(row['user_name']) if pd.notnull(row['user_name']) else ""
        phone_number = str(row['emp_phone_number']) if pd.notnull(row['emp_phone_number']) else ""
        emp_id = str(row['emp_user_office_id']) if pd.notnull(row['emp_user_office_id']) else ""
        emails = str(row['emp_user_emails']) if pd.notnull(row['emp_user_emails']) else ""
        assets_serial_number = str(row['assets_serial_number']) if pd.notnull(row['assets_serial_number']) else ""
        mo_name = str(row['assets_laptop_model']) if pd.notnull(row['assets_laptop_model']) else ""
        com_name = "default"

        insert_query = '''
            INSERT INTO l1_tree (user_name, phone_number, com_name, mo_name, emails, emp_id, assets_serial_number)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        values = (user_name, phone_number, com_name, mo_name, emails, emp_id, assets_serial_number)
        try:
            cursor.execute(insert_query, values)
        except Exception as e:
            print(f"Error inserting row {idx}: {e}")
    conn.commit()
    print("All data inserted successfully.")
except pyodbc.Error as db_err:
    print(f"Database connection error: {db_err}")
