import sqlite3
import os

# Remove existing database file if it exists (optional - for fresh start)
if os.path.exists("student.db"):
    os.remove("student.db")
    print("Existing database removed")

# Connect to SQLite
connection = sqlite3.connect("student.db")

# Create a cursor object to insert record, create table
cursor = connection.cursor()

# Create the table
table_info = """
CREATE TABLE IF NOT EXISTS STUDENT(
    NAME VARCHAR(25),
    CLASS VARCHAR(25),
    SECTION VARCHAR(25),
    MARKS INT
)
"""

cursor.execute(table_info)
print("Table created successfully")

# Insert some records
try:
    cursor.execute('''INSERT INTO STUDENT values('Krish','Data Science','A',90)''')
    cursor.execute('''INSERT INTO STUDENT values('John','Data Science','B',100)''')
    cursor.execute('''INSERT INTO STUDENT values('Mukesh','Data Science','A',86)''')
    cursor.execute('''INSERT INTO STUDENT values('Jacob','DEVOPS','A',50)''')
    cursor.execute('''INSERT INTO STUDENT values('Dipesh','DEVOPS','A',35)''')
    cursor.execute('''INSERT INTO STUDENT values('Alice','Data Science','B',95)''')
    cursor.execute('''INSERT INTO STUDENT values('Bob','DEVOPS','B',78)''')
    cursor.execute('''INSERT INTO STUDENT values('Charlie','Data Science','A',88)''')
    print("Records inserted successfully")
except sqlite3.IntegrityError as e:
    print(f"Error inserting records: {e}")

# Commit your changes in the database
connection.commit()

# Display all the records
print("\n" + "="*50)
print("The inserted records are:")
print("="*50)
data = cursor.execute('''SELECT * FROM STUDENT''')
for row in data:
    print(row)

# Get table statistics
cursor.execute('''SELECT COUNT(*) FROM STUDENT''')
count = cursor.fetchone()[0]
print("="*50)
print(f"Total number of students: {count}")

cursor.execute('''SELECT AVG(MARKS) FROM STUDENT''')
avg = cursor.fetchone()[0]
print(f"Average marks: {avg:.2f}")
print("="*50)

# Close the connection
connection.close()
print("\nDatabase connection closed successfully")