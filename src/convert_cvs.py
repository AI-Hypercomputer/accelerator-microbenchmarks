import csv

file_name = "/home/sanbao_google_com/cvs/self microbenchmarks for TPU v7 - add.csv"

try:
    with open(file_name, mode='r', encoding='utf-8') as file:
        # Use DictReader to read the CSV rows as dictionaries
        reader = csv.DictReader(file)
        
        # Iterate over each row in the CSV
        for row in reader:
            # Extract M and N, strip whitespace, and convert to integer
            m_val = int(row['M'].strip())
            n_val = int(row['N'].strip())
            
            # Print in the desired format
            print(f"- {{m: {m_val}, n: {n_val}}}")

except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")