import csv
import itertools

# Define the parameter lists
head_dim = [128,256] # [32, 64, 128, 192, 256]
num_kv_heads_per_card = [2, 4, 8]
num_q_heads_per_card = [4, 8, 16]
batch_size = [256]
input_length =  [1024, 4096, 65536, 131072] # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
context_length = [4096, 16384, 65536, 131072] #  {128, 1k, 4k, 16k, 64k, 128k}. [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
q_output_dtype = ['BF16']
kv_dtype = ['BF16', 'FP8']

# Define the headers for the CSV file
headers = [
    'Batch size',
    'InputLength',
    'OutputLength'
    'ContextLength',
    'Num Q heads per card',
    'Num KV heads per card',
    'Head dim',
    'q-output-dtype',
    'kv-dtype'
]

# Generate all combinations of the parameter lists
combinations = list(itertools.product(
    batch_size,
    input_length,
    context_length,
    num_q_heads_per_card,
    num_kv_heads_per_card,
    head_dim,
    q_output_dtype,
    kv_dtype
))

real_combinations = []

for i in range(len(combinations)):
    # insert output-len at the second position
    combinations[i] = list(combinations[i])
    input_len = combinations[i][1]
    context_len = combinations[i][2]
    num_kv_heads_per_card = combinations[i][4]
    num_q_heads_per_card = combinations[i][3]
    if input_len > context_len or num_kv_heads_per_card > num_q_heads_per_card:
        continue
    output_len = context_len - input_len
    combinations[i].insert(2, output_len)
    real_combinations.append(combinations[i])

# Write the combinations to a CSV file
combinations = real_combinations
with open('combinations.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the header row
    writer.writerow(headers)

    # Write the data rows
    writer.writerows(combinations)

print("CSV file 'combinations.csv' has been generated successfully.")