import os

tmp_path = '/tmp/bit_logs/'

for folder in os.listdir(tmp_path):
    for file in os.listdir(f'{tmp_path}{folder}'):
        print(f'{file}\n')
        if '.png' in file:
            os.remove(f'{tmp_path}{folder}/{file}')
            print(f"deleted {tmp_path}{folder}/{file}")
            continue