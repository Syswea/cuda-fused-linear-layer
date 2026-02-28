import random
import struct
import sys

def generate_random_floats():
    rows = 8192
    cols = 4096
    
    sys.stderr.write(f"Generating {rows} rows x {cols} floats...\n")
    
    for i in range(rows):
        row = [random.uniform(-1.0, 1.0) for _ in range(cols)]
        print(' '.join(f'{x:.6f}' for x in row))
        
        if (i + 1) % 500 == 0:
            sys.stderr.write(f"Generated {i + 1}/{rows} rows...\n")

if __name__ == '__main__':
    generate_random_floats()
