import subprocess
hash_bit_values = [16,32,64,128]
for hash_bit in hash_bit_values:
    command = [
        'python','main_coco.py',
        '--hash_bit', str(hash_bit),
    ]
    subprocess.run(command)