import re

with open('bench_results.txt', 'r', encoding="UTF-16LE") as f:
    content = f.read()

pattern = r'matmul f16 Benchmarks/(hpt|torch|hpt\(builtin\))/(\d+)\s+time:\s+\[\d+\.\d+ ms (\d+\.\d+) ms \d+\.\d+ ms\]'

results = {
    'hpt': [],
    'torch': [],
    'hpt(builtin)': []
}

sizes = []
last_size = None

matches = re.findall(pattern, content)
for test_type, size, median in matches:
    if test_type == 'hpt' and (last_size is None or int(size) != last_size):
        sizes.append(int(size))
        last_size = int(size)
    results[test_type].append(float(median))

print("input size:", sizes)
print("HPT:", results['hpt'])
print("PyTorch:", results['torch'])
print("HPT(builtin):", results['hpt(builtin)'])
