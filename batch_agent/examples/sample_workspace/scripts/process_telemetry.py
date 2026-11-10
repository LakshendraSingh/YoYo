"""Telemetry processing script for batch analytics."""
import csv
import sys

def analyze(csv_path: str):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cpu_vals = [float(r['cpu_usage']) for r in rows]
    avg_cpu = sum(cpu_vals) / len(cpu_vals) if cpu_vals else 0.0

    print(f"Total Records: {len(rows)}")
    print(f"Average CPU: {avg_cpu:.2f}%")
    return avg_cpu

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/telemetry.csv'
    analyze(path)

