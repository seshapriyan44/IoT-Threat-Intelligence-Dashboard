import subprocess
import sys

processes = []

for i in range(1, 10):
    print(f"Starting Client {i}")

    p = subprocess.Popen(
        [sys.executable, "fl_client.py", str(i)]
    )

    processes.append(p)

print("Launched 9 clients")

for p in processes:
    p.wait()