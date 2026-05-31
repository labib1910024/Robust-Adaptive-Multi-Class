# STEP 1: Synthetic AIS Trajectory Generation

## Description
This step generates 100 synthetic AIS-like vessel trajectories using random slopes and intercepts. Each trajectory is modeled as a straight line and represents a possible target path in the surveillance region.

## Input Code
```python
# STEP 1: Synthetic AIS Trajectory Generation
import numpy as np
import matplotlib.pyplot as plt

# Number of trajectories
N = 100

# Generate random slopes and intercepts
m = np.random.uniform(-2, 2, N)
b = np.random.uniform(-50, 50, N)

# Plot trajectories
plt.figure(figsize=(8,6))

x = np.linspace(-50, 50, 100)

for i in range(N):
    y = m[i] * x + b[i]
    plt.plot(x, y, alpha=0.3)

plt.title("Synthetic AIS Trajectories")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)

plt.show()
```

## Output
<img width="707" height="547" alt="image" src="https://github.com/user-attachments/assets/e3821c56-f157-4012-bf82-b1fe039fad13" />

**Figure 1:** Synthetic AIS Trajectories showing randomly generated vessel paths crossing the monitoring area.
```
