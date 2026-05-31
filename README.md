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


# STEP 2: Trajectory Representation Space Transformation

## Description
This step transforms each trajectory from the Cartesian space \((x,y)\) into the representation space \((\alpha,p)\). Here, \(\alpha\) represents the trajectory orientation, while \(p\) represents the shortest distance from the origin. This transformation converts line trajectories into points, enabling efficient LGCLP modeling and optimization.

## Input Code
```python
# STEP 2: Trajectory Representation Space Transformation
import numpy as np
import matplotlib.pyplot as plt

# Transformation

alpha = np.pi/2 + np.arctan(m)

p = b / np.sqrt(1 + m**2)

# Plot Representation Space

plt.figure(figsize=(8,6))

plt.scatter(
    alpha,
    p,
    s=25,
    color='blue'
)

plt.title("Trajectory Representation Space")
plt.xlabel("Alpha (α)")
plt.ylabel("Distance (p)")
plt.grid(True)

plt.show()

# Optional check

print("Alpha Range:")
print(alpha.min(), alpha.max())

print("\nP Range:")
print(p.min(), p.max())
```

## Output
<img width="698" height="547" alt="image" src="https://github.com/user-attachments/assets/e69f8f56-63be-4c6d-86f4-cee23e30e680" />

**Figure 2:** Trajectory Representation Space showing each vessel trajectory as a point in the \((\alpha,p)\) domain.

# STEP 3: Multi-Class Trajectory Classification

## Description

This step classifies the transformed trajectories into multiple target classes to represent different vessel categories. Each trajectory is assigned to one of three classes: Commercial Ships, Fishing Vessels, or Illegal Vessels. This classification enables the proposed RMABC framework to model class-specific trajectory intensities and prioritize high-risk targets during sensor placement optimization.

## Input Code

```python
#STEP 3: Multi-Class Trajectory Classification
import numpy as np
import matplotlib.pyplot as plt


classes = np.random.choice(
    [1,2,3],
    size=len(alpha),
    p=[0.6,0.3,0.1]
)

plt.figure(figsize=(8,6))

colors = {
    1:'blue',
    2:'green',
    3:'red'
}

labels = {
    1:'Commercial Ship',
    2:'Fishing Vessel',
    3:'Illegal Vessel'
}

for k in [1,2,3]:

    idx = classes == k

    plt.scatter(
        alpha[idx],
        p[idx],
        color=colors[k],
        label=labels[k],
        s=30
    )

plt.title(
    "Multi-Class Trajectory Representation"
)

plt.xlabel("Alpha (α)")
plt.ylabel("Distance (p)")

plt.legend()
plt.grid(True)

plt.show()


for k in [1,2,3]:
    print(
        labels[k],
        ":",
        np.sum(classes==k)
    )
```

### Class Distribution

- Class 1: Commercial Ships (60%)
- Class 2: Fishing Vessels (30%)
- Class 3: Illegal Vessels (10%)

## Output
<img width="698" height="547" alt="image" src="https://github.com/user-attachments/assets/042ea2e4-16e0-4d45-951c-0cba734a3d13" />


**Figure 3:** Multi-Class Trajectory Representation showing the transformed trajectories in the representation space \((\alpha,p)\) categorized into Commercial Ships, Fishing Vessels, and Illegal Vessels.
