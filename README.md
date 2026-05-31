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


# STEP 4: Multi-Class LGCLP Intensity Estimation

## Description

This step estimates the class-specific trajectory intensity maps in the representation space \((\alpha,p)\). For each target class, a two-dimensional histogram is constructed to approximate the LGCLP mean intensity \(\mu_k(l)\), representing the spatial distribution of stochastic target trajectories.

## Input Code

```python
# STEP 4: Multi-Class LGCLP Intensity Estimation
import numpy as np
import matplotlib.pyplot as plt

bins = 25

fig, ax = plt.subplots(
    1,
    3,
    figsize=(15,4)
)

titles = [
    "Commercial Ship μ₁",
    "Fishing Vessel μ₂",
    "Illegal Vessel μ₃"
]

for k in [1,2,3]:

    idx = classes == k

    H, xedges, yedges = np.histogram2d(
        alpha[idx],
        p[idx],
        bins=bins
    )

    ax[k-1].imshow(
        H.T,
        origin='lower',
        aspect='auto'
    )

    ax[k-1].set_title(
        titles[k-1]
    )

plt.tight_layout()
plt.show()
```

### Estimated Intensities

- \(\mu_1(l)\): Commercial Ship Intensity
- \(\mu_2(l)\): Fishing Vessel Intensity
- \(\mu_3(l)\): Illegal Vessel Intensity

## Output
<img width="1489" height="390" alt="image" src="https://github.com/user-attachments/assets/335079bb-6e9d-4608-9259-a5d39f5810ba" />
<img width="1489" height="390" alt="image" src="https://github.com/user-attachments/assets/dfdc8f67-58df-4193-ab5e-d0279e62d689" />
<img width="1489" height="390" alt="image" src="https://github.com/user-attachments/assets/aba62f03-230f-4931-bb04-852c1473bf5f" />



**Figure 4:** Multi-Class LGCLP Mean Intensity Maps showing the estimated trajectory intensity distributions for Commercial Ships, Fishing Vessels, and Illegal Vessels in the representation space.

# STEP 5: Intensity Uncertainty Estimation

## Description

This step estimates the uncertainty associated with each class-specific LGCLP intensity map. The uncertainty is represented by the standard deviation \(\sigma_k(l)\), which measures the variability of target trajectory intensity within the representation space. These uncertainty maps are later incorporated into the robust intensity formulation to improve sensor placement under intensity estimation uncertainty.

## Input Code

```python
sigma = np.sqrt(H + 1)
```

### Estimated Uncertainty Maps

- \(\sigma_1(l)\): Commercial Ship Uncertainty
- \(\sigma_2(l)\): Fishing Vessel Uncertainty
- \(\sigma_3(l)\): Illegal Vessel Uncertainty

## Output

<img width="1489" height="390" alt="image" src="https://github.com/user-attachments/assets/f710e4ea-78fd-41aa-9b19-df702882a695" />
<img width="1489" height="390" alt="image" src="https://github.com/user-attachments/assets/75ad42c5-5b42-4e23-a462-98ca64ca69fd" />
<img width="1489" height="390" alt="image" src="https://github.com/user-attachments/assets/e18db0ad-0f32-49b0-a6d0-f6176f94cd8d" />


**Figure 5:** Multi-Class Intensity Uncertainty Maps showing the uncertainty distribution for Commercial Ships, Fishing Vessels, and Illegal Vessels in the representation space \((\alpha,p)\).


# STEP 6: Robust Multi-Class Intensity Construction

## Description

This step combines the class-specific mean intensity maps and uncertainty maps to construct the Robust Multi-Class Intensity \(\Lambda_{RM}(l)\). Different target classes are assigned different importance weights, while the uncertainty term improves robustness against intensity estimation errors. The resulting intensity map highlights high-priority and high-uncertainty regions for sensor placement optimization.

## Input Code

```python
# STEP 6: Robust Multi-Class Intensity Construction
import numpy as np
import matplotlib.pyplot as plt

bins = 25

# class weights

w1 = 1      # Commercial
w2 = 2     # Fishing
w3 = 3     # Illegal

kappa = 1



idx = classes == 1

H1, _, _ = np.histogram2d(
    alpha[idx],
    p[idx],
    bins=bins
)

mu1 = H1

sigma1 = np.sqrt(H1 + 1)


idx = classes == 2

H2, _, _ = np.histogram2d(
    alpha[idx],
    p[idx],
    bins=bins
)

mu2 = H2

sigma2 = np.sqrt(H2 + 1)

idx = classes == 3

H3, _, _ = np.histogram2d(
    alpha[idx],
    p[idx],
    bins=bins
)

mu3 = H3

sigma3 = np.sqrt(H3 + 1)


mu_total = mu1 + mu2 + mu3

Lambda_RM = (
    0.6 * mu_total
    +
    0.4 * (
        w1*(mu1 + kappa*sigma1)
        + w2*(mu2 + kappa*sigma2)
        + w3*(mu3 + kappa*sigma3)
    )
)


plt.figure(figsize=(8,6))

plt.imshow(
    Lambda_RM.T,
    origin='lower',
    aspect='auto'
)

plt.colorbar(
    label='ΛRM'
)

plt.title(
    'Robust Multi-Class Intensity Map'
)

plt.xlabel('Alpha (α)')
plt.ylabel('Distance (p)')

plt.show()


print("Maximum ΛRM =", np.max(Lambda_RM))
print("Minimum ΛRM =", np.min(Lambda_RM))
print("Mean ΛRM =", np.mean(Lambda_RM))
```

### Parameters

- \(w_1 = 1\): Commercial Ship Weight
- \(w_2 = 2\): Fishing Vessel Weight
- \(w_3 = 3\): Illegal Vessel Weight
- \(\kappa = 1\): Robustness Coefficient

### Robust Intensity Formulation

\[
\Lambda_{RM}(l)
=
\sum_{k=1}^{K}
w_k
\left(
\mu_k(l)
+
\kappa \sigma_k(l)
\right)
\]

## Output
<img width="655" height="547" alt="image" src="https://github.com/user-attachments/assets/dca159fd-cefe-4da6-8dea-423165d49266" />

**Figure 6:** Robust Multi-Class Intensity Map showing the combined effect of target priority, trajectory density, and intensity uncertainty in the representation space.

**Statistics:**

- Maximum \(\Lambda_{RM}\)
- Minimum \(\Lambda_{RM}\)
- Mean \(\Lambda_{RM}\)

These values provide an overall summary of the robust trajectory intensity distribution used for adaptive sensor placement optimization.

