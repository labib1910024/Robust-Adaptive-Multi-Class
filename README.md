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
# STEP 7: Original LGCLP-Based Sensor Placement

## Description

This step implements the sensor placement strategy used in the original paper. Sensor locations are selected based on the estimated LGCLP intensity map, where sensors are placed in the regions with the highest trajectory intensity. The objective is to maximize target detection by covering the most frequently occurring target trajectories.

## Input Code

```python
# STEP 7: Original LGCLP-Based Sensor Placement
import numpy as np
import matplotlib.pyplot as plt


H_all, _, _ = np.histogram2d(
    alpha,
    p,
    bins=25
)


num_sensors = 10

flat_indices = np.argsort(
    H_all.ravel()
)[-num_sensors:]

sensor_x = []
sensor_y = []

for idx in flat_indices:

    x = idx // H_all.shape[1]

    y = idx % H_all.shape[1]

    sensor_x.append(x)

    sensor_y.append(y)


plt.figure(figsize=(8,6))

plt.imshow(
    H_all.T,
    origin='lower',
    aspect='auto'
)

plt.scatter(
    sensor_x,
    sensor_y,
    c='red',
    s=150,
    marker='o',
    label='Sensors'
)

plt.colorbar(
    label='Intensity'
)

plt.title(
    'Original Paper Sensor Placement'
)

plt.xlabel('Alpha (α)')
plt.ylabel('Distance (p)')

plt.legend()

plt.show()

print("Sensor X:",sensor_x)
print("Sensor Y:",sensor_y)
```

### Parameters

- Number of Sensors: 10
- Intensity Map: \( \lambda(l) \)
- Placement Strategy: Highest Intensity Locations

## Output
<img width="668" height="547" alt="image" src="https://github.com/user-attachments/assets/c6fb18e5-3e92-4965-916b-4db70444b3b3" />


**Figure 7:** Original LGCLP-Based Sensor Placement showing the sensor locations selected from the highest-density regions of the estimated trajectory intensity map.

**Results:**

- Sensor X Coordinates
- Sensor Y Coordinates

These sensor locations represent the static deployment generated by the original LGCLP-based optimization framework.


# STEP 8: RMABC Sensor Placement

## Description

This step performs sensor placement using the proposed Robust Adaptive Multi-Class Barrier Coverage (RMABC) framework. Instead of relying only on trajectory density, sensor locations are selected based on the Robust Multi-Class Intensity Map \(\Lambda_{RM}(l)\), which incorporates target priority, trajectory intensity, and intensity estimation uncertainty. As a result, sensors are deployed in regions that are both high-risk and highly uncertain.

## Input Code

```python
#STEP 8: RMABC Sensor Placement
import numpy as np
import matplotlib.pyplot as plt

num_sensors = 10

flat_indices = np.argsort(
    Lambda_RM.ravel()
)[-num_sensors:]

sensor_x_rm = []
sensor_y_rm = []

for idx in flat_indices:

    x = idx // Lambda_RM.shape[1]

    y = idx % Lambda_RM.shape[1]

    sensor_x_rm.append(x)
    sensor_y_rm.append(y)


plt.figure(figsize=(8,6))

plt.imshow(
    Lambda_RM.T,
    origin='lower',
    aspect='auto'
)

plt.scatter(
    sensor_x_rm,
    sensor_y_rm,
    c='white',
    s=180,
    marker='o',
    edgecolors='black',
    label='RMABC Sensors'
)

plt.colorbar(
    label='ΛRM'
)

plt.title(
    'RMABC Sensor Placement'
)

plt.xlabel('Alpha (α)')
plt.ylabel('Distance (p)')

plt.legend()

plt.show()

print("RMABC Sensor X:", sensor_x_rm)
print("RMABC Sensor Y:", sensor_y_rm)
```

### Parameters

- Number of Sensors: 10
- Robust Intensity Map: \(\Lambda_{RM}(l)\)
- Target Weights:
  - Commercial Ship = 1
  - Fishing Vessel = 2
  - Illegal Vessel = 3
- Robustness Coefficient: \(\kappa = 1\)

### Optimization Objective

\[
a^{*}
=
\arg\max \nu_{RM}(a)
\]

where sensor locations are selected to maximize the robust void probability using the robust multi-class intensity.

## Output
<img width="668" height="547" alt="image" src="https://github.com/user-attachments/assets/4533af53-5996-455f-aae5-bba569e24a03" />


**Figure 8:** RMABC Sensor Placement showing sensor locations selected from the Robust Multi-Class Intensity Map \(\Lambda_{RM}(l)\). The deployment prioritizes high-risk target classes and regions with greater intensity uncertainty.

**Results:**

- RMABC Sensor X Coordinates
- RMABC Sensor Y Coordinates

These sensor locations represent the robust multi-class deployment generated by the proposed RMABC framework.



# STEP 9: Detection Rate Evaluation

## Description

This step evaluates and compares the target detection performance of the Original LGCLP-Based Sensor Placement and the proposed RMABC Sensor Placement. A target trajectory is considered detected if its mapped point in the representation space lies within the sensing radius of at least one sensor. The detection rate is computed as the ratio of detected trajectories to the total number of trajectories.

## Input Code

```python
# STEP 9: Detection Rate Evaluation
import numpy as np
import matplotlib.pyplot as plt


radius = 6



detected_original = 0

for i in range(len(alpha)):

    x = int(alpha[i]/alpha.max()*24)

    y = int((p[i]-p.min())/
            (p.max()-p.min())*24)

    for sx,sy in zip(
            sensor_x,
            sensor_y):

        d = np.sqrt(
            (x-sx)**2 +
            (y-sy)**2
        )

        if d < radius:

            detected_original += 1

            break


detected_rmabc = 0

for i in range(len(alpha)):

    x = int(alpha[i]/alpha.max()*24)

    y = int((p[i]-p.min())/
            (p.max()-p.min())*24)

    for sx,sy in zip(
            sensor_x_rm,
            sensor_y_rm):

        d = np.sqrt(
            (x-sx)**2 +
            (y-sy)**2
        )

        if d < radius:

            detected_rmabc += 1

            break


rate_original = \
detected_original/len(alpha)

rate_rmabc = \
detected_rmabc/len(alpha)

print(
"Original Detection:",
rate_original
)

print(
"RMABC Detection:",
rate_rmabc
)


plt.figure(figsize=(6,5))

plt.bar(
    ['Original',
     'RMABC'],
    [rate_original,
     rate_rmabc]
)

plt.ylabel(
'Detection Rate'
)

plt.title(
'Detection Rate Comparison'
)

plt.show()
```

### Detection Criterion

A target is detected when:

\[
d < r
\]

where:

- \(d\) = Euclidean distance between trajectory point and sensor
- \(r = 6\) = sensing radius

### Detection Rate

\[
\text{Detection Rate}
=
\frac{\text{Detected Trajectories}}
{\text{Total Trajectories}}
\]

## Output

<img width="536" height="451" alt="image" src="https://github.com/user-attachments/assets/1734ffc3-6ce2-4198-805f-094764654e7c" />

**Figure 9:** Detection Rate Comparison between the Original LGCLP-Based Sensor Placement and the proposed RMABC Sensor Placement.

**Results:**

- Original Detection Rate
- RMABC Detection Rate

A higher detection rate indicates better target coverage and improved surveillance performance.


# STEP 10: Void Probability Analysis

## Description

This step evaluates the void probability for both the Original LGCLP-Based Sensor Placement and the proposed RMABC framework. The void probability represents the probability that all target trajectories are successfully detected, or equivalently, that no target remains undetected within the surveillance region. A higher void probability indicates better overall barrier coverage performance.

## Input Code

```python
# STEP 10: Void Probability Analysis
import numpy as np
import matplotlib.pyplot as plt


miss_original = 1 - rate_original

void_original = np.exp(
    -miss_original
)


miss_rmabc = 1 - rate_rmabc

void_rmabc = np.exp(
    -miss_rmabc
)

print(
    "Original Void Probability =",
    void_original
)

print(
    "RMABC Void Probability =",
    void_rmabc
)


plt.figure(figsize=(6,5))

plt.bar(
    ['Original','RMABC'],
    [void_original,
     void_rmabc]
)

plt.ylabel(
    'Void Probability'
)

plt.title(
    'Void Probability Comparison'
)

plt.show()
```

### Void Probability Formulation

\[
\nu = e^{-P_{miss}}
\]

where:

- \(P_{miss} = 1 - \text{Detection Rate}\)
- \(\nu\) = Void Probability

A larger value of \(\nu\) indicates a lower probability of missed targets.

## Output
<img width="536" height="451" alt="image" src="https://github.com/user-attachments/assets/3598087d-1361-4f84-b5cd-bdc0b5a2e5b3" />

**Figure 10:** Void Probability Comparison between the Original LGCLP-Based Sensor Placement and the proposed RMABC Sensor Placement.

**Results:**

- Original Void Probability
- RMABC Void Probability

The comparison illustrates the effectiveness of the proposed RMABC framework in reducing undetected target trajectories and improving overall surveillance coverage.

# STEP 11: GBAC-Based Adaptive Sensor Optimization

## Description

This step implements the Gradient-Based Adaptive Controller (GBAC) to continuously refine the RMABC sensor locations. Starting from the initial RMABC deployment, the controller computes the gradient of the robust objective function and iteratively adjusts sensor positions toward regions with higher robust multi-class intensity. A Gaussian-smoothed intensity map is used to improve optimization stability and convergence.

## Input Code

```python
# STEP 11: GBAC-Based Adaptive Sensor Optimization
import numpy as np
import matplotlib.pyplot as plt


eta = 0.05
iterations = 50

sensor_x = np.array(sensor_x_rm, dtype=float)
sensor_y = np.array(sensor_y_rm, dtype=float)

history = []

from scipy.ndimage import gaussian_filter

Lambda_RM_smooth = gaussian_filter(
    Lambda_RM,
    sigma=1.5
)


def objective(sx, sy):

    score = 0

    for x, y in zip(sx, sy):

        xi = int(np.clip(round(x),0,24))
        yi = int(np.clip(round(y),0,24))

        score += Lambda_RM_smooth[xi,yi]

    return score



for step in range(iterations):

    current = objective(
        sensor_x,
        sensor_y
    )

    history.append(current)

    grad_x = np.zeros(len(sensor_x))
    grad_y = np.zeros(len(sensor_y))

    eps = 0.5

    for i in range(len(sensor_x)):

        plus = sensor_x.copy()
        minus = sensor_x.copy()

        plus[i] += eps
        minus[i] -= eps

        grad_x[i] = (
            objective(plus,sensor_y)
            -
            objective(minus,sensor_y)
        )/(2*eps)

        # y gradient

        plus = sensor_y.copy()
        minus = sensor_y.copy()

        plus[i] += eps
        minus[i] -= eps

        grad_y[i] = (
            objective(sensor_x,plus)
            -
            objective(sensor_x,minus)
        )/(2*eps)

    

    sensor_x += eta*grad_x
    sensor_y += eta*grad_y

    sensor_x = np.clip(
        sensor_x,
        0,
        24
    )

    sensor_y = np.clip(
        sensor_y,
        0,
        24
    )


plt.figure(figsize=(8,6))

plt.imshow(
    Lambda_RM_smooth.T,
    origin='lower',
    aspect='auto'
)

plt.scatter(
    sensor_x,
    sensor_y,
    c='red',
    s=180,
    label='GBAC Sensors'
)

plt.colorbar(label='ΛRM')

plt.title(
    'Adaptive RMABC Sensor Placement'
)

plt.legend()

plt.show()


plt.figure(figsize=(8,5))

plt.plot(
    history,
    linewidth=2
)

plt.xlabel('Iteration')
plt.ylabel('Objective')

plt.title(
    'GBAC Convergence'
)

plt.grid(True)

plt.show()

print(
    "Initial Objective =",
    history[0]
)

print(
    "Final Objective =",
    history[-1]
)
```

### Controller Parameters

- Learning Rate: \(\eta = 0.05\)
- Perturbation Size: \(\epsilon = 0.5\)
- Iterations: 50
- Smoothing Parameter: \(\sigma = 1.5\)

### GBAC Update Rule

\[
a_i(t+1)
=
a_i(t)
+
\eta \nabla_{a_i}\nu_{RM}(a,t)
\]

where:

- \(a_i(t)\) = current sensor position
- \(\eta\) = learning rate
- \(\nabla_{a_i}\nu_{RM}(a,t)\) = gradient of the robust objective function

## Output

<img width="648" height="528" alt="image" src="https://github.com/user-attachments/assets/18ddec44-58cb-4635-a968-b6c5e5f5305d" />

### Figure 11
**Adaptive RMABC Sensor Placement** showing the final sensor locations after GBAC optimization on the robust multi-class intensity map.


<img width="708" height="470" alt="image" src="https://github.com/user-attachments/assets/0c859a67-9b82-47b0-aefe-07bfc7aa8756" />

### Figure 12
**GBAC Convergence Curve** illustrating the evolution of the objective function during adaptive sensor optimization.

### Results

- Initial Objective Value
- Final Objective Value

An increase in the objective value indicates that GBAC successfully improves sensor deployment by relocating sensors toward more informative surveillance regions.

