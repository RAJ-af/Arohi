import nengo
import numpy as np

# Yeh brain ki actual learning rule hai
# "Jo saath fire karte hain, saath wire ho jaate hain"

class STDPLearning:
    """
    Spike Timing Dependent Plasticity
    Agar A → B sequence mein fire ho
    toh A→B connection strong hoga
    """
    def __init__(self):
        self.lr_plus = 0.01   # strengthen
        self.lr_minus = 0.005  # weaken
        self.tau = 0.02        # time window
    
    def update(self, pre_spike_time, post_spike_time, weight):
        dt = post_spike_time - pre_spike_time
        
        if dt > 0:  # pre pehle fire kiya → strengthen
            dw = self.lr_plus * np.exp(-dt / self.tau)
        else:       # post pehle fire kiya → weaken
            dw = -self.lr_minus * np.exp(dt / self.tau)
        
        return np.clip(weight + dw, 0, 1)

# Use karo
stdp = STDPLearning()
weight = 0.5

# Simulate learning
for step in range(100):
    pre_t  = np.random.uniform(0, 0.1)
    post_t = pre_t + np.random.uniform(-0.05, 0.05)
    weight = stdp.update(pre_t, post_t, weight)

print(f"Learned weight: {weight:.4f}")
```

---

## Mobile Pe Chalane Ka Tarika
```
Python Code
    │
    ▼
Compile with Cython / Nuitka
    │
    ▼
Pure C code ban jaata hai
    │
    ├──► Android (JNI se)
    ├──► iOS (C library)
    └──► Web (WASM)

Ya seedha:
Python → TensorFlow Lite → Mobile
