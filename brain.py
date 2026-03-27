import numpy as np
from collections import deque
import time

class Synapse:
    """
    Ek connection — memory yahan rehti hai
    Weight change hoti hai experience se
    """
    def __init__(self):
        self.weight   = np.random.randn() * 0.1
        self.last_pre  = -np.inf   # kab pre neuron ne fire kiya
        self.last_post = -np.inf   # kab post neuron ne fire kiya
        
        # Eligibility trace — "kuch hua tha yaad hai"
        # Dopamine aane tak wait karta hai
        self.eligibility = 0.0


class Neuron:
    """
    Single spiking neuron — bilkul biological
    """
    def __init__(self, neuron_id):
        self.id            = neuron_id
        self.voltage       = 0.0      # membrane potential
        self.threshold     = 1.0      # fire kab karo
        self.rest          = -0.1     # resting voltage
        self.decay         = 0.95     # voltage decay per step
        self.refractory    = 0        # recovery time after spike
        self.last_spike    = -np.inf  # kab last fire kiya
        self.spike_history = deque(maxlen=100)
    
    def receive(self, current: float, t: float) -> bool:
        """
        Input current receive karo
        Spike karo ya nahi decide karo
        """
        if self.refractory > 0:
            self.refractory -= 1
            return False  # abhi recover ho raha hai
        
        # Voltage update (leaky integrate)
        self.voltage = (
            self.voltage * self.decay 
            + current 
            + self.rest * (1 - self.decay)
        )
        
        # Threshold cross hua?
        if self.voltage >= self.threshold:
            self.voltage    = self.rest   # reset
            self.refractory = 3           # 3 steps rest
            self.last_spike = t
            self.spike_history.append(t)
            return True  # SPIKE!
        
        return False


class RealtimeBrain:
    """
    Poora system — CPU pe, realtime learning ke saath
    
    Mechanisms:
    1. STDP    — spike timing se weights update
    2. Dopamine — reward/punishment signal
    3. Homeostasis — neurons bahut zyada ya kam fire na karein
    4. Consolidation — important cheezein stable ho jaati hain
    """
    
    def __init__(self, layer_sizes: list[int]):
        self.t = 0.0   # internal time
        self.dt = 0.001  # 1ms timestep (biological)
        
        # Neurons banao
        self.layers: list[list[Neuron]] = []
        neuron_id = 0
        for size in layer_sizes:
            layer = [Neuron(neuron_id + i) for i in range(size)]
            self.layers.append(layer)
            neuron_id += size
        
        # Synapses banao (har layer ko next se connect karo)
        self.synapses: dict[tuple, Synapse] = {}
        for li in range(len(self.layers) - 1):
            for pre in self.layers[li]:
                for post in self.layers[li + 1]:
                    self.synapses[(pre.id, post.id)] = Synapse()
        
        # Dopamine system
        self.dopamine        = 0.0   # current level
        self.dopamine_decay  = 0.95  # per step decay
        
        # Memory consolidation
        self.experience_log: list[dict] = []
        self.consolidated   = {}  # longterm memory
        
        # Stats
        self.total_spikes   = 0
        self.learning_steps = 0
    
    # ── Core: Forward Pass ──────────────────────────────────────────
    
    def step(self, inputs: np.ndarray) -> np.ndarray:
        """
        Ek timestep — inputs dalo, output lo
        Sab kuch realtime hota hai
        """
        self.t += self.dt
        
        # Layer 0 ko input dena
        spikes_per_layer = []
        current_spikes   = []
        
        for i, neuron in enumerate(self.layers[0]):
            fired = neuron.receive(
                inputs[i] if i < len(inputs) else 0.0, 
                self.t
            )
            current_spikes.append(fired)
        spikes_per_layer.append(current_spikes)
        
        # Baaki layers propagate karo
        for li in range(1, len(self.layers)):
            prev_spikes  = spikes_per_layer[li - 1]
            layer_spikes = []
            
            for post in self.layers[li]:
                total_input = 0.0
                
                for pi, pre in enumerate(self.layers[li - 1]):
                    if prev_spikes[pi]:  # pre ne fire kiya
                        syn = self.synapses.get((pre.id, post.id))
                        if syn:
                            total_input      += syn.weight
                            syn.last_pre      = self.t
                            # Eligibility trace update
                            syn.eligibility   = (
                                syn.eligibility * 0.9 + 0.1
                            )
                
                fired = post.receive(total_input, self.t)
                layer_spikes.append(fired)
                if fired:
                    self.total_spikes += 1
            
            spikes_per_layer.append(layer_spikes)
        
        # STDP learning apply karo
        self._stdp_update(spikes_per_layer)
        
        # Dopamine decay
        self.dopamine *= self.dopamine_decay
        
        # Homeostasis (neurons ko balance rakhna)
        if self.learning_steps % 1000 == 0:
            self._homeostasis()
        
        self.learning_steps += 1
        
        # Output layer spikes return karo
        return np.array(spikes_per_layer[-1], dtype=float)
    
    # ── STDP: Spike Timing Learning ─────────────────────────────────
    
    def _stdp_update(self, spikes_per_layer):
        """
        Hebbian learning — biological aur local
        Koi backpropagation nahi
        Koi GPU nahi
        """
        A_plus  = 0.005   # strengthen karo
        A_minus = 0.003   # weaken karo
        tau     = 0.02    # time window
        
        for li in range(len(self.layers) - 1):
            for pi, pre in enumerate(self.layers[li]):
                for post in self.layers[li + 1]:
                    syn = self.synapses.get((pre.id, post.id))
                    if not syn:
                        continue
                    
                    pre_fired  = spikes_per_layer[li][pi]
                    post_fired = spikes_per_layer[li + 1][
                        self.layers[li + 1].index(post)
                    ]
                    
                    if pre_fired and post.last_spike > -np.inf:
                        # Pre ne fire kiya — post ka history dekho
                        dt = self.t - post.last_spike
                        if abs(dt) < 0.1:  # 100ms window
                            if dt > 0:
                                # Post pehle fire kiya — weaken
                                dw = -A_minus * np.exp(-abs(dt)/tau)
                            else:
                                # Pre pehle fire kiya — strengthen
                                dw = A_plus * np.exp(-abs(dt)/tau)
                            
                            # Dopamine modulate karta hai learning
                            dw *= (1 + self.dopamine * 2)
                            
                            syn.weight = np.clip(
                                syn.weight + dw, -2.0, 2.0
                            )
    
    # ── Dopamine: Reward Signal ──────────────────────────────────────
    
    def reward(self, amount: float = 1.0):
        """
        "Yeh accha tha" — weights strengthen karo
        Jaise brain mein dopamine release hota hai
        """
        self.dopamine = min(self.dopamine + amount, 5.0)
        
        # Eligibility traces pe dopamine apply karo
        # (jo recently active tha woh strengthen hoga)
        for syn in self.synapses.values():
            if syn.eligibility > 0.1:
                syn.weight = np.clip(
                    syn.weight + 0.01 * syn.eligibility * amount,
                    -2.0, 2.0
                )
                syn.eligibility *= 0.5  # use ho gaya
    
    def punish(self, amount: float = 1.0):
        """
        "Yeh galat tha" — jo hua woh weaken karo
        Negative dopamine (adrenaline jaise)
        """
        self.dopamine = max(self.dopamine - amount, -2.0)
        
        for syn in self.synapses.values():
            if syn.eligibility > 0.1:
                syn.weight = np.clip(
                    syn.weight - 0.01 * syn.eligibility * amount,
                    -2.0, 2.0
                )
                syn.eligibility *= 0.5
    
    # ── Homeostasis: Self-Regulation ────────────────────────────────
    
    def _homeostasis(self):
        """
        Neurons khud ko regulate karte hain
        Bahut zyada fire → threshold badhao
        Bahut kam fire → threshold ghataao
        
        Brain mein yeh hamesha hota rehta hai
        """
        target_rate = 0.1  # 10% neurons active ideal
        
        for layer in self.layers:
            for neuron in layer:
                if len(neuron.spike_history) > 10:
                    recent = [
                        s for s in neuron.spike_history 
                        if self.t - s < 1.0
                    ]
                    actual_rate = len(recent) / 1.0
                    
                    if actual_rate > target_rate * 2:
                        neuron.threshold += 0.01  # kam fire karo
                    elif actual_rate < target_rate * 0.5:
                        neuron.threshold -= 0.005  # zyada fire karo
                    
                    neuron.threshold = np.clip(
                        neuron.threshold, 0.3, 3.0
                    )
    
    # ── Consolidation: Long-term Memory ─────────────────────────────
    
    def consolidate(self):
        """
        "Neend" wala mechanism — important cheezein save karo
        
        Jo synapses frequently active hain
        aur dopamine ke saath fire hue hain
        → woh stable ho jaate hain (longterm memory)
        """
        consolidated_count = 0
        
        for (pre_id, post_id), syn in self.synapses.items():
            # High weight + recently used = important
            importance = abs(syn.weight) * syn.eligibility
            
            if importance > 0.3:
                # Yeh connection important hai
                self.consolidated[(pre_id, post_id)] = syn.weight
                consolidated_count += 1
                
                # Consolidated connections decay nahi karte
                syn.eligibility = 0  # reset trace
        
        print(f"Consolidated {consolidated_count} connections")
    for episode in range(10):
        print(f"\nEpisode {episode + 1}")

        # --- YE WALA HISSA BADLO ---
        # Pattern A ko 50 baar dikhao (buildup ke liye)
        for _ in range(50):
            out_A = my_brain.step(pattern_A)
        
        # Ab check karo ki neuron fire hua ya nahi
        if out_A.sum() > 0:
            action = np.argmax(out_A)
            if action == 0:
                my_brain.reward(1.0)
                print(f"Pattern A -> Neuron {action}: Sahi! (Dopamine)")
            else:
                my_brain.punish(0.5)
                print(f"Pattern A -> Neuron {action}: Galat!")
        else:
            print("Pattern A: No Spikes (Dimaag shant hai)")
        # ---------------------------

        print(f"Stats: {my_brain.status()}")
        return consolidated_count
    
    # ── Stats ────────────────────────────────────────────────────────
    
    def status(self):
        weights = [s.weight for s in self.synapses.values()]
        return {
            "time":          f"{self.t:.3f}s",
            "total_spikes":  self.total_spikes,
            "avg_weight":    f"{np.mean(weights):.4f}",
            "dopamine":      f"{self.dopamine:.3f}",
            "longterm_mem":  len(self.consolidated),
            "synapses":      len(self.synapses),
        }
if __name__ == "__main__":
    # 1. Brain setup: 4 inputs -> 8 hidden -> 2 outputs
    my_brain = RealtimeBrain(layer_sizes=[4, 8, 2])


    

    # 2. Training data (Patterns)
    pattern_A = np.array([1.0, 0.0, 1.0, 0.0]) # Maan lo ye "Right" signal hai




    pattern_B = np.array([0.0, 1.0, 0.0, 1.0]) # Maan lo ye "Left" signal hai




    print("\n--- Starting AI Brain Training ---")
    
       for episode in range(10):
        print(f"\nEpisode {episode + 1}")

        # --- YE WALA HISSA BADLO ---
        # Pattern A ko 50 baar dikhao (buildup ke liye)
        for _ in range(50):
            out_A = my_brain.step(pattern_A)

        # Ab check karo ki neuron fire hua ya nahi
        if out_A.sum() > 0:
            action = np.argmax(out_A)
            if action == 0:
                my_brain.reward(1.0)
                print(f"Pattern A -> Neuron {action}: Sahi! (Dopamine)")
            else:
                my_brain.punish(0.5)
                print(f"Pattern A -> Neuron {action}: Galat!")
        else:
            print("Pattern A: No Spikes (Dimaag shant hai)")
        # ---------------------------

        print(f"Stats: {my_brain.status()}")

