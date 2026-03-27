import numpy as np
from collections import deque
import time
import sys

class Synapse:
    """
    Ek connection — memory yahan rehti hai
    Weight change hoti hai experience se
    """
    def __init__(self):
        # Initial weights boosted for better punch!
        self.weight   = np.random.uniform(0.5, 1.0)
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
        self.threshold     = 0.05     # Extreme Sensitivity
        self.bias          = 0.01     # Nengo style bias — dimaag thoda active rahe
        self.rest          = -0.1     # resting voltage
        self.decay         = 0.95     # voltage decay per step
        self.refractory    = 0        # recovery time after spike
        self.last_spike    = -np.inf  # kab last fire kiya
        self.fatigue       = 0.0      # thakan factor (adaptation)
        self.spike_history = deque(maxlen=100)
    
    def receive(self, current: float, t: float) -> bool:
        """
        Input current receive karo
        Spike karo ya nahi decide karo
        """
        if self.refractory > 0:
            self.refractory -= 1
            return False  # abhi recover ho raha hai
        
        # Fatigue decay (Dheere dheere thakan khatam hoti hai)
        self.fatigue *= 0.8 # Fast recovery

        # Voltage update (leaky integrate)
        # Bias add kiya jaise Nengo mein hota hai
        self.voltage = (
            self.voltage * self.decay 
            + (current + self.bias)
            + self.rest * (1 - self.decay)
        )
        
        # Effective threshold
        effective_threshold = self.threshold + (self.fatigue * 2.0)

        # Threshold cross hua?
        if self.voltage >= effective_threshold:
            self.voltage    = self.rest   # reset
            self.refractory = 2           # rest
            self.fatigue   += 1.0         # Balanced Fatigue
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
        
        # Lateral Inhibition (Muqabala)
        # Reduced slightly to allow neurons to stay alive while competing
        self.inhibition_strength = 0.5

        # Success Lock
        self.frozen = False

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
            # Input scaling: 3.0 multiplier for stronger stimulus
            fired = neuron.receive(
                (inputs[i] * 3.0) if i < len(inputs) else 0.0,
                self.t
            )
            current_spikes.append(fired)
        spikes_per_layer.append(current_spikes)
        
        # Baaki layers propagate karo
        for li in range(1, len(self.layers)):
            prev_spikes  = spikes_per_layer[li - 1]
            layer_spikes = []
            
            # current layer neurons
            current_layer = self.layers[li]

            for post in current_layer:
                total_input = 0.0
                
                for pi, pre in enumerate(self.layers[li - 1]):
                    if prev_spikes[pi]:  # pre ne fire kiya
                        syn = self.synapses.get((pre.id, post.id))
                        if syn:
                            total_input      += syn.weight
                            syn.last_pre      = self.t
                            if not self.frozen:
                                # Eligibility trace update
                                syn.eligibility   = (
                                    syn.eligibility * 0.9 + 0.1
                                )
                
                fired = post.receive(total_input, self.t)
                layer_spikes.append(fired)
                if fired:
                    self.total_spikes += 1
                    # Lateral Inhibition: Peer neurons (same layer) ko dabbao
                    for peer in current_layer:
                        if peer.id != post.id:
                            peer.voltage -= self.inhibition_strength
            
            spikes_per_layer.append(layer_spikes)
        
        # STDP learning apply karo
        if not self.frozen:
            self._stdp_update(spikes_per_layer)
        
        # Dopamine decay
        self.dopamine *= self.dopamine_decay
        
        # Homeostasis (neurons ko balance rakhna)
        # Frequent checks for faster scaling
        if self.learning_steps % 100 == 0:
            self._homeostasis()
        
        self.learning_steps += 1
        
        # Output layer spikes return karo
        return np.array(spikes_per_layer[-1], dtype=float)
    
    # ── STDP: Spike Timing Learning ─────────────────────────────────
    
    def _stdp_update(self, spikes_per_layer):
        """
        Hebbian learning (Integrated from stdplearn.py)
        Jo neurons saath fire hote hain, woh wire ho jaate hain!
        """
        if self.frozen: return # Success Lock: Permanent Memory!

        A_plus  = 0.1     # Balanced
        A_minus = 0.2     # Strong Pruning
        tau     = 0.02    # Standard window
        
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
                    
                    dw = 0.0

                    # 1. Post fired: check pre history (Strengthen)
                    if post_fired and pre.last_spike > -np.inf:
                        dt = self.t - pre.last_spike
                        if 0 <= dt < 0.1:
                            dw += A_plus * np.exp(-dt / tau)

                    # 2. Pre fired: check post history (Weaken)
                    if pre_fired and post.last_spike > -np.inf:
                        dt = self.t - post.last_spike
                        if 0 < dt < 0.1: # dt > 0 strictly to avoid double-counting same-step
                            dw -= A_minus * np.exp(-dt / tau)

                    if dw != 0:
                        # Dopamine modulate karta hai learning
                        dw *= (1 + self.dopamine * 2)
                        # Weights positive rehne chahiye (Excitatory neurons)
                        syn.weight = np.clip(syn.weight + dw, 0.01, 2.0)
    
    # ── Dopamine: Reward Signal ──────────────────────────────────────
    
    def reward(self, amount: float = 1.0):
        """
        "Yeh accha tha" — weights strengthen karo
        Jaise brain mein dopamine release hota hai
        """
        if self.frozen: return # permanent memory locked!

        self.dopamine = min(self.dopamine + amount, 10.0)
        
        # Eligibility traces pe dopamine apply karo
        # (jo recently active tha woh strengthen hoga)
        for syn in self.synapses.values():
            if syn.eligibility > 0.1:
                syn.weight = np.clip(
                    syn.weight + 0.2 * syn.eligibility * amount,
                    0.01, 2.0
                )
                syn.eligibility *= 0.5  # use ho gaya
    
    def punish(self, amount: float = 1.0):
        """
        "Yeh galat tha" — jo hua woh weaken karo
        Negative dopamine (adrenaline jaise)
        """
        if self.frozen: return # permanent memory locked!

        self.dopamine = max(self.dopamine - amount, -20.0)
        
        for syn in self.synapses.values():
            if syn.eligibility > 0.1:
                syn.weight = np.clip(
                    syn.weight - 0.2 * syn.eligibility * amount,
                    0.01, 2.0
                )
                syn.eligibility *= 0.5
    
    # ── Homeostasis: Self-Regulation ────────────────────────────────
    
    def _homeostasis(self):
        """
        Homeostatic Scaling — competitive balance
        Neuron 0 bahut fire karega → threshold upar (Tired)
        Neuron 1 fire nahi karega → threshold neeche (Eager)
        """
        if self.frozen: return # Success Lock: Permanent Memory!

        target_rate = 0.05  # Lower target for sparse coding
        
        for layer in self.layers:
            for neuron in layer:
                recent = [
                    s for s in neuron.spike_history
                    if self.t - s < 0.5
                ]
                actual_rate = len(recent) / 0.5

                if actual_rate > target_rate:
                    neuron.threshold += 0.01  # Slow scaling
                elif actual_rate < target_rate * 0.1:
                    neuron.threshold -= 0.005 # Slow scaling

                # Dynamic range for extreme sensitivity
                neuron.threshold = np.clip(neuron.threshold, 0.01, 1.0)
    
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
        
        if consolidated_count > 0:
            print(f"Consolidated {consolidated_count} connections")
        return consolidated_count

    def normalize_weights(self, target_avg: float = 1.0):
        """
        Weight Normalization — weights ko stable rakho
        """
        weights = [s.weight for s in self.synapses.values()]
        if not weights: return

        current_avg = np.mean(weights)
        if current_avg == 0: return

        factor = target_avg / current_avg
        for syn in self.synapses.values():
            syn.weight = np.clip(syn.weight * factor, 0.01, 2.0)

    # ── Inference Mode ───────────────────────────────────────────────

    def run_inference(self, pattern_A, pattern_B, episodes=100):
        """
        Inference Mode — bina lalach (dopamine) ke test karo
        """
        print("\n--- Starting AI Brain Inference (Test Mode) ---")
        self.frozen = True  # lock weights
        self.dopamine = 0.0 # reset dopamine

        correct_A = 0
        correct_B = 0

        for episode in range(episodes):
            # Test Pattern A
            out_A = np.zeros(len(self.layers[-1]))
            for _ in range(100):
                out_A += self.step(pattern_A * 20.0)

            if out_A.sum() > 0 and np.argmax(out_A) == 0:
                correct_A += 1

            # Rest
            for _ in range(10): self.step(np.zeros(4))

            # Test Pattern B
            out_B = np.zeros(len(self.layers[-1]))
            for _ in range(100):
                out_B += self.step(pattern_B * 20.0)

            if out_B.sum() > 0 and np.argmax(out_B) == 1:
                correct_B += 1

            if (episode + 1) % 20 == 0:
                print(f"Episode {episode+1}/{episodes} | Accuracy A: {correct_A/(episode+1):.2%} | B: {correct_B/(episode+1):.2%}")

        print("\n--- Inference Results ---")
        print(f"Final Accuracy - Pattern A: {correct_A/episodes:.2%}, Pattern B: {correct_B/episodes:.2%}")
        if (correct_A + correct_B) / (2 * episodes) > 0.8:
            print("Dimaag ne bina lalach ke zabardast perform kiya! Phase 1 Successful. ✅")
        else:
            print("Dimaag thoda confuse hai, shayad aur training ki zaroorat hai. ⚠️")

    # ── Stats ────────────────────────────────────────────────────────
    
    def show_weights(self):
        """
        Weight Visualizer — weights ko map mein save karo
        """
        with open("weights_map.txt", "w") as f:
            f.write(f"Brain Time: {self.t:.2f}s\n")
            f.write("-" * 30 + "\n")
            for (pre, post), syn in sorted(self.synapses.items()):
                # Visual ASCII bar
                bar = "#" * int(syn.weight * 10)
                f.write(f"S[{pre}->{post}]: {syn.weight:.3f} | {bar}\n")
        print("Weights map saved to weights_map.txt")

    def reset_fatigue(self):
        """
        Emergency Fatigue Reset
        """
        for layer in self.layers:
            for neuron in layer:
                neuron.fatigue = 0.0

    def status(self):
        weights = [s.weight for s in self.synapses.values()]
        return (
            f"T:{self.t:.2f}s | "
            f"Spikes:{self.total_spikes} | "
            f"W_avg:{np.mean(weights):.3f} | "
            f"DA:{self.dopamine:.2f} | "
            f"Mem:{len(self.consolidated)}"
        )
if __name__ == "__main__":
    # 1. Brain setup: 4 inputs -> 8 hidden -> 2 outputs
    my_brain = RealtimeBrain(layer_sizes=[4, 8, 2])


    

    # 2. Training data (Patterns)
    pattern_A = np.array([1.0, 0.0, 1.0, 0.0]) # Maan lo ye "Right" signal hai




    pattern_B = np.array([0.0, 1.0, 0.0, 1.0]) # Maan lo ye "Left" signal hai




    if "--test" in sys.argv:
        my_brain.run_inference(pattern_A, pattern_B)
    else:
        print("\n--- Starting AI Brain Training ---")

        # History for Balanced Success Lock
        history_A = deque(maxlen=20)
        history_B = deque(maxlen=20)

        for episode in range(500):
            if (episode + 1) % 50 == 0:
                print(f"\nEpisode {episode + 1}")

            success_A = 0
            success_B = 0

            # Targeted Recovery multipliers & Threshold assistance
            reward_mult_A = 1.0
            reward_mult_B = 1.0
            if len(history_A) == 20:
                if acc_A < 0.3: # Critical recovery
                    reward_mult_A = 3.0
                    my_brain.layers[-1][0].threshold = max(0.01, my_brain.layers[-1][0].threshold - 0.02)
                    my_brain.reset_fatigue()
                if acc_B < 0.3: # Critical recovery
                    reward_mult_B = 3.0
                    my_brain.layers[-1][1].threshold = max(0.01, my_brain.layers[-1][1].threshold - 0.02)
                    my_brain.reset_fatigue()

            # Inhibition Reset logic
            if not my_brain.frozen and len(history_A) == 20 and (acc_A < 0.1 or acc_B < 0.1):
                 current_inhibition = 0.1 # Temporarily reduce to let neurons breathe
            else:
                 current_inhibition = 0.5
            my_brain.inhibition_strength = current_inhibition

            # 1. Pattern A dikhao
            out_A = np.zeros(len(my_brain.layers[-1]))
            for _ in range(100):
                step_out = my_brain.step(pattern_A * 20.0)
                out_A += step_out

            if out_A.sum() > 0:
                action = np.argmax(out_A)
                if action == 0:
                    my_brain.reward(2.0 * reward_mult_A)
                    success_A = 1
                    if (episode + 1) % 50 == 0: print(f"Pattern A -> Neuron {action}: Sahi! (Dopamine)")
                else:
                    my_brain.punish(2.0)
                    if (episode + 1) % 50 == 0: print(f"Pattern A -> Neuron {action}: Galat!")
            elif (episode + 1) % 50 == 0:
                print("Pattern A: No Spikes")

            # Small rest period
            for _ in range(10): my_brain.step(np.zeros(4))

            # 2. Pattern B dikhao
            out_B = np.zeros(len(my_brain.layers[-1]))
            for _ in range(100):
                step_out = my_brain.step(pattern_B * 20.0)
                out_B += step_out

            if out_B.sum() > 0:
                action = np.argmax(out_B)
                if action == 1:
                    my_brain.reward(5.0 * reward_mult_B) # Bonus + Multiplier
                    success_B = 1
                    if (episode + 1) % 50 == 0: print(f"Pattern B -> Neuron {action}: Sahi! (Bonus Dopamine)")
                else:
                    my_brain.punish(2.0)
                    if (episode + 1) % 50 == 0: print(f"Pattern B -> Neuron {action}: Galat!")
            elif (episode + 1) % 50 == 0:
                print("Pattern B: No Spikes")

            # Balanced Success Lock checking
            history_A.append(success_A)
            history_B.append(success_B)

            acc_A = sum(history_A) / len(history_A) if len(history_A) > 0 else 0
            acc_B = sum(history_B) / len(history_B) if len(history_B) > 0 else 0

            if acc_A > 0.9 and acc_B > 0.9 and len(history_A) == 20 and not my_brain.frozen:
                my_brain.frozen = True
                print(f"!!! Balanced Success Lock at Episode {episode+1}: Accuracy > 90%! !!!")
                my_brain.show_weights() # Map save karo

            # Weight Normalization & Memory Decay
            if not my_brain.frozen:
                for syn in my_brain.synapses.values():
                    syn.weight *= 0.9999 # Very slow decay to hold onto Sahi! connections
                my_brain.normalize_weights(target_avg=1.0)

            if (episode + 1) % 50 == 0:
                print(f"Stats: {my_brain.status()} | Acc_A: {acc_A:.0%} | Acc_B: {acc_B:.0%}")

