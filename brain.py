import numpy as np
from collections import deque
import time
import sys

class Synapse:
    """
    Ek connection — memory yahan rehti hai
    Weight change hoti hai experience se (STDP)
    """
    def __init__(self):
        # Initial weights boosted for better punch!
        self.weight   = np.random.uniform(0.5, 0.8)
        self.last_pre  = -np.inf   # kab pre neuron ne fire kiya
        self.last_post = -np.inf   # kab post neuron ne fire kiya
        
        # Eligibility trace — "kuch hua tha yaad hai"
        self.eligibility = 0.0


class Neuron:
    """
    Single spiking neuron — biological LIF model
    """
    def __init__(self, neuron_id):
        self.id            = neuron_id
        self.voltage       = 0.0      # membrane potential
        self.threshold     = 0.05     # Extreme Sensitivity
        self.bias          = 0.01     # Nengo style bias
        self.rest          = -0.1     # resting voltage
        self.decay         = 0.95     # voltage decay per step
        self.refractory    = 0        # recovery time after spike
        self.last_spike    = -np.inf  # kab last fire kiya
        self.fatigue       = 0.0      # thakan factor (adaptation)
        self.spike_history = deque(maxlen=100)
    
    def receive(self, current: float, t: float) -> bool:
        if self.refractory > 0:
            self.refractory -= 1
            return False
        
        # Fatigue decay
        self.fatigue *= 0.8

        # Voltage update
        self.voltage = (
            self.voltage * self.decay 
            + (current + self.bias)
            + self.rest * (1 - self.decay)
        )
        
        # Effective threshold
        effective_threshold = self.threshold + (self.fatigue * 2.0)

        if self.voltage >= effective_threshold:
            self.voltage    = self.rest   # reset
            self.refractory = 2           # rest
            self.fatigue   += 1.0         # increment fatigue
            self.last_spike = t
            self.spike_history.append(t)
            return True
        
        return False

    def reset(self):
        """Internal state reset karo for stability"""
        self.voltage = self.rest
        self.fatigue = 0.0
        self.refractory = 0


class RealtimeBrain:
    """
    Poora system — CPU pe, realtime learning ke saath
    """
    def __init__(self, layer_sizes: list[int]):
        self.t = 0.0
        self.dt = 0.001
        
        self.layers: list[list[Neuron]] = []
        neuron_id = 0
        for size in layer_sizes:
            layer = [Neuron(neuron_id + i) for i in range(size)]
            self.layers.append(layer)
            neuron_id += size
        
        self.synapses: dict[tuple, Synapse] = {}
        for li in range(len(self.layers) - 1):
            for pre in self.layers[li]:
                for post in self.layers[li + 1]:
                    self.synapses[(pre.id, post.id)] = Synapse()
        
        self.dopamine        = 0.0
        self.dopamine_decay  = 0.95
        self.inhibition_strength = 0.5
        self.frozen = False
        self.total_spikes   = 0
        self.learning_steps = 0
    
    def reset_network_state(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.reset()

    def step(self, inputs: np.ndarray) -> np.ndarray:
        self.t += self.dt
        spikes_per_layer = []
        current_spikes   = []
        
        for i, neuron in enumerate(self.layers[0]):
            fired = neuron.receive(
                (inputs[i] * 3.0) if i < len(inputs) else 0.0,
                self.t
            )
            current_spikes.append(fired)
        spikes_per_layer.append(current_spikes)
        
        for li in range(1, len(self.layers)):
            prev_spikes  = spikes_per_layer[li - 1]
            layer_neurons = self.layers[li]
            layer_spikes = [False] * len(layer_neurons)
            
            # Randomized order to prevent directional bias
            indices = list(range(len(layer_neurons)))
            np.random.shuffle(indices)

            for idx in indices:
                post = layer_neurons[idx]
                total_input = 0.0
                for pi, pre in enumerate(self.layers[li - 1]):
                    if prev_spikes[pi]:
                        syn = self.synapses.get((pre.id, post.id))
                        if syn:
                            total_input      += syn.weight
                            syn.last_pre      = self.t
                            if not self.frozen:
                                syn.eligibility = (syn.eligibility * 0.9 + 0.1)
                
                fired = post.receive(total_input, self.t)
                layer_spikes[idx] = fired
                if fired:
                    self.total_spikes += 1
                    # Lateral Inhibition
                    for peer in layer_neurons:
                        if peer.id != post.id:
                            peer.voltage -= self.inhibition_strength
            
            spikes_per_layer.append(layer_spikes)
        
        if not self.frozen:
            self._stdp_update(spikes_per_layer)
        
        self.dopamine *= self.dopamine_decay
        if not self.frozen and self.learning_steps % 100 == 0:
            self._homeostasis()
        
        self.learning_steps += 1
        return np.array(spikes_per_layer[-1], dtype=float)
    
    def _stdp_update(self, spikes_per_layer):
        if self.frozen: return
        A_plus, A_minus, tau = 0.1, 0.2, 0.02
        
        for li in range(len(self.layers) - 1):
            for pi, pre in enumerate(self.layers[li]):
                for post_idx, post in enumerate(self.layers[li + 1]):
                    syn = self.synapses.get((pre.id, post.id))
                    if not syn: continue
                    
                    pre_fired  = spikes_per_layer[li][pi]
                    post_fired = spikes_per_layer[li + 1][post_idx]
                    dw = 0.0

                    if post_fired and pre.last_spike > -np.inf:
                        dt = self.t - pre.last_spike
                        if 0 <= dt < 0.1: dw += A_plus * np.exp(-dt / tau)

                    if pre_fired and post.last_spike > -np.inf:
                        dt = self.t - post.last_spike
                        if 0 < dt < 0.1: dw -= A_minus * np.exp(-dt / tau)

                    if dw != 0:
                        if dw < 0 and self.dopamine < 0: dw *= 2.0
                        dw *= (1 + self.dopamine * 2)

                        old_weight = syn.weight
                        syn.weight = np.clip(old_weight + dw, 0.01, 1.2)
                        if syn.weight > old_weight:
                            for other_post in self.layers[li + 1]:
                                if other_post.id != post.id:
                                    other_syn = self.synapses.get((pre.id, other_post.id))
                                    if other_syn: other_syn.weight = np.clip(other_syn.weight - 0.01, 0.01, 1.2)
    
    def reward(self, amount: float = 1.0):
        if self.frozen: return
        self.dopamine = min(self.dopamine + amount, 10.0)
        for syn in self.synapses.values():
            if syn.eligibility > 0.1:
                syn.weight = np.clip(syn.weight + 0.2 * syn.eligibility * amount, 0.01, 1.2)
                syn.eligibility *= 0.5
    
    def punish(self, amount: float = 1.0):
        if self.frozen: return
        self.dopamine = max(self.dopamine - amount, -20.0)
        for syn in self.synapses.values():
            if syn.eligibility > 0.1:
                syn.weight = np.clip(syn.weight - 0.2 * syn.eligibility * amount, 0.01, 1.2)
                syn.eligibility *= 0.5

    def _homeostasis(self):
        if self.frozen: return
        target_rate = 0.05
        for layer in self.layers:
            for neuron in layer:
                recent = [s for s in neuron.spike_history if self.t - s < 0.5]
                actual_rate = len(recent) / 0.5
                if actual_rate > target_rate: neuron.threshold += 0.01
                elif actual_rate < target_rate * 0.1: neuron.threshold -= 0.005
                neuron.threshold = np.clip(neuron.threshold, 0.01, 1.0)
    
    def normalize_weights(self, target_avg: float = 1.0):
        weights = [s.weight for s in self.synapses.values()]
        if not weights: return
        current_avg = np.mean(weights)
        if current_avg == 0: return
        factor = target_avg / current_avg
        for syn in self.synapses.values():
            syn.weight = np.clip(syn.weight * factor, 0.01, 1.2)

    def apply_manual_controls(self):
        try:
            with open("control.txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 3 and parts[0] == "SET":
                        ids = parts[1].split("->")
                        pre, post = int(ids[0]), int(ids[1])
                        val = float(parts[2])
                        if (pre, post) in self.synapses:
                            self.synapses[(pre, post)].weight = np.clip(val, 0.01, 1.2)
                            print(f"Manual Control: Synapse {pre}->{post} set to {val}")
            open("control.txt", "w").close()
        except FileNotFoundError: pass
        except Exception as e: print(f"Manual Control Error: {e}")

    def run_inference(self, pattern_A, pattern_B, episodes=100):
        print("\n--- Starting AI Brain Inference (Test Mode) ---")
        self.frozen = True
        self.dopamine = 0.0
        correct_A = correct_B = 0
        for episode in range(episodes):
            self.reset_network_state()
            out_A = np.zeros(len(self.layers[-1]))
            for _ in range(100): out_A += self.step(pattern_A * 50.0)
            if out_A.sum() > 0 and np.argmax(out_A) == 0: correct_A += 1
            for _ in range(20): self.step(np.zeros(4))
            self.reset_network_state()
            out_B = np.zeros(len(self.layers[-1]))
            for _ in range(100): out_B += self.step(pattern_B * 50.0)
            if out_B.sum() > 0 and np.argmax(out_B) == 1: correct_B += 1
            if (episode + 1) % 20 == 0:
                print(f"Episode {episode+1}/{episodes} | Accuracy A: {correct_A/(episode+1):.2%} | B: {correct_B/(episode+1):.2%}")
        print(f"\nResults: A: {correct_A/episodes:.2%}, B: {correct_B/episodes:.2%}")

    def show_weights(self):
        with open("weights_map.txt", "w") as f:
            f.write(f"Brain Time: {self.t:.2f}s | Mode: {'Frozen' if self.frozen else 'Training'}\n")
            f.write("-" * 50 + "\n")
            for (pre, post), syn in sorted(self.synapses.items()):
                filled = int((syn.weight / 1.2) * 20)
                bar = "[" + "#" * filled + "-" * (20 - filled) + "]"
                f.write(f"S[{pre:02}->{post:02}]: {syn.weight:.3f} {bar}\n")
        print(f"Weights map updated (Avg: {np.mean([s.weight for s in self.synapses.values()]):.3f})")

    def status(self, acc_A, acc_B, is_assisted):
        weights = [s.weight for s in self.synapses.values()]
        return (f"T:{self.t:.2f}s | Spikes:{self.total_spikes} | W_avg:{np.mean(weights):.3f} | "
                f"Acc A: {acc_A:.0%}, B: {acc_B:.0%} | Assisted: {'YES' if is_assisted else 'NO'}")

if __name__ == "__main__":
    my_brain = RealtimeBrain(layer_sizes=[4, 16, 2])
    pattern_A = np.array([1.0, 0.0, 1.0, 0.0])
    pattern_B = np.array([0.0, 1.0, 0.0, 1.0])

    if "--test" in sys.argv:
        my_brain.run_inference(pattern_A, pattern_B)
    else:
        print("\n--- Starting AI Brain Training ---")
        history_A = deque(maxlen=20)
        history_B = deque(maxlen=20)
        history_assisted = deque(maxlen=20)
        acc_A = acc_B = 0.0

        for episode in range(1000):
            my_brain.total_spikes = 0
            my_brain.apply_manual_controls()

            is_assisted = False
            reward_mult_A = 1.0
            reward_mult_B = 1.0

            if not my_brain.frozen and len(history_A) == 20:
                if acc_A < 0.3 or acc_B < 0.3:
                    is_assisted = True
                    my_brain.inhibition_strength = 0.1
                    if acc_A < 0.3: reward_mult_A = 2.0
                    if acc_B < 0.3: reward_mult_B = 2.0
                else:
                    my_brain.inhibition_strength = 0.5

            # Trial A
            my_brain.reset_network_state()
            for _ in range(50): my_brain.step(np.zeros(4))
            out_A = np.zeros(len(my_brain.layers[-1]))
            for _ in range(100): out_A += my_brain.step(pattern_A * 50.0)
            success_A = 1 if (out_A.sum() > 0 and np.argmax(out_A) == 0) else 0
            if success_A: my_brain.reward(2.0 * reward_mult_A)
            elif out_A.sum() > 0: my_brain.punish(2.0)

            # Trial B
            my_brain.reset_network_state()
            for _ in range(50): my_brain.step(np.zeros(4))
            out_B = np.zeros(len(my_brain.layers[-1]))
            for _ in range(100): out_B += my_brain.step(pattern_B * 50.0)
            success_B = 1 if (out_B.sum() > 0 and np.argmax(out_B) == 1) else 0
            if success_B: my_brain.reward(4.0 * reward_mult_B)
            elif out_B.sum() > 0: my_brain.punish(2.0)

            history_A.append(success_A)
            history_B.append(success_B)
            history_assisted.append(is_assisted)
            acc_A, acc_B = sum(history_A)/len(history_A), sum(history_B)/len(history_B)

            # CRITICAL FIX: STRICT AND LOCK
            # Lock ONLY if BOTH reach 95% threshold AND no assistance was used in the window.
            if not my_brain.frozen and len(history_A) == 20:
                # STRICT INDEPENDENT AND CONDITION
                if (acc_A >= 0.95) and (acc_B >= 0.95):
                    # Stability Guard: Ensure no assistance was active in the entire 20-episode window
                    if not any(history_assisted):
                        my_brain.frozen = True
                        print(f"\n!!! Balanced Success Lock at Episode {episode+1} !!!")
                        print(f"Independent Accuracies Verified: [Acc A: {acc_A:.0%}] AND [Acc B: {acc_B:.0%}]")
                        my_brain.show_weights()

            if not my_brain.frozen:
                for syn in my_brain.synapses.values(): syn.weight *= 0.9999
                if (episode + 1) % 10 == 0: my_brain.normalize_weights(target_avg=0.8)

            if (episode + 1) % 5 == 0:
                my_brain.show_weights()
                if (episode + 1) % 50 == 0:
                    print(f"Stats: {my_brain.status(acc_A, acc_B, is_assisted)}")
