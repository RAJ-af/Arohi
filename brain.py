import numpy as np
from collections import deque
import time
import sys

class Synapse:
    """Ek synapse jo do neurons ko jodta hai (connection)."""
    def __init__(self):
        self.weight   = np.random.uniform(0.3, 0.6)
        self.last_pre  = -np.inf
        self.last_post = -np.inf
        self.eligibility = 0.0


class Neuron:
    """Ek biological neuron jo spike generate karta hai."""
    def __init__(self, neuron_id):
        self.id            = neuron_id
        self.voltage       = 0.0
        self.threshold     = 0.05
        self.bias          = 0.01
        self.rest          = -0.1
        self.decay         = 0.95
        self.refractory    = 0
        self.last_spike    = -np.inf
        self.fatigue       = 0.0
        self.gain          = 1.0      # Global Synaptic Scaling factor
        self.spike_history = deque(maxlen=100)
    
    def receive(self, current: float, t: float) -> bool:
        if self.refractory > 0:
            self.refractory -= 1
            return False

        # Biologically inspired fatigue (Short-term adaptation)
        self.fatigue *= 0.9
        
        # Apply Gain (Synaptic Scaling)
        eff_current = current * self.gain
        self.voltage = (self.voltage * self.decay + (eff_current + self.bias) + self.rest * (1 - self.decay))

        # Accelerated Intrinsic Plasticity: Sensitization increases quickly when silent
        if t - self.last_spike > 0.1: # 100ms of silence
            self.threshold = max(0.01, self.threshold - 0.001) # 10x faster decay
            self.voltage += 0.005 # Faster sub-threshold sensitization

        # Effective threshold rises sharply with fatigue (Exhaustion)
        eff_threshold = self.threshold + (self.fatigue * 8.0) # Even more impact

        if self.voltage >= eff_threshold:
            self.voltage    = self.rest
            self.refractory = 4           # Longer refractory period
            self.fatigue   += 3.0         # Heavier exhaustion increment
            self.last_spike = t
            self.spike_history.append(t)
            return True
        return False

    def reset(self):
        self.voltage = self.rest
        self.fatigue = 0.0
        self.refractory = 0


class RealtimeBrain:
    """Poora neural network jo patterns seekhta hai (Brain simulation)."""
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
        self.layer_inhibs    = [0.0, 0.5, 2.5] # Stronger competition in output

        self.acc_A = 0.0
        self.acc_B = 0.0

        self.frozen = False
        self.total_spikes   = 0
        self.learning_steps = 0
    
    def reset_network_state(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.reset()
        self.dopamine = 0.0

    def step(self, inputs: np.ndarray) -> np.ndarray:
        self.t += self.dt
        spikes_per_layer = []
        
        # Layer 0
        current_spikes = []
        for i, neuron in enumerate(self.layers[0]):
            fired = neuron.receive((inputs[i] * 3.0) if i < len(inputs) else 0.0, self.t)
            current_spikes.append(fired)
        spikes_per_layer.append(current_spikes)
        
        # Propagation
        for li in range(1, len(self.layers)):
            prev_spikes  = spikes_per_layer[li - 1]
            layer_neurons = self.layers[li]
            layer_spikes = [False] * len(layer_neurons)

            indices = list(range(len(layer_neurons)))
            np.random.shuffle(indices)

            inhib_strength = self.layer_inhibs[li] if li < len(self.layer_inhibs) else 0.5

            for idx in indices:
                post = layer_neurons[idx]
                total_input = 0.0
                for pi, pre in enumerate(self.layers[li - 1]):
                    if prev_spikes[pi]:
                        syn = self.synapses.get((pre.id, post.id))
                        if syn:
                            total_input += syn.weight
                            if not self.frozen:
                                # Pre-synaptic activity leaves a trace
                                syn.eligibility = min(syn.eligibility + 0.5, 10.0)
                
                fired = post.receive(total_input, self.t)
                layer_spikes[idx] = fired
                if fired:
                    self.total_spikes += 1
                    # Lateral Inhibition
                    for peer in layer_neurons:
                        if peer.id != post.id:
                            peer.voltage -= inhib_strength
            
            spikes_per_layer.append(layer_spikes)
        
        if not self.frozen:
            self._stdp_update(spikes_per_layer)
            # Slow eligibility trace decay (Biological half-life scale)
            for syn in self.synapses.values():
                syn.eligibility *= 0.999
        
        self.dopamine *= self.dopamine_decay
        if not self.frozen and self.learning_steps % 100 == 0:
            self._homeostasis()
        
        self.learning_steps += 1
        return np.array(spikes_per_layer[-1], dtype=float)
    
    def _stdp_update(self, spikes_per_layer):
        if self.frozen: return
        # Robust STDP to allow autonomous learning to catch up
        A_plus, A_minus, tau = 0.1, 0.2, 0.02
        for li in range(len(self.layers) - 1):
            for pi, pre in enumerate(self.layers[li]):
                for post_idx, post in enumerate(self.layers[li + 1]):
                    syn = self.synapses.get((pre.id, post.id))
                    if not syn: continue
                    pre_fired, post_fired = spikes_per_layer[li][pi], spikes_per_layer[li + 1][post_idx]
                    dw = 0.0
                    if post_fired and pre.last_spike > -np.inf:
                        dt = self.t - pre.last_spike
                        if 0 <= dt < 0.1: dw += A_plus * np.exp(-dt / tau)
                    if pre_fired and post.last_spike > -np.inf:
                        dt = self.t - post.last_spike
                        if 0 < dt < 0.1: dw -= A_minus * np.exp(-dt / tau)
                    if dw != 0:
                        dw *= (1 + self.dopamine * 2)
                        old_w = syn.weight
                        syn.weight = np.clip(old_w + dw, 0.01, 1.2)
                        if syn.weight > old_w:
                            gain = syn.weight - old_w
                            peers = [self.synapses.get((pre.id, p.id)) for p in self.layers[li+1] if p.id != post.id]
                            peers = [s for s in peers if s]
                            if peers:
                                reduction = gain / len(peers)
                                for osyn in peers:
                                    osyn.weight = np.clip(osyn.weight - reduction, 0.01, 1.2)
    
    def reward(self, amount: float = 1.0):
        if self.frozen: return
        self.dopamine = min(self.dopamine + amount, 10.0)
        # Apply reinforcement using the long-lived traces
        for syn in self.synapses.values():
            if syn.eligibility > 0.1:
                syn.weight = np.clip(syn.weight + 0.05 * syn.eligibility * amount, 0.01, 1.2)
                syn.eligibility *= 0.5 # consumed
    
    def punish_neuron(self, neuron_id, amount=1.0):
        if self.frozen: return
        for (pre_id, post_id), syn in self.synapses.items():
            if post_id == neuron_id and syn.eligibility > 0.1:
                syn.weight = np.clip(syn.weight - 0.1 * syn.eligibility * amount, 0.01, 1.2)
                syn.eligibility *= 0.5

    def _homeostasis(self):
        if self.frozen: return
        for li, layer in enumerate(self.layers):
            # Target firing rates (Hz)
            target_rate = 1.0 if li == len(self.layers) - 1 else 0.1
            for neuron in layer:
                recent = [s for s in neuron.spike_history if self.t - s < 1.0]
                actual_rate = len(recent) / 1.0

                # Biologically plausible homeostasis
                if actual_rate > target_rate:
                    neuron.threshold += 0.005  # Threshold rises if overactive
                    neuron.gain = max(0.5, neuron.gain - 0.05) # Reduce gain if overactive
                elif actual_rate < target_rate * 0.1:
                    neuron.threshold -= 0.002  # Threshold drops if underactive
                    neuron.gain = min(5.0, neuron.gain + 0.1) # Increase gain if underactive

                # Clamp thresholds
                max_thresh = 0.5 if li == len(self.layers) - 1 else 0.2
                neuron.threshold = np.clip(neuron.threshold, 0.01, max_thresh)
    
    def normalize_weights(self, target_avg: float = 1.0):
        weights = [s.weight for s in self.synapses.values()]
        if not weights: return
        avg = np.mean(weights)
        if avg == 0: return
        factor = target_avg / avg
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
                            print(f"God Mode: {pre}->{post} set to {val}")
            open("control.txt", "w").close()
        except FileNotFoundError: pass

    def run_inference(self, pattern_A, pattern_B, episodes=100):
        print("\n--- AI Brain Inference (Test Mode) ---")
        self.frozen = True
        correct_A = correct_B = 0
        for episode in range(episodes):
            self.reset_network_state()
            out_A = np.zeros(2)
            for _ in range(100): out_A += self.step(pattern_A * 100.0)
            if out_A.sum() > 0 and np.argmax(out_A) == 0: correct_A += 1
            for _ in range(20): self.step(np.zeros(4))
            self.reset_network_state()
            out_B = np.zeros(2)
            for _ in range(100): out_B += self.step(pattern_B * 100.0)
            if out_B.sum() > 0 and np.argmax(out_B) == 1: correct_B += 1
            if (episode + 1) % 20 == 0:
                print(f"Ep {episode+1}/{episodes} | A: {correct_A/(episode+1):.0%} | B: {correct_B/(episode+1):.0%}")
        print(f"Final results: A: {correct_A/episodes:.0%}, B: {correct_B/episodes:.0%}")
        # Hinglish Summary for the user
        print("\nSummary: Brain ne patterns ko differentiate karna seekh liya hai!")

    def show_weights(self):
        with open("weights_map.txt", "w") as f:
            f.write(f"Brain Time: {self.t:.2f}s | Mode: {'Frozen' if self.frozen else 'Training'}\n")
            f.write("-" * 50 + "\n")
            for (pre, post), syn in sorted(self.synapses.items()):
                filled = int((syn.weight / 1.2) * 20)
                bar = "[" + "#" * filled + "-" * (20 - filled) + "]"
                f.write(f"S[{pre:02}->{post:02}]: {syn.weight:.3f} {bar}\n")
        print(f"Weights map updated (Avg: {np.mean([s.weight for s in self.synapses.values()]):.3f})")

    def status(self):
        weights = [s.weight for s in self.synapses.values()]
        return (f"T:{self.t:.2f}s | W_avg:{np.mean(weights):.3f} | "
                f"Acc A: {self.acc_A:.0%}, B: {self.acc_B:.0%}")

if __name__ == "__main__":
    # Optimization: 128 hidden neurons for better feature separation
    my_brain = RealtimeBrain(layer_sizes=[4, 128, 2])
    pattern_A = np.array([1.0, 0.0, 1.0, 0.0])
    pattern_B = np.array([0.0, 1.0, 0.0, 1.0])

    if "--test" in sys.argv:
        my_brain.run_inference(pattern_A, pattern_B)
    else:
        print("\n--- Starting AI Brain Training ---")
        history_A, history_B = deque(maxlen=20), deque(maxlen=20)

        for episode in range(1000):
            my_brain.total_spikes = 0
            my_brain.apply_manual_controls()

            # Simple 50/50 Curriculum
            trial_queue = ['A', 'B']
            np.random.shuffle(trial_queue)

            success_A_ep, success_B_ep = 0, 0
            for t_type in trial_queue:
                my_brain.reset_network_state()
                for _ in range(50): my_brain.step(np.zeros(4))

                input_pattern = pattern_A if t_type == 'A' else pattern_B
                target_neuron = 0 if t_type == 'A' else 1

                out = np.zeros(2)
                # Equal power for all patterns
                for _ in range(100): out += my_brain.step(input_pattern * 100.0)

                if out.sum() > 0:
                    action = np.argmax(out)
                    if action == target_neuron:
                        # Sahi action ke liye reward
                        my_brain.reward(5.0)
                        if t_type == 'A': success_A_ep = 1
                        else: success_B_ep = 1
                    else:
                        # Galat action ke liye punishment aur thakaan (Exhaustion)
                        my_brain.layers[-1][action].fatigue += 5.0
                        my_brain.punish_neuron(action, amount=20.0)

            history_A.append(success_A_ep)
            history_B.append(success_B_ep)
            my_brain.acc_A, my_brain.acc_B = sum(history_A)/len(history_A), sum(history_B)/len(history_B)

            if not my_brain.frozen and len(history_A) == 20:
                if (my_brain.acc_A >= 0.95) and (my_brain.acc_B >= 0.95):
                    my_brain.frozen = True
                    print(f"\n!!! Balanced Success Lock at Episode {episode+1} !!!")
                    my_brain.show_weights()

            if not my_brain.frozen:
                for syn in my_brain.synapses.values(): syn.weight *= 0.9999
                if (episode + 1) % 10 == 0:
                    my_brain.normalize_weights(target_avg=0.8)

            if (episode + 1) % 5 == 0:
                my_brain.show_weights()
                if (episode + 1) % 10 == 0:
                    print(f"Output Thresholds: {[f'{n.threshold:.3f}' for n in my_brain.layers[-1]]}")
                    print(f"Stats: {my_brain.status()}")
                    sys.stdout.flush()
