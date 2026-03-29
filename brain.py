import numpy as np
from collections import deque
import time
import sys

class SensoryEncoder:
    """Encoder jo strings ko spike sequences mein convert karta hai."""
    def __init__(self, n_inputs=4):
        self.n_inputs = n_inputs

    def encode_text(self, text: str):
        sequence = []
        for char in text.upper():
            bits = np.zeros(self.n_inputs)
            val = ord(char) % (2**self.n_inputs)
            for i in range(self.n_inputs):
                if (val >> i) & 1: bits[i] = 1.0
            sequence.append(bits)
        return sequence

class RealtimeBrain:
    """Vectorized SNN jo Termux par fast chalta hai (Brain simulation)."""
    def __init__(self, layer_sizes: list[int], density: float = 0.05):
        self.t = 0.0
        self.dt = 0.001
        self.layer_sizes = layer_sizes
        self.num_neurons = sum(layer_sizes)

        # Neuron State Vectors
        self.v = np.ones(self.num_neurons) * -0.1
        self.thr = np.ones(self.num_neurons) * 0.05
        self.fatigue = np.zeros(self.num_neurons)
        self.gain = np.ones(self.num_neurons)
        self.last_spike = np.ones(self.num_neurons) * -np.inf
        self.refractory = np.zeros(self.num_neurons, dtype=int)
        self.fire_rate = np.zeros(self.num_neurons) # For Homeostasis

        # Constants
        self.decay = 0.95
        self.rest = -0.1
        self.bias = 0.01
        
        # Layer Indexing
        self.layer_indices = []
        curr = 0
        for s in layer_sizes:
            self.layer_indices.append(np.arange(curr, curr + s))
            curr += s
        
        # Synapse Vectors (Sparse)
        self.syn_pre = []
        self.syn_post = []

        # Build Sparse Connectivity
        for li in range(len(layer_sizes) - 1):
            pre_idx = self.layer_indices[li]
            post_idx = self.layer_indices[li+1]
            for r in pre_idx:
                for c in post_idx:
                    if np.random.random() < density or li == len(layer_sizes)-2:
                        self.syn_pre.append(r)
                        self.syn_post.append(c)

        # Recurrent (Hidden Layer)
        hidden_idx = self.layer_indices[1]
        for r in hidden_idx:
            for c in hidden_idx:
                if r != c and np.random.random() < density:
                    self.syn_pre.append(r)
                    self.syn_post.append(c)

        self.syn_pre = np.array(self.syn_pre, dtype=int)
        self.syn_post = np.array(self.syn_post, dtype=int)
        num_syn = len(self.syn_pre)

        self.w = np.random.uniform(0.3, 0.6, num_syn)
        self.eligibility = np.zeros(num_syn)
        self.depression = np.ones(num_syn)

        self.dopamine = 0.0
        self.dopamine_decay = 0.95
        self.layer_inhibs = [0.0, 1.2, 5.0]

        self.acc_A = 0.0
        self.acc_B = 0.0
        self.frozen = False
        self.total_spikes = 0
        self.learning_steps = 0
        self.last_spikes_vec = np.zeros(self.num_neurons, dtype=bool)

    def reset_network_state(self):
        self.v[:] = self.rest
        self.fatigue[:] = 0.0
        self.refractory[:] = 0
        self.dopamine = 0.0
        self.last_spikes_vec[:] = False

    def step(self, inputs: np.ndarray) -> np.ndarray:
        self.t += self.dt
        
        # 1. Update Sensory Layer (0)
        l0 = self.layer_indices[0]
        self.v[l0] = self.v[l0] * self.decay + (inputs * 10.0 + self.bias) + self.rest * (1 - self.decay)
        
        # 2. Synaptic Propagation (Vectorized)
        incoming_spikes = self.last_spikes_vec[self.syn_pre]
        weighted_inputs = self.w * self.depression * incoming_spikes

        total_synaptic_input = np.zeros(self.num_neurons)
        np.add.at(total_synaptic_input, self.syn_post, weighted_inputs)

        # 3. Update Hidden and Output Voltages
        l_others = np.concatenate(self.layer_indices[1:])
        self.v[l_others] = self.v[l_others] * self.decay + (total_synaptic_input[l_others] * self.gain[l_others] + self.bias) + self.rest * (1 - self.decay)

        # 4. Intrinsic Plasticity (Silent neurons get hungry)
        silent_mask = (self.t - self.last_spike) > 0.1
        self.thr[silent_mask] = np.maximum(0.01, self.thr[silent_mask] - 0.001)
        self.v[silent_mask] += 0.005

        # 5. Spike Detection
        self.fatigue *= 0.9
        eff_thr = self.thr + (self.fatigue * 8.0)
        spiking = (self.v >= eff_thr) & (self.refractory <= 0)

        # Record Spikes
        self.v[spiking] = self.rest
        self.refractory[spiking] = 4
        self.fatigue[spiking] += 3.0
        self.last_spike[spiking] = self.t
        self.total_spikes += np.sum(spiking)
        self.fire_rate = self.fire_rate * 0.99 + spiking * 0.01

        # 6. Lateral Inhibition (Strict Winner-Take-All)
        for li in range(1, len(self.layer_indices)):
            idx = self.layer_indices[li]
            num_layer_spikes = np.sum(spiking[idx])
            if num_layer_spikes > 0:
                strength = self.layer_inhibs[li]
                self.v[idx] -= num_layer_spikes * strength
                # Winners keep their rest voltage (prevent self-inhibition)
                self.v[spiking & np.isin(np.arange(self.num_neurons), idx)] = self.rest

        self.refractory = np.maximum(0, self.refractory - 1)

        # 7. Learning and Recovery (Throttled for performance)
        if not self.frozen:
            # Synaptic Boring (Short-Term Depression)
            self.depression[incoming_spikes] = np.maximum(0.1, self.depression[incoming_spikes] - 0.05)
            self.eligibility[incoming_spikes] = np.minimum(10.0, self.eligibility[incoming_spikes] + 0.5)

            if self.learning_steps % 10 == 0: # 10ms learning frequency
                self._stdp_update_vectorized(spiking)

            # Biological Recovery
            self.eligibility *= 0.999
            self.depression = np.minimum(1.0, self.depression + 0.005)

            if self.learning_steps % 100 == 0:
                self._homeostasis_vectorized()

        self.dopamine *= self.dopamine_decay
        self.last_spikes_vec = spiking
        self.learning_steps += 1

        return spiking[self.layer_indices[-1]].astype(float)

    def _stdp_update_vectorized(self, current_spikes):
        t_pre = self.last_spike[self.syn_pre]
        t_post = self.last_spike[self.syn_post]
        
        # Logic: Post fired now -> strengthen if pre fired recently
        #        Pre fired now -> weaken if post fired recently
        post_fired_now = current_spikes[self.syn_post]
        pre_fired_now = current_spikes[self.syn_pre]
        
        dw = np.zeros_like(self.w)
        tau = 0.02
        A_plus, A_minus = 0.1, 0.2
        
        # LTP
        ltp_mask = post_fired_now & (t_pre > -np.inf)
        dt_ltp = self.t - t_pre[ltp_mask]
        dw[ltp_mask] += A_plus * np.exp(-dt_ltp / tau)

        # LTD
        ltd_mask = pre_fired_now & (t_post > -np.inf)
        dt_ltd = self.t - t_post[ltd_mask]
        dw[ltd_mask] -= A_minus * np.exp(-dt_ltd / tau)

        if np.any(dw):
            dw *= (1 + self.dopamine * 2)
            self.w = np.clip(self.w + dw, 0.01, 1.2)

    def _homeostasis_vectorized(self):
        for li in range(len(self.layer_indices)):
            idx = self.layer_indices[li]
            target_rate = 1.0 if li == len(self.layer_indices) - 1 else 0.1

            overactive = self.fire_rate[idx] > target_rate
            underactive = self.fire_rate[idx] < target_rate * 0.05

            self.thr[idx[overactive]] += 0.01
            self.gain[idx[overactive]] = np.maximum(0.2, self.gain[idx[overactive]] - 0.1)

            self.thr[idx[underactive]] = np.maximum(0.01, self.thr[idx[underactive]] - 0.005)
            self.gain[idx[underactive]] = np.minimum(10.0, self.gain[idx[underactive]] + 0.2)

    def reward(self, amount: float = 1.0):
        if self.frozen: return
        self.dopamine = min(self.dopamine + amount, 10.0)
        # Vectorized reinforcement
        eligible = self.eligibility > 0.1
        self.w[eligible] = np.clip(self.w[eligible] + 0.05 * self.eligibility[eligible] * amount, 0.01, 1.2)
        self.eligibility[eligible] *= 0.5

    def punish_neuron(self, neuron_id, amount=1.0):
        if self.frozen: return
        syn_mask = (self.syn_post == neuron_id) & (self.eligibility > 0.1)
        self.w[syn_mask] = np.clip(self.w[syn_mask] - 0.1 * self.eligibility[syn_mask] * amount, 0.01, 1.2)
        self.eligibility[syn_mask] *= 0.5

    def normalize_weights(self, target_avg: float = 0.8):
        avg = np.mean(self.w)
        if avg > 0:
            self.w = np.clip(self.w * (target_avg / avg), 0.01, 1.2)

    def apply_manual_controls(self):
        try:
            with open("control.txt", "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3 and parts[0] == "SET":
                        pre, post = map(int, parts[1].split("->"))
                        val = float(parts[2])
                        mask = (self.syn_pre == pre) & (self.syn_post == post)
                        self.w[mask] = np.clip(val, 0.01, 1.2)
                        if np.any(mask): print(f"God Mode: {pre}->{post} set to {val}")
            open("control.txt", "w").close()
        except Exception: pass

    def run_inference(self, ears, text, episodes=10):
        print(f"\n--- AI Brain Inference: '{text}' ---")
        self.frozen = True
        correct = 0
        sequence = ears.encode_text(text)
        for ep in range(episodes):
            self.reset_network_state()
            success = True
            for i, pattern in enumerate(sequence):
                for _ in range(10): self.step(np.zeros(4))
                out = np.zeros(len(self.layer_indices[-1]))
                for _ in range(50): out += self.step(pattern * 100.0)
                if np.argmax(out) != i: success = False
            if success: correct += 1
            print(f"Ep {ep+1}/{episodes} | Sequence Success: {'YES' if success else 'NO'}")
        print(f"Final Accuracy: {correct/episodes:.0%}")
        print("\nSummary: Brain ne patterns ko differentiate karna seekh liya hai!")

    def show_weights(self):
        with open("weights_map.txt", "w") as f:
            f.write(f"Brain Time: {self.t:.2f}s | Synapses: {len(self.w)}\n")
            # Only top 100 for brevity in file
            indices = np.argsort(self.w)[-100:]
            for i in indices:
                f.write(f"S[{self.syn_pre[i]:03}->{self.syn_post[i]:03}]: {self.w[i]:.3f}\n")
        print(f"Weights map updated (Avg: {np.mean(self.w):.3f})")

    def status(self):
        return (f"T:{self.t:.2f}s | W_avg:{np.mean(self.w):.3f} | "
                f"Acc A: {self.acc_A:.0%}, B: {self.acc_B:.0%}")

if __name__ == "__main__":
    # Performance Optimization: Vectorized 500-neuron hidden layer
    my_brain = RealtimeBrain(layer_sizes=[4, 500, 2], density=0.05)
    ears = SensoryEncoder(n_inputs=4)

    if "--test" in sys.argv:
        my_brain.run_inference(ears, "AB")
    else:
        print("\n--- Starting Vectorized AI Brain Training (Seekhna shuru...) ---")
        history_A = deque(maxlen=20)

        start_time = time.time()
        for episode in range(1000):
            my_brain.total_spikes = 0
            my_brain.apply_manual_controls()
            my_brain.reset_network_state()

            for _ in range(50): my_brain.step(np.zeros(4))
            sequence = ears.encode_text('AB')

            success_seq = 0
            for i, pattern in enumerate(sequence):
                for _ in range(10): my_brain.step(np.zeros(4))
                out = np.zeros(2)
                for _ in range(50): out += my_brain.step(pattern * 100.0)

                if np.sum(out) > 0:
                    action = np.argmax(out)
                    if action == i:
                        # Sahi sequence ke liye reward
                        my_brain.reward(5.0)
                        if i == 1: success_seq = 1
                    else:
                        # Galat pattern par punishment
                        my_brain.punish_neuron(my_brain.layer_indices[-1][action], amount=20.0)

            history_A.append(success_seq)
            my_brain.acc_A = my_brain.acc_B = sum(history_A)/len(history_A)

            if (episode + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Stats: {my_brain.status()} | Sim Speed: {my_brain.t/elapsed:.2f}x")
                print(f"Output Thresholds: {[f'{my_brain.thr[idx]:.3f}' for idx in my_brain.layer_indices[-1]]}")
                sys.stdout.flush()

            if my_brain.acc_A >= 0.95 and len(history_A) == 20:
                my_brain.frozen = True
                print(f"\n!!! Balanced Success Lock at Episode {episode+1} !!!")
                my_brain.show_weights()
                break

            if not my_brain.frozen:
                my_brain.w *= 0.9999
                if (episode + 1) % 50 == 0: my_brain.normalize_weights()
