import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

# extra plotting functions
def plot_weight_evolution(W_history, neuron_index=(0, 0)):
    i, j = neuron_index  # (output neuron, reservoir neuron)
    values = [W[i, j] for W in W_history]
    plt.figure(figsize=(8, 3))
    plt.plot(values)
    plt.xlabel("Time (steps)")
    plt.ylabel(f"Weight [{i}, {j}]")
    plt.title(f"Evolution of Weight out_W[{i}, {j}]")
    plt.grid(True)
    plt.show()

def plot_spike_raster(spike_log, title="Spike Raster"):
    import matplotlib.pyplot as plt
    times, neurons = [], []
    for t, spikes in enumerate(spike_log):
        for i, val in enumerate(spikes):
            if val:
                times.append(t)
                neurons.append(i)
    plt.figure(figsize=(10, 4))
    plt.scatter(times, neurons, s=2)
    plt.xlabel('Time step')
    plt.ylabel('Neuron index')
    plt.title(title)
    plt.show()

def plot_firing_rate(spike_log, title="Average Firing Rate"):
    spike_array = np.array(spike_log)
    avg_rate = spike_array.mean(axis=1)  # mean over neurons, per timestep
    plt.figure(figsize=(8, 3))
    plt.plot(avg_rate)
    plt.xlabel("Time step")
    plt.ylabel("Mean firing rate")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_weight_heatmap(W, title="Output Weight Matrix"):
    plt.figure(figsize=(6, 4))
    plt.imshow(W, aspect='auto', cmap='viridis')
    plt.colorbar(label="Weight value")
    plt.xlabel("Reservoir Neuron Index")
    plt.ylabel("Output Neuron Index")
    plt.title(title)
    plt.show()

def plot_weight_histogram(W, title="Weight Distribution"):
    plt.figure(figsize=(6, 3))
    plt.hist(W.flatten(), bins=30, color='skyblue', edgecolor='k')
    plt.xlabel("Weight Value")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_input_spikes(spike_train, title="Input Spike Trains"):
    times, neurons = np.where(spike_train)
    plt.figure(figsize=(10, 3))
    plt.scatter(times, neurons, s=2)
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Input neuron")
    plt.grid(True)
    plt.show()

def create_pattern_labels(steps, pattern_A_label=0, pattern_B_label=1):
    labels = np.zeros(steps, dtype=int)
    labels[steps // 2:] = pattern_B_label
    return labels

def generate_switching_patterns(num_inputs=4, steps=200, seed=42):
    np.random.seed(seed)
    spikes = np.zeros((steps, num_inputs), dtype=bool)

    # Define temporal spike intervals for 2 groups
    pattern_A_intervals = [2, 4]  # for neurons 0, 1
    pattern_B_intervals = [30, 28]  # for neurons 2, 3

    # Phase offsets
    offsets_A = [0, 3]
    offsets_B = [2, 6]

    for i in range(num_inputs):
        if i < 2:  # Pattern A
            interval = pattern_A_intervals[i]
            offset = offsets_A[i]
            for t in range(0, steps // 2, interval):
                if t + offset < steps:
                    spikes[t + offset, i] = True
            # Switch to B at halfway
            interval = pattern_B_intervals[i]
            offset = offsets_B[i]
            for t in range(steps // 2, steps, interval):
                if t + offset < steps:
                    spikes[t + offset, i] = True
        else:  # Pattern B
            interval = pattern_B_intervals[i - 2]
            offset = offsets_B[i - 2]
            for t in range(0, steps // 2, interval):
                if t + offset < steps:
                    spikes[t + offset, i] = True
            # Switch to A
            interval = pattern_A_intervals[i - 2]
            offset = offsets_A[i - 2]
            for t in range(steps // 2, steps, interval):
                if t + offset < steps:
                    spikes[t + offset, i] = True

    return spikes

def animate_weight_matrix(W_history, interval=20):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(W_history[0], aspect='auto', cmap='viridis')
    fig.colorbar(im, ax=ax)
    ax.set_title("Animated Weight Evolution (out_W)")
    ax.set_xlabel("Reservoir Neuron Index")
    ax.set_ylabel("Output Neuron Index")

    def update(frame):
        im.set_array(W_history[frame])
        ax.set_title(f"Weight Matrix at t={frame}")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(W_history),
                                  interval=interval, blit=False, repeat=False)
    plt.show()

# network
class Reservoir():
    def __init__(self, res_size, inp_size, out_size, sparsity=0.2, spectral_radius=0.95):
        self.res_size = res_size
        self.out_size = out_size

        self.weights = np.random.uniform(-1, 1, (res_size, inp_size))
        W = np.random.randn(res_size, res_size)
        mask = np.random.rand(res_size, res_size) < sparsity
        W *= mask
        eigvals = np.linalg.eigvals(W)
        sr = max(abs(eigvals))
        self.W = (W / sr) * spectral_radius

        # Reservoir state
        self.state = np.zeros(res_size)
        self.out_pop = np.zeros(out_size)

        # Time constants
        self.taus = np.random.uniform(0.8, 1, res_size)
        self.taus_out = 0.8 #

        # Output weights (reservoir → feedforward)
        self.out_W = np.random.uniform(-1, 1, (out_size, res_size))

        # STDP traces
        self.pre_trace = np.zeros(res_size)
        self.post_trace = np.zeros(out_size)
        self.out_r = np.zeros(res_size)

        # STDP parameters
        self.tau_pre = 20 #
        self.tau_post = 20 #
        self.A_pre = 0.01 #
        self.A_post = -self.A_pre * 1.05 #
        self.w_max = 1.0 #

        # Monitor (optional)
        self.W_history = []

    def neural_update_eq(self, x):
        # Update reservoir
        self.state = self.taus * self.state + x @ self.weights.T + self.out_r @ self.W.T
        self.out_r = self.state > 1
        self.state[self.out_r] = 0

        # Update output
        self.out_pop = self.taus_out * self.out_pop + self.out_W @ self.out_r.astype(float)
        out_spikes = self.out_pop > 1
        self.out_pop[out_spikes] = 0

        # Update weights with STDP
        # self.STDP(out_r, out_spikes)

        return self.out_r, out_spikes

    # def STDP(self, out_r, out_spikes):
    #     """
    #     Online STDP learning rule, modifies self.out_W
    #     """
    #     # Decay traces
    #     self.pre_trace *= np.exp(-1 / self.tau_pre)
    #     self.post_trace *= np.exp(-1 / self.tau_post)

    #     # Update traces
    #     self.pre_trace[out_r] += self.A_pre
    #     self.post_trace[out_spikes] += self.A_post

    #     # STDP weight updates
    #     for i in range(self.res_size):
    #         if out_r[i]:
    #             self.out_W[:, i] += self.post_trace

    #     for j in range(self.out_size):
    #         if out_spikes[j]:
    #             self.out_W[j, :] += self.pre_trace

    #     # Clip weights
    #     self.out_W = np.clip(self.out_W, 0, self.w_max)

    #     # Optionally log
    #     self.W_history.append(self.out_W.copy())

# training
def train_with_teacher(reservoir, input_spikes, pattern_labels, steps=1000):
    # STDP parameters
        tau_pre, tau_post = 20, 20
        A_pre = 0.01
        A_post = -A_pre * 1.05
        w_max = 1.0

        N_res = reservoir.res_size
        N_out = reservoir.out_size

        pre_trace = np.zeros(N_res)
        post_trace = np.zeros(N_out)

        for t in range(steps):
            x_t = input_spikes[t].astype(float)
            res_spikes, _ = reservoir.neural_update_eq(x_t)

            # Determine teacher signal
            teacher_spikes = np.zeros(N_out, dtype=bool)
            label = pattern_labels[t]
            teacher_spikes[label] = True  # only 1 neuron active per pattern

            # STDP traces
            pre_trace *= np.exp(-1 / tau_pre)
            post_trace *= np.exp(-1 / tau_post)
            pre_trace[res_spikes] += A_pre
            post_trace[teacher_spikes] += A_post

            # STDP update
            for i in range(N_res):
                if res_spikes[i]:
                    reservoir.out_W[:, i] += post_trace
            for j in range(N_out):
                if teacher_spikes[j]:
                    reservoir.out_W[j, :] += pre_trace

            # Clip
            reservoir.out_W = np.clip(reservoir.out_W, 0, w_max)

            # if t % 100 == 0:
            reservoir.W_history.append(reservoir.out_W.copy())

# test
def run_test(reservoir, input_spikes, pattern_labels):
    steps = len(input_spikes)

    # Logs
    res_spike_log = []
    out_spike_log = []
    predicted_labels = []

    for t in range(steps):
        input_vector = input_spikes[t].astype(float)
        res_spikes, out_spikes = reservoir.neural_update_eq(input_vector)
        res_spike_log.append(res_spikes)
        out_spike_log.append(out_spikes)
        predicted_labels.append(np.argmax(out_spikes))  # optional: 0 or 1

    # Plot results
    plot_spike_raster(res_spike_log, title="Reservoir Spikes (Test)")
    plot_spike_raster(out_spike_log, title="Output Spikes (Test)")
    # plot_firing_rate(res_spike_log, title="Reservoir Firing Rate (Test)")
    # plot_firing_rate(out_spike_log, title="Output Layer Firing Rate (Test)")
    plot_weight_heatmap(reservoir.out_W, title="Final out_W Matrix (Test)")
    # plot_weight_histogram(reservoir.out_W, title="Final Weight Distribution (Test)")

    # Evaluate classification
    predicted_labels = np.array(predicted_labels)
    accuracy = np.mean(predicted_labels == pattern_labels)
    print(f"Test Accuracy (based on first spike winner-take-all): {accuracy * 100:.2f}%")

if __name__ == "__main__":
    steps = 200

    input_spikes = generate_switching_patterns(num_inputs=4, steps=steps)

    plot_input_spikes(input_spikes)
    pattern_labels = create_pattern_labels(steps)
    # print(pattern_labels)

    res = Reservoir(res_size=200, inp_size=4, out_size=2)

    # Training phase
    train_with_teacher(res, input_spikes, pattern_labels, steps=steps)

    # Testing phase (same input)
    run_test(res, input_spikes, pattern_labels)

    # Animate weights
    animate_weight_matrix(res.W_history)


# print(len(res.W_history))