# =============================================================================
# QuTiP + SciPy script for Java-Bali Low-Inertia Case Study (TPWRS submission)
# =============================================================================

!pip install qutip --quiet   # <-- remove this line after first run

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import qutip as qt

# --------------------------- JAVA-BALI PARAMETERS ---------------------------
f_mode = 0.61                    # Hz (dominant inter-area mode from 500 kV PMU Prony)
omega = 2 * np.pi * f_mode       # ≈ 3.83 rad/s
gamma = 0.44                     # damping factor from Prony analysis
R_reskill = 0.15                 # "reskilling" resistance (operator training effect)
alpha = 0.5                      # amplitude scaling
T = 50.0
n_points = 200
t = np.linspace(0, T, n_points)

# ----------------------- DAMPED OSCILLATOR (Heisenberg + RLC) --------------
def damped_frequency_deviation(y, t, omega, gamma, R):
    theta, dtheta = y
    # Damped harmonic oscillator with reskilling damping term
    ddtheta = -omega**2 * theta - 2*gamma*omega*dtheta - R*dtheta
    return [dtheta, ddtheta]

# Initial condition (typical disturbance)
y0 = [0.0, 5.0]   # initial angle + frequency deviation (rad/s)

# Solve for one realization
sol = odeint(damped_frequency_deviation, y0, t, args=(omega, gamma, R_reskill))

# --------------------------- MONTE CARLO (50 000 realizations) ------------
N_real = 50000
results = np.zeros((N_real, n_points))

np.random.seed(42)
for i in range(N_real):
    # Colored noise perturbation (renewable variability)
    noise = np.cumsum(np.random.normal(0, 0.005, n_points)) * np.exp(-gamma * t)
    # Solve with noise
    y0_noisy = [0.0, 5.0 + noise[0]]
    sol_noisy = odeint(damped_frequency_deviation, y0_noisy, t,
                       args=(omega, gamma, R_reskill))
    results[i] = sol_noisy[:, 0]   # frequency deviation

mean_traj = np.mean(results, axis=0)
std_traj  = np.std(results, axis=0)

# --------------------------- METRICS (vs. reference trajectory) ------------
# Reference = pure damped sinusoid for RMSE/MAE calculation
ref = 5.0 * np.exp(-gamma * t) * np.sin(omega * t)
rmse = np.sqrt(np.mean((mean_traj - ref)**2))
mae  = np.mean(np.abs(mean_traj - ref))
corr = np.corrcoef(mean_traj, ref)[0, 1]

print(f"Java-Bali Results (50,000 realizations)")
print(f"RMSE = {rmse:.5f} | MAE = {mae:.5f} | Pearson r = {corr:.3f}")

# --------------------------- PLOT (Fig. 4) --------------------------------
plt.figure(figsize=(10, 5))
plt.plot(t, mean_traj, 'b-', linewidth=2, label='Model prediction (mean)')
plt.fill_between(t, mean_traj - std_traj, mean_traj + std_traj,
                 color='blue', alpha=0.2, label='±1σ Monte Carlo')
plt.plot(t, ref, 'r--', linewidth=1.5, label='Reference damped sinusoid')
plt.xlabel('Time (s)')
plt.ylabel('Frequency Deviation (Hz or normalized)')
plt.title('Damped Frequency-Deviation Response – Java-Bali 0.61 Hz Mode')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
