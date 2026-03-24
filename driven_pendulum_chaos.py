"""
Study of the chaotic and non-chaotic regimes of a driven damped pendulum.

The equations of motion are integrated using the Euler-Cromer method.
The script allows the analysis of:
1. Short-time dynamics and phase space
2. Long-time Poincaré section
3. Lyapunov exponent estimate
"""

from math import sin, pi
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PHYSICAL CONSTANTS AND DEFAULT PARAMETERS
# =============================================================================

l = 9.8                  # Pendulum length
g = 9.8                  # Gravitational acceleration
q = 0.5                  # Damping constant
drive_freq = 2 / 3       # Driving force frequency

omega0 = 0.0             # Initial angular velocity
t0 = 0.0                 # Initial time
dt = 0.01                # Time step

theta0_list = [0.20]     # First initial angle list [rad]
theta1_list = [0.21]     # Second initial angle list [rad]

f_chaotic = 1.2          # Driving force for chaotic regime
f_non_chaotic = 0.5      # Driving force for non-chaotic regime


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def least_squares(x, y):
    """
    Linear least-squares fit:
        y = a0 + a1 * x
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    n = len(x)
    xm = np.mean(x)
    ym = np.mean(y)
    sx = np.sum(x)
    sy = np.sum(y)
    sxy = np.sum(x * y)
    sx2 = np.sum(x**2)

    a1 = (n * sxy - sx * sy) / (n * sx2 - sx**2)
    a0 = ym - a1 * xm
    y_fit = a0 + a1 * x

    print(f"Estimated Lyapunov exponent: {a1:.4f}")
    return y_fit, a0, a1


# =============================================================================
# SIMULATION OF THE PENDULUM
# =============================================================================

def simulate_pendulum_pair(theta0, theta1, driving_force, case, tf):
    """
    Simulate two nearby pendulum trajectories using the Euler-Cromer method
    """
    n_steps = int(tf / dt)

    theta = np.zeros(n_steps, dtype=float)
    omega = np.zeros(n_steps, dtype=float)

    theta2 = np.zeros(n_steps, dtype=float)
    omega2 = np.zeros(n_steps, dtype=float)

    theta_p1 = np.zeros(n_steps, dtype=float)
    omega_p1 = np.zeros(n_steps, dtype=float)

    theta_p2 = np.zeros(n_steps, dtype=float)
    omega_p2 = np.zeros(n_steps, dtype=float)

    log_diff = np.zeros(n_steps, dtype=float)
    t = np.zeros(n_steps, dtype=float)

    theta[0] = theta0
    theta2[0] = theta1
    omega[0] = omega0
    omega2[0] = omega0
    t[0] = t0

    eps = 1e-12
    log_diff[0] = np.log(max(abs(theta2[0] - theta[0]), eps))

    # Driving period
    drive_period = 2 * pi / drive_freq

    # Next sampling time for the Poincaré section
    next_poincare_time = drive_period

    for i in range(n_steps - 1):
        
        # Euler-Cromer update for the first trajectory
        omega[i + 1] = (omega[i] - (g / l) * sin(theta[i]) * dt
            + driving_force * dt * sin(drive_freq * t[i])
            - q * omega[i] * dt)
        
        theta[i + 1] = theta[i] + omega[i + 1] * dt

        # Euler-Cromer update for the second trajectory
        omega2[i + 1] = (omega2[i] - (g / l) * sin(theta2[i]) * dt
            + driving_force * dt * sin(drive_freq * t[i])
            - q * omega2[i] * dt)
        
        theta2[i + 1] = theta2[i] + omega2[i + 1] * dt

        # Time update
        t[i + 1] = t[i] + dt

        # Logarithm of the angular separation
        diff = abs(theta2[i + 1] - theta[i + 1])
        log_diff[i + 1] = np.log(max(diff, eps))

        # Wrap angles only for cases 1 and 2
        if case == 1 or case == 2:
            if theta[i + 1] > pi:
                theta[i + 1] -= 2 * pi
            elif theta[i + 1] < - pi:
                theta[i + 1] += 2 * pi
        
            if theta2[i + 1] > pi:
                theta2[i + 1] -= 2 * pi
            elif theta2[i + 1] < - pi:
                theta2[i + 1] += 2 * pi


        # Poincaré section: sample once every driving period
        if t[i] < next_poincare_time <= t[i + 1]:
            theta_p1[i + 1] = theta[i + 1]
            omega_p1[i + 1] = omega[i + 1]
            theta_p2[i + 1] = theta2[i + 1]
            omega_p2[i + 1] = omega2[i + 1]
            next_poincare_time += drive_period

    return {
        "t": t,
        "theta": theta,
        "omega": omega,
        "theta2": theta2,
        "omega2": omega2,
        "theta_p1": theta_p1,
        "omega_p1": omega_p1,
        "theta_p2": theta_p2,
        "omega_p2": omega_p2,
        "log_diff": log_diff,
    }


def run_for_multiple_initial_angles(theta0_values, theta1_values, driving_force, case, tf):
    """
    Simulate one pair of trajectories for each pair of initial angles.
    """
    if len(theta0_values) != len(theta1_values):
        raise ValueError("Initial angle lists must have the same length.")

    results = []

    for theta0, theta1 in zip(theta0_values, theta1_values):
        data = simulate_pendulum_pair(
            theta0=theta0,
            theta1=theta1,
            driving_force=driving_force,
            case=case,
            tf=tf)
        results.append((theta0, theta1, data))

    return results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_solution(results, title_text, fig_num):
    """
    Plot angular position as a function of time.
    """
    plt.figure(fig_num)

    for theta0, theta1, data in results:
        plt.plot(data["t"], data["theta"], label=f"{theta0} rad")
        plt.plot(data["t"], data["theta2"], label=f"{theta1} rad")

    plt.xlabel("t (s)")
    plt.ylabel("theta (rad)")
    plt.title(title_text)
    plt.legend(loc=1)


def plot_phase_space(results, title_text, fig_num):
    """
    Plot phase space trajectories.
    """
    plt.figure(fig_num)

    for theta0, theta1, data in results:
        plt.plot(data["theta"], data["omega"], label=f"{theta0} rad")
        plt.plot(data["theta2"], data["omega2"], label=f"{theta1} rad")

    plt.xlabel("theta (rad)")
    plt.ylabel("omega")
    plt.title(title_text)
    plt.legend(loc=1)


def plot_poincare(results, title_text, fig_num):
    """
    Plot Poincaré section.
    """
    plt.figure(fig_num)

    for theta0, theta1, data in results:
        plt.plot(data["theta_p1"], data["omega_p1"], ".", label=f"{theta0} rad")
        plt.plot(data["theta_p2"], data["omega_p2"], ".", label=f"{theta1} rad")

    plt.xlabel("theta Poincaré")
    plt.ylabel("omega Poincaré")
    plt.title(title_text)
    plt.legend(loc=1)


def plot_lyapunov(results, title_text, fig_num):
    """
    Plot log separation and linear fit used to estimate the Lyapunov exponent.
    """
    plt.figure(fig_num)

    for theta0, theta1, data in results:
        fit, _, _ = least_squares(data["t"], data["log_diff"])
        plt.plot(data["t"], data["log_diff"], label=f"{theta0} rad and {theta1} rad")
        plt.plot(data["t"], fit, "-.", label="Linear fit")

    plt.xlabel("t (s)")
    plt.ylabel("log|Δtheta|")
    plt.title(title_text)
    plt.legend(loc=1)


# =============================================================================
# PREDEFINED REGIMES
# =============================================================================

def non_chaotic_regime(case, tf):
    """
    Run and plot the non-chaotic regime.
    """
    print("\nNON-CHAOTIC REGIME")
    print(f"Parameters: F = {f_non_chaotic}, q = {q}, drive_freq = {drive_freq}")

    results = run_for_multiple_initial_angles(
        theta0_values=theta0_list,
        theta1_values=theta1_list,
        driving_force=f_non_chaotic,
        case=case,
        tf=tf)

    if case == 1:
        plot_solution(results, "Driven damped pendulum (non-chaotic)", fig_num=1)
        plot_phase_space(results, "Phase space - non-chaotic regime", fig_num=2)

    elif case == 2:
        plot_poincare(results, "Poincaré section - non-chaotic regime", fig_num=1)

    elif case == 3:
        plot_lyapunov(results, "Lyapunov exponent (non-chaotic regime)", fig_num=1)


def chaotic_regime(case, tf):
    """
    Run and plot the chaotic regime.
    """
    print("\nCHAOTIC REGIME")
    print(f"Parameters: F = {f_chaotic}, q = {q}, drive_freq = {drive_freq}")

    results = run_for_multiple_initial_angles(
        theta0_values=theta0_list,
        theta1_values=theta1_list,
        driving_force=f_chaotic,
        case=case,
        tf=tf)

    if case == 1:
        plot_solution(results, "Driven damped pendulum (chaotic)", fig_num=3)
        plot_phase_space(results, "Phase space - chaotic regime", fig_num=4)

    elif case == 2:
        plot_poincare(results, "Poincaré section - chaotic regime", fig_num=2)

    elif case == 3:
        plot_lyapunov(results, "Lyapunov exponent (chaotic regime)", fig_num=2)

# =============================================================================
# MAIN
# =============================================================================

def main(mode="default"):
    cases = {
        1: "Short-time dynamics and phase space",
        2: "Long-time Poincaré section",
        3: "Lyapunov exponent estimate"
    }

    if mode == "interactive":
        print("Available cases:")
        print("1: Short-time dynamics and phase space")
        print("2: Long-time Poincaré section")
        print("3: Lyapunov exponent estimate")

        case = int(input("Enter the case number (1, 2 or 3): "))

        if case == 1:
            print("\nRunning case 1: Short-time dynamics\n")
            tf = 100
        elif case == 2:
            print("\nRunning case 2: Long-time Poincaré section\n")
            tf = 2500
        elif case == 3:
            print("\nRunning case 3: Lyapunov exponent estimate\n")
            tf = 100
        else:
            raise ValueError("Invalid case selection.")

    else:
        case = 2
        description = cases[case]
        print(f"\nRunning case {case}: {description}\n")
        print("Use interactive mode to explore other cases.\n")

        if case == 1:
            tf = 100
        elif case == 2:
            tf = 2500
        elif case == 3:
            tf = 100

    non_chaotic_regime(case, tf)
    chaotic_regime(case, tf)

    plt.show()


if __name__ == "__main__":
    main()