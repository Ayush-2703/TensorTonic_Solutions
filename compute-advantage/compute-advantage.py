import numpy as np

def compute_advantage(states: list, rewards: list, V: list, gamma: float) -> np.ndarray:
    """
    Returns the advantages as a NumPy array.
    """
    # Write code here
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    v_arr = np.asarray(V, dtype=np.float64)
    states_arr = np.asarray(states, dtype=np.intp)

    returns = np.zeros_like(rewards_arr, dtype=np.float64)
    g_next = 0.0

    for t in range(len(rewards_arr) - 1, -1, -1):
      g_t = rewards_arr[t] + gamma * g_next
      returns[t] = g_t
      g_next = g_t

    advantages = returns - v_arr[states_arr]
    return np.round(advantages, 4)
    pass