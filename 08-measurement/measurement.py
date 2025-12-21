"""
Multiple-Qubit States - Concept 8: Measurement
==============================================
Measurement in multi-qubit systems and state collapse.
"""

import numpy as np

def main():
    print("=" * 60)
    print("MEASUREMENT IN MULTI-QUBIT SYSTEMS")
    print("=" * 60)
    
    # Born rule
    print("\n1. BORN RULE FOR MULTI-QUBIT STATES")
    print("-" * 40)
    print("For state |ψ⟩ = Σ α_k |k⟩:")
    print("  P(k) = |α_k|² = probability of measuring |k⟩")
    print("  Σ P(k) = 1 (normalization)")
    
    # Example state
    print("\n2. EXAMPLE: MEASUREMENT PROBABILITIES")
    print("-" * 40)
    
    # State: (|00⟩ + 2|01⟩ + |10⟩)/√6
    state = np.array([1, 2, 1, 0]) / np.sqrt(6)
    probs = np.abs(state)**2
    
    print(f"State: |ψ⟩ = (|00⟩ + 2|01⟩ + |10⟩)/√6")
    print(f"Amplitudes: {np.round(state, 4)}")
    print()
    print("Measurement probabilities:")
    print(f"  P(00) = |1/√6|² = 1/6 = {probs[0]:.4f}")
    print(f"  P(01) = |2/√6|² = 4/6 = {probs[1]:.4f}")
    print(f"  P(10) = |1/√6|² = 1/6 = {probs[2]:.4f}")
    print(f"  P(11) = |0|² = 0 = {probs[3]:.4f}")
    print(f"  Total = {np.sum(probs):.4f}")
    
    # Measurement simulation
    print("\n3. MEASUREMENT SIMULATION")
    print("-" * 40)
    
    def simulate_measurement(state, n_shots=10000):
        """Simulate quantum measurement."""
        probs = np.abs(state)**2
        outcomes = np.random.choice(len(state), size=n_shots, p=probs)
        
        counts = {}
        labels = ['00', '01', '10', '11']
        for i, label in enumerate(labels):
            counts[label] = np.sum(outcomes == i)
        
        return counts
    
    counts = simulate_measurement(state)
    print(f"Simulating 10,000 measurements of |ψ⟩:")
    print()
    for outcome, count in counts.items():
        bar = '█' * int(count / 200)
        print(f"  |{outcome}⟩: {count:5d} ({count/100:.1f}%) {bar}")
    
    # Bell state measurement
    print("\n4. MEASURING BELL STATES")
    print("-" * 40)
    
    bell_phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
    bell_psi_plus = np.array([0, 1, 1, 0]) / np.sqrt(2)
    
    print("|Φ+⟩ = (|00⟩ + |11⟩)/√2:")
    counts = simulate_measurement(bell_phi_plus)
    for outcome, count in counts.items():
        print(f"  |{outcome}⟩: {count/100:.1f}%")
    
    print("\n|Ψ+⟩ = (|01⟩ + |10⟩)/√2:")
    counts = simulate_measurement(bell_psi_plus)
    for outcome, count in counts.items():
        print(f"  |{outcome}⟩: {count/100:.1f}%")
    
    # State collapse
    print("\n5. STATE COLLAPSE")
    print("-" * 40)
    print("After measurement, the state collapses to the measured outcome.")
    print()
    print("Example: Measuring |Φ+⟩ = (|00⟩ + |11⟩)/√2")
    print()
    print("  If outcome is |00⟩:")
    print("    Post-measurement state: |00⟩")
    print("    Second qubit is definitely |0⟩")
    print()
    print("  If outcome is |11⟩:")
    print("    Post-measurement state: |11⟩")
    print("    Second qubit is definitely |1⟩")
    
    # Partial measurement
    print("\n6. PARTIAL MEASUREMENT (First Qubit Only)")
    print("-" * 40)
    
    def partial_measurement_first_qubit(state):
        """
        Measure only the first qubit.
        Returns probabilities and post-measurement states.
        """
        # Probability of measuring |0⟩ on first qubit
        p0 = np.abs(state[0])**2 + np.abs(state[1])**2
        # Probability of measuring |1⟩ on first qubit
        p1 = np.abs(state[2])**2 + np.abs(state[3])**2
        
        # Post-measurement states (normalized)
        if p0 > 1e-10:
            state_0 = np.array([state[0], state[1], 0, 0]) / np.sqrt(p0)
        else:
            state_0 = None
            
        if p1 > 1e-10:
            state_1 = np.array([0, 0, state[2], state[3]]) / np.sqrt(p1)
        else:
            state_1 = None
        
        return p0, p1, state_0, state_1
    
    print("For |Φ+⟩ = (|00⟩ + |11⟩)/√2:")
    p0, p1, state_0, state_1 = partial_measurement_first_qubit(bell_phi_plus)
    print(f"  P(first qubit = 0) = {p0:.4f}")
    print(f"  P(first qubit = 1) = {p1:.4f}")
    print(f"  If 0: post-state = {np.round(state_0, 4) if state_0 is not None else 'N/A'}")
    print(f"  If 1: post-state = {np.round(state_1, 4) if state_1 is not None else 'N/A'}")
    
    print("\nFor |+⟩⊗|0⟩ = (|00⟩ + |10⟩)/√2:")
    plus_zero = np.array([1, 0, 1, 0]) / np.sqrt(2)
    p0, p1, state_0, state_1 = partial_measurement_first_qubit(plus_zero)
    print(f"  P(first qubit = 0) = {p0:.4f}")
    print(f"  P(first qubit = 1) = {p1:.4f}")
    print(f"  If 0: post-state = {np.round(state_0, 4) if state_0 is not None else 'N/A'}")
    print(f"  If 1: post-state = {np.round(state_1, 4) if state_1 is not None else 'N/A'}")
    
    # Correlation demonstration
    print("\n7. ENTANGLEMENT CORRELATIONS")
    print("-" * 40)
    
    def measure_correlations(state, n_shots=10000):
        """Measure and compute correlations."""
        probs = np.abs(state)**2
        outcomes = np.random.choice(4, size=n_shots, p=probs)
        
        # Extract individual qubit outcomes
        q1 = outcomes // 2  # First qubit
        q2 = outcomes % 2   # Second qubit
        
        # Correlation coefficient
        correlation = np.corrcoef(q1, q2)[0, 1]
        
        # Same/different outcomes
        same = np.mean(q1 == q2)
        
        return correlation, same
    
    print("Correlation analysis (10,000 shots):")
    print()
    
    states = [
        (bell_phi_plus, '|Φ+⟩'),
        (bell_psi_plus, '|Ψ+⟩'),
        (plus_zero, '|+⟩⊗|0⟩'),
        (np.array([1, 1, 1, 1])/2, '|+⟩⊗|+⟩')
    ]
    
    for state, name in states:
        corr, same = measure_correlations(state)
        print(f"  {name}: correlation = {corr:+.3f}, same outcomes = {same:.1%}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: Measurement collapses superposition and")
    print("reveals correlations in entangled states!")
    print("=" * 60)

if __name__ == "__main__":
    main()
