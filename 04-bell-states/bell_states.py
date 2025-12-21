"""
Multiple-Qubit States - Concept 4: Bell States
===============================================
The four maximally entangled two-qubit states.
"""

import numpy as np

def main():
    print("=" * 60)
    print("BELL STATES - MAXIMALLY ENTANGLED STATES")
    print("=" * 60)
    
    # Define the four Bell states
    print("\n1. THE FOUR BELL STATES")
    print("-" * 40)
    
    # |Φ+⟩ = (|00⟩ + |11⟩)/√2
    phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
    print(f"|Φ+⟩ = (|00⟩ + |11⟩)/√2 = {phi_plus}")
    
    # |Φ-⟩ = (|00⟩ - |11⟩)/√2
    phi_minus = np.array([1, 0, 0, -1]) / np.sqrt(2)
    print(f"|Φ-⟩ = (|00⟩ - |11⟩)/√2 = {phi_minus}")
    
    # |Ψ+⟩ = (|01⟩ + |10⟩)/√2
    psi_plus = np.array([0, 1, 1, 0]) / np.sqrt(2)
    print(f"|Ψ+⟩ = (|01⟩ + |10⟩)/√2 = {psi_plus}")
    
    # |Ψ-⟩ = (|01⟩ - |10⟩)/√2
    psi_minus = np.array([0, 1, -1, 0]) / np.sqrt(2)
    print(f"|Ψ-⟩ = (|01⟩ - |10⟩)/√2 = {psi_minus}")
    
    # Orthonormality
    print("\n2. ORTHONORMALITY CHECK")
    print("-" * 40)
    bell_states = [phi_plus, phi_minus, psi_plus, psi_minus]
    names = ['|Φ+⟩', '|Φ-⟩', '|Ψ+⟩', '|Ψ-⟩']
    
    print("Inner products ⟨i|j⟩:")
    print(f"{'':8}", end='')
    for name in names:
        print(f"{name:8}", end='')
    print()
    
    for i, (state_i, name_i) in enumerate(zip(bell_states, names)):
        print(f"{name_i:8}", end='')
        for j, state_j in enumerate(bell_states):
            inner = np.vdot(state_i, state_j)
            print(f"{inner.real:8.4f}", end='')
        print()
    
    print("\nBell states form an orthonormal basis for 2-qubit space!")
    
    # Measurement correlations
    print("\n3. MEASUREMENT CORRELATIONS")
    print("-" * 40)
    
    def analyze_correlations(state, name):
        probs = np.abs(state)**2
        print(f"\n{name}:")
        print(f"  P(00) = {probs[0]:.4f}")
        print(f"  P(01) = {probs[1]:.4f}")
        print(f"  P(10) = {probs[2]:.4f}")
        print(f"  P(11) = {probs[3]:.4f}")
        
        same = probs[0] + probs[3]
        diff = probs[1] + probs[2]
        print(f"  Same outcomes: {same:.2%}")
        print(f"  Different outcomes: {diff:.2%}")
    
    for state, name in zip(bell_states, names):
        analyze_correlations(state, name)
    
    # Density matrices
    print("\n4. DENSITY MATRICES")
    print("-" * 40)
    
    def density_matrix(state):
        return np.outer(state, np.conj(state))
    
    rho_phi_plus = density_matrix(phi_plus)
    print("|Φ+⟩⟨Φ+| =")
    print(np.round(rho_phi_plus, 4))
    
    # Reduced density matrix (partial trace)
    print("\n5. REDUCED DENSITY MATRIX (Partial Trace)")
    print("-" * 40)
    
    def partial_trace_B(rho):
        """Trace out the second qubit."""
        rho_A = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    rho_A[i, j] += rho[2*i + k, 2*j + k]
        return rho_A
    
    rho_A = partial_trace_B(rho_phi_plus)
    print("ρ_A = Tr_B(|Φ+⟩⟨Φ+|) =")
    print(np.round(rho_A.real, 4))
    
    purity = np.trace(rho_A @ rho_A).real
    print(f"\nPurity Tr(ρ_A²) = {purity:.4f}")
    print("Purity = 0.5 indicates maximally mixed state!")
    print("This is the signature of maximal entanglement.")
    
    # Bell state creation circuit
    print("\n6. CREATING BELL STATES")
    print("-" * 40)
    print("Starting from computational basis states:")
    print()
    print("  |00⟩ → H⊗I → CNOT → |Φ+⟩")
    print("  |01⟩ → H⊗I → CNOT → |Ψ+⟩")
    print("  |10⟩ → H⊗I → CNOT → |Φ-⟩")
    print("  |11⟩ → H⊗I → CNOT → |Ψ-⟩")
    
    # Verify with matrices
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    I = np.eye(2)
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])
    
    ket_00 = np.array([1, 0, 0, 0])
    result = CNOT @ np.kron(H, I) @ ket_00
    print(f"\nVerification: CNOT(H⊗I)|00⟩ = {np.round(result, 4)}")
    print(f"Expected |Φ+⟩ = {phi_plus}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: Bell states are maximally entangled -")
    print("measuring one qubit instantly determines the other!")
    print("=" * 60)

if __name__ == "__main__":
    main()
