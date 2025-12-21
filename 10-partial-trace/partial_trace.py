"""
Multiple-Qubit States - Concept 10: Partial Trace
=================================================
Describing subsystems and detecting entanglement.
"""

import numpy as np

def main():
    print("=" * 60)
    print("PARTIAL TRACE AND SUBSYSTEMS")
    print("=" * 60)
    
    # Density matrices
    print("\n1. DENSITY MATRICES")
    print("-" * 40)
    print("For a pure state |ψ⟩, the density matrix is:")
    print("  ρ = |ψ⟩⟨ψ|")
    print()
    print("Properties:")
    print("  • ρ† = ρ (Hermitian)")
    print("  • Tr(ρ) = 1 (normalized)")
    print("  • ρ² = ρ for pure states")
    print("  • Tr(ρ²) ≤ 1, with equality iff pure")
    
    # Example density matrices
    print("\n2. EXAMPLE DENSITY MATRICES")
    print("-" * 40)
    
    def density_matrix(state):
        """Create density matrix from state vector."""
        return np.outer(state, np.conj(state))
    
    # |00⟩
    ket_00 = np.array([1, 0, 0, 0])
    rho_00 = density_matrix(ket_00)
    print("|00⟩⟨00| =")
    print(rho_00)
    
    # Bell state |Φ+⟩
    phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
    rho_phi = density_matrix(phi_plus)
    print("\n|Φ+⟩⟨Φ+| =")
    print(np.round(rho_phi, 4))
    
    # Partial trace definition
    print("\n3. PARTIAL TRACE DEFINITION")
    print("-" * 40)
    print("For a bipartite system AB, the reduced density matrix of A is:")
    print("  ρ_A = Tr_B(ρ_AB)")
    print()
    print("This 'traces out' system B, leaving only information about A.")
    
    def partial_trace_B(rho_AB):
        """
        Compute partial trace over second qubit (system B).
        Input: 4x4 density matrix for 2-qubit system
        Output: 2x2 reduced density matrix for first qubit
        """
        rho_A = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    rho_A[i, j] += rho_AB[2*i + k, 2*j + k]
        return rho_A
    
    def partial_trace_A(rho_AB):
        """
        Compute partial trace over first qubit (system A).
        Input: 4x4 density matrix for 2-qubit system
        Output: 2x2 reduced density matrix for second qubit
        """
        rho_B = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    rho_B[i, j] += rho_AB[2*k + i, 2*k + j]
        return rho_B
    
    # Partial trace examples
    print("\n4. PARTIAL TRACE EXAMPLES")
    print("-" * 40)
    
    # Product state |00⟩
    print("For |00⟩:")
    rho_A = partial_trace_B(rho_00)
    print(f"  ρ_A = Tr_B(|00⟩⟨00|) =\n{rho_A}")
    print(f"  This is |0⟩⟨0| - a pure state!")
    
    # Bell state |Φ+⟩
    print("\nFor |Φ+⟩ = (|00⟩ + |11⟩)/√2:")
    rho_A = partial_trace_B(rho_phi)
    print(f"  ρ_A = Tr_B(|Φ+⟩⟨Φ+|) =\n{np.round(rho_A.real, 4)}")
    print(f"  This is I/2 - the maximally mixed state!")
    
    # Purity and entanglement
    print("\n5. PURITY AS ENTANGLEMENT INDICATOR")
    print("-" * 40)
    print("Purity: γ = Tr(ρ²)")
    print("  • γ = 1: pure state (no entanglement with environment)")
    print("  • γ < 1: mixed state (entangled with something)")
    print("  • γ = 1/d: maximally mixed (d = dimension)")
    
    def purity(rho):
        """Calculate purity Tr(ρ²)."""
        return np.trace(rho @ rho).real
    
    print("\nPurity of reduced states:")
    
    states = [
        (np.array([1, 0, 0, 0]), '|00⟩', 'Product'),
        (np.array([1, 0, 1, 0]) / np.sqrt(2), '|+⟩⊗|0⟩', 'Product'),
        (np.array([1, 0, 0, 1]) / np.sqrt(2), '|Φ+⟩', 'Maximally entangled'),
        (np.array([0, 1, 1, 0]) / np.sqrt(2), '|Ψ+⟩', 'Maximally entangled'),
        (np.array([np.sqrt(0.9), 0, 0, np.sqrt(0.1)]), 'Partial', 'Partially entangled')
    ]
    
    print(f"\n{'State':<20} {'Purity(ρ_A)':<15} {'Type'}")
    print("-" * 50)
    
    for state, name, state_type in states:
        rho = density_matrix(state)
        rho_A = partial_trace_B(rho)
        p = purity(rho_A)
        print(f"{name:<20} {p:<15.4f} {state_type}")
    
    # Von Neumann entropy
    print("\n6. VON NEUMANN ENTROPY")
    print("-" * 40)
    print("Entanglement entropy: S(ρ_A) = -Tr(ρ_A log₂ ρ_A)")
    print("  • S = 0: no entanglement")
    print("  • S = 1: maximally entangled (for 2 qubits)")
    
    def von_neumann_entropy(rho):
        """Calculate von Neumann entropy."""
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    
    print(f"\n{'State':<20} {'Entropy S(ρ_A)':<15}")
    print("-" * 35)
    
    for state, name, _ in states:
        rho = density_matrix(state)
        rho_A = partial_trace_B(rho)
        S = von_neumann_entropy(rho_A)
        print(f"{name:<20} {S:<15.4f}")
    
    # Mixed states
    print("\n7. MIXED STATES")
    print("-" * 40)
    print("A mixed state is a statistical ensemble of pure states:")
    print("  ρ = Σ p_i |ψ_i⟩⟨ψ_i|")
    print()
    
    # Example: 50-50 mixture of |0⟩ and |1⟩
    ket_0 = np.array([1, 0])
    ket_1 = np.array([0, 1])
    rho_mixed = 0.5 * np.outer(ket_0, ket_0) + 0.5 * np.outer(ket_1, ket_1)
    
    print("50-50 mixture of |0⟩ and |1⟩:")
    print(f"  ρ = 0.5|0⟩⟨0| + 0.5|1⟩⟨1| =\n{rho_mixed}")
    print(f"  Purity = {purity(rho_mixed):.4f}")
    print(f"  Entropy = {von_neumann_entropy(rho_mixed):.4f}")
    
    # Comparison with |+⟩
    ket_plus = np.array([1, 1]) / np.sqrt(2)
    rho_plus = np.outer(ket_plus, ket_plus)
    print("\nCompare with pure state |+⟩:")
    print(f"  ρ = |+⟩⟨+| =\n{np.round(rho_plus, 4)}")
    print(f"  Purity = {purity(rho_plus):.4f}")
    print(f"  Entropy = {von_neumann_entropy(rho_plus):.4f}")
    print("\nBoth give 50-50 measurement outcomes, but ρ_mixed has no coherence!")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: Partial trace reveals entanglement -")
    print("mixed reduced state ⟺ entanglement with traced-out system!")
    print("=" * 60)

if __name__ == "__main__":
    main()
