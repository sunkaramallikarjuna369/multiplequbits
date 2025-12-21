"""
Multiple-Qubit States - Concept 11: GHZ State
=============================================
Three-qubit entanglement and the Greenberger-Horne-Zeilinger state.
"""

import numpy as np

def main():
    print("=" * 60)
    print("GHZ STATE - THREE-QUBIT ENTANGLEMENT")
    print("=" * 60)
    
    # GHZ state definition
    print("\n1. GHZ STATE DEFINITION")
    print("-" * 40)
    print("The GHZ (Greenberger-Horne-Zeilinger) state:")
    print()
    print("  |GHZ⟩ = (|000⟩ + |111⟩)/√2")
    print()
    
    # Create GHZ state
    ghz = np.zeros(8)
    ghz[0] = 1 / np.sqrt(2)   # |000⟩
    ghz[7] = 1 / np.sqrt(2)   # |111⟩
    
    print(f"State vector: {ghz}")
    print()
    print("Basis state encoding:")
    for k in range(8):
        if ghz[k] != 0:
            binary = format(k, '03b')
            print(f"  |{binary}⟩ (k={k}): amplitude = {ghz[k]:.4f}")
    
    # GHZ creation circuit
    print("\n2. GHZ CREATION CIRCUIT")
    print("-" * 40)
    print("Starting from |000⟩:")
    print()
    print("|0⟩──H──●────●──")
    print("        │    │")
    print("|0⟩────⊕────│──")
    print("             │")
    print("|0⟩─────────⊕──")
    print()
    
    # Step by step creation
    I = np.eye(2)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])
    
    ket_0 = np.array([1, 0])
    ket_000 = np.kron(np.kron(ket_0, ket_0), ket_0)
    
    print("Step-by-step creation:")
    print(f"  |000⟩ = {ket_000}")
    
    # H on first qubit
    H_I_I = np.kron(np.kron(H, I), I)
    after_H = H_I_I @ ket_000
    print(f"  After H: {np.round(after_H, 4)}")
    print("         = (|000⟩ + |100⟩)/√2")
    
    # CNOT on qubits 1,2
    CNOT_12 = np.kron(CNOT, I)
    after_CNOT1 = CNOT_12 @ after_H
    print(f"  After CNOT₁₂: {np.round(after_CNOT1, 4)}")
    print("              = (|000⟩ + |110⟩)/√2")
    
    # CNOT on qubits 1,3 (need to construct this)
    def cnot_13():
        """CNOT with control on qubit 1, target on qubit 3."""
        gate = np.zeros((8, 8))
        for i in range(8):
            binary = format(i, '03b')
            if binary[0] == '1':
                # Flip third qubit
                new_binary = binary[0] + binary[1] + str(1 - int(binary[2]))
                j = int(new_binary, 2)
            else:
                j = i
            gate[j, i] = 1
        return gate
    
    CNOT_13 = cnot_13()
    ghz_created = CNOT_13 @ after_CNOT1
    print(f"  After CNOT₁₃: {np.round(ghz_created, 4)}")
    print("              = (|000⟩ + |111⟩)/√2 = |GHZ⟩")
    
    # Measurement properties
    print("\n3. MEASUREMENT PROPERTIES")
    print("-" * 40)
    
    probs = np.abs(ghz)**2
    print("Measurement probabilities:")
    for k in range(8):
        if probs[k] > 1e-10:
            binary = format(k, '03b')
            print(f"  P(|{binary}⟩) = {probs[k]:.4f} = 50%")
    
    print("\nKey property: All three qubits are ALWAYS the same!")
    print("  • If you measure |0⟩ on any qubit, all are |0⟩")
    print("  • If you measure |1⟩ on any qubit, all are |1⟩")
    
    # Simulation
    print("\n4. MEASUREMENT SIMULATION")
    print("-" * 40)
    
    n_shots = 10000
    outcomes = np.random.choice(8, size=n_shots, p=probs)
    
    print(f"Simulating {n_shots} measurements:")
    for k in range(8):
        count = np.sum(outcomes == k)
        if count > 0:
            binary = format(k, '03b')
            bar = '█' * int(count / 200)
            print(f"  |{binary}⟩: {count:5d} ({count/100:.1f}%) {bar}")
    
    # Entanglement analysis
    print("\n5. ENTANGLEMENT ANALYSIS")
    print("-" * 40)
    
    def density_matrix(state):
        return np.outer(state, np.conj(state))
    
    def partial_trace_23(rho):
        """Trace out qubits 2 and 3, keep qubit 1."""
        rho_1 = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(4):  # Sum over qubits 2,3
                    rho_1[i, j] += rho[4*i + k, 4*j + k]
        return rho_1
    
    def purity(rho):
        return np.trace(rho @ rho).real
    
    def von_neumann_entropy(rho):
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    
    rho_ghz = density_matrix(ghz)
    rho_1 = partial_trace_23(rho_ghz)
    
    print("Reduced density matrix of qubit 1:")
    print(np.round(rho_1.real, 4))
    print(f"\nPurity: {purity(rho_1):.4f}")
    print(f"Entropy: {von_neumann_entropy(rho_1):.4f} bits")
    print("\nρ₁ = I/2 (maximally mixed) → maximal entanglement!")
    
    # GHZ vs W state
    print("\n6. GHZ vs W STATE")
    print("-" * 40)
    
    # W state: (|001⟩ + |010⟩ + |100⟩)/√3
    w_state = np.zeros(8)
    w_state[1] = 1 / np.sqrt(3)  # |001⟩
    w_state[2] = 1 / np.sqrt(3)  # |010⟩
    w_state[4] = 1 / np.sqrt(3)  # |100⟩
    
    print("|W⟩ = (|001⟩ + |010⟩ + |100⟩)/√3")
    print(f"State vector: {np.round(w_state, 4)}")
    
    print("\nComparison:")
    print(f"{'Property':<30} {'GHZ':<15} {'W'}")
    print("-" * 55)
    print(f"{'Form':<30} {'|000⟩+|111⟩':<15} {'|001⟩+|010⟩+|100⟩'}")
    print(f"{'# of 1s in each term':<30} {'0 or 3':<15} {'exactly 1'}")
    
    # Robustness to qubit loss
    print("\nRobustness to qubit loss:")
    print("  GHZ: If one qubit is lost, remaining 2 are in mixed state")
    print("       (|00⟩⟨00| + |11⟩⟨11|)/2 - NO entanglement!")
    print("  W:   If one qubit is lost, remaining 2 are STILL entangled")
    print("       (|01⟩ + |10⟩)/√2 - Bell state!")
    
    # Applications
    print("\n7. APPLICATIONS OF GHZ STATES")
    print("-" * 40)
    print("• Quantum secret sharing")
    print("• Quantum error correction")
    print("• Tests of quantum nonlocality (GHZ paradox)")
    print("• Quantum metrology and sensing")
    print("• Multiparty quantum communication")
    
    # n-qubit GHZ
    print("\n8. n-QUBIT GHZ STATES")
    print("-" * 40)
    print("General n-qubit GHZ state:")
    print(f"  |GHZ_n⟩ = (|{'0'*5}...⟩ + |{'1'*5}...⟩)/√2")
    print()
    
    for n in range(2, 7):
        dim = 2**n
        print(f"  n={n}: |GHZ_{n}⟩ = (|{'0'*n}⟩ + |{'1'*n}⟩)/√2  (dim={dim})")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: GHZ state shows genuine 3-party entanglement -")
    print("correlations that cannot exist classically!")
    print("=" * 60)

if __name__ == "__main__":
    main()
