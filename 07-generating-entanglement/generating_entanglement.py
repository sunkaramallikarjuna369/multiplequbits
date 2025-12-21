"""
Multiple-Qubit States - Concept 7: Generating Entanglement
==========================================================
Creating entangled states with Hadamard and CNOT gates.
"""

import numpy as np

def main():
    print("=" * 60)
    print("GENERATING ENTANGLEMENT")
    print("=" * 60)
    
    # Define gates
    I = np.eye(2)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])
    
    # Define basis states
    ket_0 = np.array([1, 0])
    ket_1 = np.array([0, 1])
    
    # Bell state creation circuit
    print("\n1. BELL STATE CREATION CIRCUIT")
    print("-" * 40)
    print("Circuit: |00⟩ → H⊗I → CNOT → |Φ+⟩")
    print()
    print("     ┌───┐")
    print("|0⟩──┤ H ├──●──")
    print("     └───┘  │")
    print("            │")
    print("|0⟩────────⊕──")
    print()
    
    # Step by step
    print("\n2. STEP-BY-STEP CREATION OF |Φ+⟩")
    print("-" * 40)
    
    # Initial state
    ket_00 = np.kron(ket_0, ket_0)
    print(f"Step 0: |00⟩ = {ket_00}")
    
    # After Hadamard on first qubit
    H_I = np.kron(H, I)
    after_H = H_I @ ket_00
    print(f"\nStep 1: (H⊗I)|00⟩ = {np.round(after_H, 4)}")
    print("       = (|0⟩ + |1⟩)/√2 ⊗ |0⟩")
    print("       = (|00⟩ + |10⟩)/√2")
    
    # After CNOT
    bell_state = CNOT @ after_H
    print(f"\nStep 2: CNOT(|00⟩ + |10⟩)/√2 = {np.round(bell_state, 4)}")
    print("       = (|00⟩ + |11⟩)/√2 = |Φ+⟩")
    
    # Creating all four Bell states
    print("\n3. CREATING ALL FOUR BELL STATES")
    print("-" * 40)
    
    initial_states = [
        (np.kron(ket_0, ket_0), '|00⟩', '|Φ+⟩'),
        (np.kron(ket_0, ket_1), '|01⟩', '|Ψ+⟩'),
        (np.kron(ket_1, ket_0), '|10⟩', '|Φ-⟩'),
        (np.kron(ket_1, ket_1), '|11⟩', '|Ψ-⟩')
    ]
    
    print(f"{'Initial':<10} {'After H⊗I':<25} {'After CNOT':<25} {'Bell State'}")
    print("-" * 75)
    
    for initial, init_name, bell_name in initial_states:
        after_H = H_I @ initial
        final = CNOT @ after_H
        
        # Format vectors
        after_H_str = np.array2string(np.round(after_H, 3), precision=3)
        final_str = np.array2string(np.round(final, 3), precision=3)
        
        print(f"{init_name:<10} {after_H_str:<25} {final_str:<25} {bell_name}")
    
    # Why entanglement is created
    print("\n4. WHY ENTANGLEMENT IS CREATED")
    print("-" * 40)
    print("The key insight:")
    print()
    print("1. Hadamard creates superposition: |0⟩ → (|0⟩ + |1⟩)/√2")
    print()
    print("2. CNOT creates correlation:")
    print("   • When control is |0⟩, target stays |0⟩ → |00⟩")
    print("   • When control is |1⟩, target flips to |1⟩ → |11⟩")
    print()
    print("3. Result: (|00⟩ + |11⟩)/√2")
    print("   The two qubits are now correlated!")
    
    # Entanglement verification
    print("\n5. VERIFYING ENTANGLEMENT")
    print("-" * 40)
    
    def is_separable(state):
        """Check if 2-qubit state is separable using SVD."""
        matrix = state.reshape(2, 2)
        _, S, _ = np.linalg.svd(matrix)
        rank = np.sum(S > 1e-10)
        return rank == 1
    
    print(f"|Φ+⟩ is separable: {is_separable(bell_state)}")
    print(f"|00⟩ is separable: {is_separable(ket_00)}")
    print(f"(|00⟩+|10⟩)/√2 is separable: {is_separable(after_H)}")
    
    # Entanglement measure
    print("\n6. MEASURING ENTANGLEMENT")
    print("-" * 40)
    
    def entanglement_entropy(state):
        """Calculate entanglement entropy via partial trace."""
        rho = np.outer(state, np.conj(state))
        
        # Partial trace over second qubit
        rho_A = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    rho_A[i, j] += rho[2*i + k, 2*j + k]
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Von Neumann entropy
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        return entropy
    
    print("Entanglement entropy (von Neumann):")
    print(f"  |00⟩: {entanglement_entropy(ket_00):.4f} bits")
    print(f"  (|00⟩+|10⟩)/√2: {entanglement_entropy(after_H):.4f} bits")
    print(f"  |Φ+⟩: {entanglement_entropy(bell_state):.4f} bits")
    print()
    print("Maximum entanglement = 1 bit (for 2 qubits)")
    
    # Three-qubit entanglement
    print("\n7. EXTENDING TO THREE QUBITS (GHZ)")
    print("-" * 40)
    print("Circuit for GHZ state:")
    print()
    print("|0⟩──H──●────●──")
    print("        │    │")
    print("|0⟩────⊕────│──")
    print("             │")
    print("|0⟩─────────⊕──")
    print()
    
    # Create GHZ state
    ket_000 = np.kron(np.kron(ket_0, ket_0), ket_0)
    
    # H on first qubit
    H_I_I = np.kron(np.kron(H, I), I)
    after_H_3 = H_I_I @ ket_000
    
    # CNOT on qubits 1,2
    CNOT_12 = np.kron(CNOT, I)
    after_CNOT1 = CNOT_12 @ after_H_3
    
    # CNOT on qubits 1,3
    CNOT_13 = np.zeros((8, 8))
    for i in range(8):
        binary = format(i, '03b')
        if binary[0] == '1':
            new_binary = binary[0] + binary[1] + str(1 - int(binary[2]))
            j = int(new_binary, 2)
        else:
            j = i
        CNOT_13[j, i] = 1
    
    ghz = CNOT_13 @ after_CNOT1
    print(f"GHZ state = {np.round(ghz, 4)}")
    print("         = (|000⟩ + |111⟩)/√2")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: Hadamard creates superposition,")
    print("CNOT creates correlation → Together they create entanglement!")
    print("=" * 60)

if __name__ == "__main__":
    main()
