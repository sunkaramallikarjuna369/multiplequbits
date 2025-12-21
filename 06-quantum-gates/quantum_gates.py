"""
Multiple-Qubit States - Concept 6: Quantum Gates
=================================================
Single-qubit gates on multi-qubit systems and the CNOT gate.
"""

import numpy as np

def main():
    print("=" * 60)
    print("QUANTUM GATES ON MULTIPLE QUBITS")
    print("=" * 60)
    
    # Single qubit gates
    print("\n1. SINGLE-QUBIT GATES")
    print("-" * 40)
    
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    
    print("Pauli matrices:")
    print(f"I = \n{I}")
    print(f"\nX (NOT) = \n{X}")
    print(f"\nZ (Phase) = \n{Z}")
    print(f"\nH (Hadamard) = \n{np.round(H, 4)}")
    
    # Applying single-qubit gate to multi-qubit system
    print("\n2. SINGLE-QUBIT GATE ON MULTI-QUBIT SYSTEM")
    print("-" * 40)
    print("To apply gate U to qubit i in an n-qubit system:")
    print("  U_i = I ⊗ ... ⊗ I ⊗ U ⊗ I ⊗ ... ⊗ I")
    print("        (U in position i)")
    
    # Example: X on first qubit of 2-qubit system
    X_first = np.kron(X, I)
    X_second = np.kron(I, X)
    
    print("\nFor 2 qubits:")
    print(f"X₁ = X ⊗ I = \n{X_first.astype(int)}")
    print(f"\nX₂ = I ⊗ X = \n{X_second.astype(int)}")
    
    # Verify action
    ket_00 = np.array([1, 0, 0, 0])
    print(f"\nX₁|00⟩ = {X_first @ ket_00} = |10⟩")
    print(f"X₂|00⟩ = {X_second @ ket_00} = |01⟩")
    
    # CNOT gate
    print("\n3. CNOT (CONTROLLED-NOT) GATE")
    print("-" * 40)
    
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])
    
    print("CNOT matrix (control=first, target=second):")
    print(CNOT)
    print()
    print("Action: CNOT|c,t⟩ = |c, t⊕c⟩")
    print("(XOR target with control)")
    
    # CNOT truth table
    print("\n4. CNOT TRUTH TABLE")
    print("-" * 40)
    print(f"{'Input':<10} {'Output':<10} {'Explanation'}")
    print("-" * 40)
    
    basis_states = [
        (np.array([1, 0, 0, 0]), '|00⟩'),
        (np.array([0, 1, 0, 0]), '|01⟩'),
        (np.array([0, 0, 1, 0]), '|10⟩'),
        (np.array([0, 0, 0, 1]), '|11⟩')
    ]
    
    explanations = [
        "Control=0, no flip",
        "Control=0, no flip",
        "Control=1, target flips 0→1",
        "Control=1, target flips 1→0"
    ]
    
    for (state, name), explanation in zip(basis_states, explanations):
        output = CNOT @ state
        output_name = ['|00⟩', '|01⟩', '|10⟩', '|11⟩'][np.argmax(output)]
        print(f"{name:<10} {output_name:<10} {explanation}")
    
    # CNOT on superposition
    print("\n5. CNOT ON SUPERPOSITION STATES")
    print("-" * 40)
    
    ket_0 = np.array([1, 0])
    ket_plus = np.array([1, 1]) / np.sqrt(2)
    
    # |+⟩|0⟩
    plus_zero = np.kron(ket_plus, ket_0)
    result = CNOT @ plus_zero
    print(f"|+⟩|0⟩ = {np.round(plus_zero, 4)}")
    print(f"CNOT|+⟩|0⟩ = {np.round(result, 4)}")
    print("         = (|00⟩ + |11⟩)/√2 = |Φ+⟩ (Bell state!)")
    
    # Reverse CNOT
    print("\n6. REVERSE CNOT (target controls)")
    print("-" * 40)
    
    CNOT_reverse = np.array([[1, 0, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0],
                             [0, 1, 0, 0]])
    
    print("CNOT_reverse (control=second, target=first):")
    print(CNOT_reverse)
    
    # Controlled-Z gate
    print("\n7. CONTROLLED-Z (CZ) GATE")
    print("-" * 40)
    
    CZ = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, -1]])
    
    print("CZ matrix:")
    print(CZ)
    print()
    print("Action: Applies Z to target when control is |1⟩")
    print("CZ|11⟩ = -|11⟩, all other basis states unchanged")
    print()
    print("Note: CZ is symmetric - either qubit can be 'control'!")
    
    # SWAP gate
    print("\n8. SWAP GATE")
    print("-" * 40)
    
    SWAP = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]])
    
    print("SWAP matrix:")
    print(SWAP)
    print()
    print("Action: SWAP|a,b⟩ = |b,a⟩")
    print("SWAP = CNOT₁₂ · CNOT₂₁ · CNOT₁₂ (3 CNOTs)")
    
    # Verify SWAP decomposition
    CNOT_12 = CNOT
    CNOT_21 = CNOT_reverse
    SWAP_from_CNOT = CNOT_12 @ CNOT_21 @ CNOT_12
    print(f"\nVerification: CNOT₁₂·CNOT₂₁·CNOT₁₂ = SWAP? {np.allclose(SWAP, SWAP_from_CNOT)}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: CNOT + single-qubit gates form a universal")
    print("gate set - any quantum computation can be built from them!")
    print("=" * 60)

if __name__ == "__main__":
    main()
