"""
Multiple-Qubit States - Concept 1: Introduction
================================================
From single qubits to multi-qubit systems.
Understanding the exponential growth of Hilbert spaces.
"""

import numpy as np

def main():
    print("=" * 60)
    print("MULTIPLE-QUBIT STATES: INTRODUCTION")
    print("=" * 60)
    
    # Single qubit state
    print("\n1. SINGLE QUBIT STATE")
    print("-" * 40)
    single_qubit = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    print(f"Single qubit |+⟩ = (|0⟩ + |1⟩)/√2")
    print(f"State vector: {single_qubit}")
    print(f"Dimension: {len(single_qubit)}")
    
    # Two qubit state (tensor product)
    print("\n2. TWO QUBIT STATE")
    print("-" * 40)
    qubit1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # |+⟩
    qubit2 = np.array([1, 0])  # |0⟩
    two_qubit = np.kron(qubit1, qubit2)
    print(f"|+⟩ ⊗ |0⟩ = {two_qubit}")
    print(f"Dimension: {len(two_qubit)}")
    print(f"This represents: {two_qubit[0]:.4f}|00⟩ + {two_qubit[1]:.4f}|01⟩ + {two_qubit[2]:.4f}|10⟩ + {two_qubit[3]:.4f}|11⟩")
    
    # Three qubit state
    print("\n3. THREE QUBIT STATE")
    print("-" * 40)
    qubit3 = np.array([1, 0])  # |0⟩
    three_qubit = np.kron(two_qubit, qubit3)
    print(f"|+⟩ ⊗ |0⟩ ⊗ |0⟩")
    print(f"State vector: {three_qubit}")
    print(f"Dimension: {len(three_qubit)}")
    
    # Hilbert space dimension growth
    print("\n4. HILBERT SPACE DIMENSION GROWTH")
    print("-" * 40)
    print(f"{'Qubits':<10} {'Dimension':<15} {'Description'}")
    print("-" * 40)
    for n in range(1, 11):
        dim = 2**n
        if n <= 3:
            desc = f"|{'0'*n}⟩ to |{'1'*n}⟩"
        elif n == 10:
            desc = "~1 thousand states"
        else:
            desc = f"{dim:,} basis states"
        print(f"{n:<10} {dim:<15,} {desc}")
    
    # Memory requirements
    print("\n5. CLASSICAL SIMULATION MEMORY REQUIREMENTS")
    print("-" * 40)
    print("To simulate n qubits classically, we need 2^n complex numbers")
    print("Each complex number = 16 bytes (complex128)")
    print()
    for n in [10, 20, 30, 40, 50]:
        dim = 2**n
        memory_bytes = dim * 16
        if memory_bytes < 1024**3:
            memory_str = f"{memory_bytes / 1024**2:.2f} MB"
        elif memory_bytes < 1024**4:
            memory_str = f"{memory_bytes / 1024**3:.2f} GB"
        elif memory_bytes < 1024**5:
            memory_str = f"{memory_bytes / 1024**4:.2f} TB"
        else:
            memory_str = f"{memory_bytes / 1024**5:.2f} PB"
        print(f"{n} qubits: {dim:>20,} amplitudes = {memory_str}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: n qubits live in a 2^n dimensional Hilbert space!")
    print("This exponential growth is the source of quantum parallelism.")
    print("=" * 60)

if __name__ == "__main__":
    main()
