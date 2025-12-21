"""
Multiple-Qubit States - Concept 9: n-Qubit States
=================================================
General representation of n-qubit quantum states.
"""

import numpy as np

def main():
    print("=" * 60)
    print("n-QUBIT STATES - GENERAL REPRESENTATION")
    print("=" * 60)
    
    # General form
    print("\n1. GENERAL n-QUBIT PURE STATE")
    print("-" * 40)
    print("A general n-qubit pure state:")
    print()
    print("  |ψ⟩ = Σ_{k=0}^{2^n - 1} α_k |k⟩")
    print()
    print("where:")
    print("  • k ranges from 0 to 2^n - 1")
    print("  • |k⟩ is the k-th computational basis state")
    print("  • α_k ∈ ℂ are complex amplitudes")
    print("  • Σ|α_k|² = 1 (normalization)")
    
    # Binary encoding
    print("\n2. BINARY ENCODING")
    print("-" * 40)
    print("Each basis state |k⟩ corresponds to a binary string:")
    print()
    
    for n in range(1, 5):
        print(f"n = {n} qubits:")
        for k in range(min(2**n, 8)):
            binary = format(k, f'0{n}b')
            print(f"  |{k}⟩ = |{binary}⟩")
        if 2**n > 8:
            print(f"  ... ({2**n - 8} more states)")
        print()
    
    # Creating n-qubit states
    print("\n3. CREATING n-QUBIT STATES")
    print("-" * 40)
    
    def create_computational_basis(n, k):
        """Create the k-th computational basis state for n qubits."""
        state = np.zeros(2**n)
        state[k] = 1
        return state
    
    def create_uniform_superposition(n):
        """Create uniform superposition of all basis states."""
        return np.ones(2**n) / np.sqrt(2**n)
    
    def create_ghz_state(n):
        """Create GHZ state: (|00...0⟩ + |11...1⟩)/√2"""
        state = np.zeros(2**n)
        state[0] = 1 / np.sqrt(2)      # |00...0⟩
        state[-1] = 1 / np.sqrt(2)     # |11...1⟩
        return state
    
    def create_w_state(n):
        """Create W state: (|100...0⟩ + |010...0⟩ + ... + |00...01⟩)/√n"""
        state = np.zeros(2**n)
        for i in range(n):
            k = 2**(n - 1 - i)  # Single 1 in position i
            state[k] = 1 / np.sqrt(n)
        return state
    
    n = 3
    print(f"Examples for n = {n} qubits:")
    print()
    
    # Computational basis
    print(f"|000⟩ = {create_computational_basis(n, 0)}")
    print(f"|101⟩ = {create_computational_basis(n, 5)}")
    
    # Uniform superposition
    uniform = create_uniform_superposition(n)
    print(f"\nUniform superposition:")
    print(f"  |ψ⟩ = {np.round(uniform, 4)}")
    print(f"  Each amplitude = 1/√{2**n} = {1/np.sqrt(2**n):.4f}")
    
    # GHZ state
    ghz = create_ghz_state(n)
    print(f"\nGHZ state (|{'0'*n}⟩ + |{'1'*n}⟩)/√2:")
    print(f"  |GHZ⟩ = {ghz}")
    
    # W state
    w = create_w_state(n)
    print(f"\nW state:")
    print(f"  |W⟩ = {np.round(w, 4)}")
    
    # State space dimension
    print("\n4. STATE SPACE DIMENSION")
    print("-" * 40)
    print(f"{'n':<5} {'dim':<15} {'Real params':<15} {'Memory (complex128)'}")
    print("-" * 50)
    
    for n in range(1, 11):
        dim = 2**n
        real_params = 2*dim - 2  # Minus normalization and global phase
        memory = dim * 16  # 16 bytes per complex128
        
        if memory < 1024:
            mem_str = f"{memory} B"
        elif memory < 1024**2:
            mem_str = f"{memory/1024:.1f} KB"
        elif memory < 1024**3:
            mem_str = f"{memory/1024**2:.1f} MB"
        else:
            mem_str = f"{memory/1024**3:.1f} GB"
        
        print(f"{n:<5} {dim:<15} {real_params:<15} {mem_str}")
    
    # Tensor product construction
    print("\n5. TENSOR PRODUCT CONSTRUCTION")
    print("-" * 40)
    
    def tensor_product_states(states):
        """Compute tensor product of multiple single-qubit states."""
        result = states[0]
        for state in states[1:]:
            result = np.kron(result, state)
        return result
    
    ket_0 = np.array([1, 0])
    ket_1 = np.array([0, 1])
    ket_plus = np.array([1, 1]) / np.sqrt(2)
    
    # |+⟩ ⊗ |0⟩ ⊗ |1⟩
    state_plus_0_1 = tensor_product_states([ket_plus, ket_0, ket_1])
    print(f"|+⟩ ⊗ |0⟩ ⊗ |1⟩ = {np.round(state_plus_0_1, 4)}")
    print("             = (|001⟩ + |101⟩)/√2")
    
    # |+⟩ ⊗ |+⟩ ⊗ |+⟩
    state_plus_3 = tensor_product_states([ket_plus, ket_plus, ket_plus])
    print(f"\n|+⟩⊗³ = {np.round(state_plus_3, 4)}")
    print("     = uniform superposition of all 8 basis states")
    
    # Measurement probabilities
    print("\n6. MEASUREMENT PROBABILITIES")
    print("-" * 40)
    
    def print_measurement_probs(state, name):
        n = int(np.log2(len(state)))
        probs = np.abs(state)**2
        print(f"\n{name}:")
        for k in range(len(state)):
            if probs[k] > 1e-10:
                binary = format(k, f'0{n}b')
                print(f"  P(|{binary}⟩) = {probs[k]:.4f}")
    
    print_measurement_probs(ghz, "GHZ state")
    print_measurement_probs(w, "W state")
    print_measurement_probs(state_plus_0_1, "|+⟩⊗|0⟩⊗|1⟩")
    
    # Entanglement in n-qubit states
    print("\n7. ENTANGLEMENT IN n-QUBIT STATES")
    print("-" * 40)
    print("Types of multi-qubit entanglement:")
    print()
    print("• Fully separable: |ψ⟩ = |ψ₁⟩ ⊗ |ψ₂⟩ ⊗ ... ⊗ |ψₙ⟩")
    print("• Biseparable: Can be written as product across some bipartition")
    print("• Genuinely multipartite entangled: Not separable across ANY cut")
    print()
    print("Examples:")
    print("  |+⟩⊗ⁿ: Fully separable")
    print("  |GHZ⟩: Genuinely multipartite entangled")
    print("  |W⟩: Genuinely multipartite entangled (different type)")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: n-qubit states live in 2^n dimensional space,")
    print("enabling exponential quantum parallelism!")
    print("=" * 60)

if __name__ == "__main__":
    main()
