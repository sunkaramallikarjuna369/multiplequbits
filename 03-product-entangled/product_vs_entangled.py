"""
Multiple-Qubit States - Concept 3: Product vs Entangled States
===============================================================
Understanding separability and quantum correlations.
"""

import numpy as np

def is_separable_2qubit(state):
    """
    Check if a 2-qubit state is separable by attempting factorization.
    Uses SVD to determine if the state matrix has rank 1.
    Returns True if separable, False if entangled.
    """
    # Reshape state vector into 2x2 matrix
    matrix = state.reshape(2, 2)
    
    # Compute singular value decomposition
    U, S, Vh = np.linalg.svd(matrix)
    
    # Count non-zero singular values (with tolerance)
    rank = np.sum(S > 1e-10)
    
    # Separable iff rank = 1
    return rank == 1

def main():
    print("=" * 60)
    print("PRODUCT VS ENTANGLED STATES")
    print("=" * 60)
    
    # Define basis states
    ket_0 = np.array([1, 0])
    ket_1 = np.array([0, 1])
    ket_plus = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    
    # Product states
    print("\n1. PRODUCT STATES (SEPARABLE)")
    print("-" * 40)
    
    # |00⟩
    product_00 = np.kron(ket_0, ket_0)
    print(f"|00⟩ = {product_00}")
    print(f"Separable: {is_separable_2qubit(product_00)}")
    
    # |+⟩ ⊗ |0⟩
    product_plus0 = np.kron(ket_plus, ket_0)
    print(f"\n|+⟩ ⊗ |0⟩ = {product_plus0}")
    print(f"         = (|00⟩ + |10⟩)/√2")
    print(f"Separable: {is_separable_2qubit(product_plus0)}")
    
    # |+⟩ ⊗ |+⟩
    product_plusplus = np.kron(ket_plus, ket_plus)
    print(f"\n|+⟩ ⊗ |+⟩ = {product_plusplus}")
    print(f"         = (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2")
    print(f"Separable: {is_separable_2qubit(product_plusplus)}")
    
    # Entangled states (Bell states)
    print("\n2. ENTANGLED STATES (NON-SEPARABLE)")
    print("-" * 40)
    
    # |Φ+⟩ = (|00⟩ + |11⟩)/√2
    phi_plus = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    print(f"|Φ+⟩ = (|00⟩ + |11⟩)/√2 = {phi_plus}")
    print(f"Separable: {is_separable_2qubit(phi_plus)}")
    
    # |Ψ+⟩ = (|01⟩ + |10⟩)/√2
    psi_plus = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0])
    print(f"\n|Ψ+⟩ = (|01⟩ + |10⟩)/√2 = {psi_plus}")
    print(f"Separable: {is_separable_2qubit(psi_plus)}")
    
    # Why Bell states can't be factored
    print("\n3. WHY BELL STATES CAN'T BE FACTORED")
    print("-" * 40)
    print("For |Φ+⟩ = (|00⟩ + |11⟩)/√2, assume it equals |a⟩ ⊗ |b⟩")
    print("where |a⟩ = α|0⟩ + β|1⟩ and |b⟩ = γ|0⟩ + δ|1⟩")
    print()
    print("Then: |a⟩ ⊗ |b⟩ = αγ|00⟩ + αδ|01⟩ + βγ|10⟩ + βδ|11⟩")
    print()
    print("Matching coefficients:")
    print("  αγ = 1/√2  (coefficient of |00⟩)")
    print("  αδ = 0     (coefficient of |01⟩)")
    print("  βγ = 0     (coefficient of |10⟩)")
    print("  βδ = 1/√2  (coefficient of |11⟩)")
    print()
    print("From αδ = 0: either α = 0 or δ = 0")
    print("From βγ = 0: either β = 0 or γ = 0")
    print("But if α = 0, then αγ ≠ 1/√2. Contradiction!")
    print("Therefore, |Φ+⟩ CANNOT be factored.")
    
    # Physical meaning
    print("\n4. PHYSICAL MEANING OF ENTANGLEMENT")
    print("-" * 40)
    print("In the Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2:")
    print()
    print("• If qubit 1 measures |0⟩ → qubit 2 MUST be |0⟩")
    print("• If qubit 1 measures |1⟩ → qubit 2 MUST be |1⟩")
    print()
    print("This correlation is INSTANT, regardless of distance!")
    print("(But cannot be used for faster-than-light communication)")
    
    # Measurement correlations
    print("\n5. MEASUREMENT CORRELATIONS")
    print("-" * 40)
    
    def simulate_measurements(state, n_measurements=10000):
        """Simulate measurements and return correlation."""
        probs = np.abs(state)**2
        outcomes = np.random.choice(4, size=n_measurements, p=probs)
        
        # Count correlations
        same = np.sum((outcomes == 0) | (outcomes == 3))  # |00⟩ or |11⟩
        diff = np.sum((outcomes == 1) | (outcomes == 2))  # |01⟩ or |10⟩
        
        return same / n_measurements, diff / n_measurements
    
    print("Simulating 10,000 measurements...")
    print()
    
    same, diff = simulate_measurements(phi_plus)
    print(f"|Φ+⟩: Same outcomes: {same:.2%}, Different: {diff:.2%}")
    
    same, diff = simulate_measurements(psi_plus)
    print(f"|Ψ+⟩: Same outcomes: {same:.2%}, Different: {diff:.2%}")
    
    same, diff = simulate_measurements(product_plus0)
    print(f"|+⟩⊗|0⟩: Same outcomes: {same:.2%}, Different: {diff:.2%}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: Entangled states have correlations that")
    print("CANNOT be explained by any classical theory!")
    print("=" * 60)

if __name__ == "__main__":
    main()
