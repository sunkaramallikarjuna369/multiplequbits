"""
Multiple-Qubit States - Concept 2: Tensor Product Structure
============================================================
Building multi-qubit states from single qubits using tensor products.
"""

import numpy as np

def main():
    print("=" * 60)
    print("TENSOR PRODUCT STRUCTURE")
    print("=" * 60)
    
    # Define single qubit states
    print("\n1. SINGLE QUBIT BASIS STATES")
    print("-" * 40)
    ket_0 = np.array([1, 0])
    ket_1 = np.array([0, 1])
    ket_plus = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    ket_minus = np.array([1/np.sqrt(2), -1/np.sqrt(2)])
    
    print(f"|0⟩ = {ket_0}")
    print(f"|1⟩ = {ket_1}")
    print(f"|+⟩ = (|0⟩ + |1⟩)/√2 = {ket_plus}")
    print(f"|-⟩ = (|0⟩ - |1⟩)/√2 = {ket_minus}")
    
    # Two-qubit computational basis
    print("\n2. TWO-QUBIT COMPUTATIONAL BASIS")
    print("-" * 40)
    ket_00 = np.kron(ket_0, ket_0)
    ket_01 = np.kron(ket_0, ket_1)
    ket_10 = np.kron(ket_1, ket_0)
    ket_11 = np.kron(ket_1, ket_1)
    
    print(f"|00⟩ = |0⟩ ⊗ |0⟩ = {ket_00}")
    print(f"|01⟩ = |0⟩ ⊗ |1⟩ = {ket_01}")
    print(f"|10⟩ = |1⟩ ⊗ |0⟩ = {ket_10}")
    print(f"|11⟩ = |1⟩ ⊗ |1⟩ = {ket_11}")
    
    # Tensor product examples
    print("\n3. TENSOR PRODUCT EXAMPLES")
    print("-" * 40)
    
    # |+⟩ ⊗ |0⟩
    plus_zero = np.kron(ket_plus, ket_0)
    print(f"|+⟩ ⊗ |0⟩ = {plus_zero}")
    print(f"         = {plus_zero[0]:.4f}|00⟩ + {plus_zero[1]:.4f}|01⟩ + {plus_zero[2]:.4f}|10⟩ + {plus_zero[3]:.4f}|11⟩")
    
    # |+⟩ ⊗ |+⟩
    plus_plus = np.kron(ket_plus, ket_plus)
    print(f"\n|+⟩ ⊗ |+⟩ = {plus_plus}")
    print(f"         = {plus_plus[0]:.4f}|00⟩ + {plus_plus[1]:.4f}|01⟩ + {plus_plus[2]:.4f}|10⟩ + {plus_plus[3]:.4f}|11⟩")
    
    # Kronecker product formula
    print("\n4. KRONECKER PRODUCT FORMULA")
    print("-" * 40)
    print("For |a⟩ = [a₀, a₁]ᵀ and |b⟩ = [b₀, b₁]ᵀ:")
    print("|a⟩ ⊗ |b⟩ = [a₀b₀, a₀b₁, a₁b₀, a₁b₁]ᵀ")
    
    a = np.array([0.6, 0.8])
    b = np.array([0.8, 0.6])
    ab = np.kron(a, b)
    print(f"\nExample: |a⟩ = {a}, |b⟩ = {b}")
    print(f"|a⟩ ⊗ |b⟩ = [{a[0]*b[0]:.2f}, {a[0]*b[1]:.2f}, {a[1]*b[0]:.2f}, {a[1]*b[1]:.2f}]")
    print(f"         = {ab}")
    
    # Verify normalization
    print("\n5. NORMALIZATION CHECK")
    print("-" * 40)
    print(f"||a⟩|² = {np.sum(np.abs(a)**2):.4f}")
    print(f"||b⟩|² = {np.sum(np.abs(b)**2):.4f}")
    print(f"||a⟩ ⊗ |b⟩|² = {np.sum(np.abs(ab)**2):.4f}")
    print("Product of normalized states is normalized!")
    
    # Three-qubit tensor product
    print("\n6. THREE-QUBIT TENSOR PRODUCT")
    print("-" * 40)
    ket_000 = np.kron(np.kron(ket_0, ket_0), ket_0)
    ket_111 = np.kron(np.kron(ket_1, ket_1), ket_1)
    print(f"|000⟩ = {ket_000}")
    print(f"|111⟩ = {ket_111}")
    print(f"Dimension: {len(ket_000)} (2³ = 8)")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: Tensor product combines qubit spaces")
    print("dim(A ⊗ B) = dim(A) × dim(B)")
    print("=" * 60)

if __name__ == "__main__":
    main()
