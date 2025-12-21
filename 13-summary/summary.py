"""
Multiple-Qubit States - Concept 13: Summary
===========================================
Complete overview and quick reference.
"""

import numpy as np

def main():
    print("=" * 70)
    print("MULTIPLE-QUBIT STATES: COMPLETE SUMMARY")
    print("=" * 70)
    
    # 1. Hilbert Space
    print("\n" + "=" * 70)
    print("1. HILBERT SPACE")
    print("=" * 70)
    print("""
    n qubits live in a 2^n dimensional complex Hilbert space:
    
        H_n = C^(2^n)
    
    Dimension growth:
        1 qubit  → 2 dimensions
        2 qubits → 4 dimensions
        3 qubits → 8 dimensions
        n qubits → 2^n dimensions
    """)
    
    # 2. Tensor Product
    print("=" * 70)
    print("2. TENSOR PRODUCT")
    print("=" * 70)
    print("""
    Multi-qubit states are built using tensor products:
    
        |ψ⟩ = |ψ₁⟩ ⊗ |ψ₂⟩ ⊗ ... ⊗ |ψₙ⟩
    
    For vectors:
        |a⟩ ⊗ |b⟩ = [a₀b₀, a₀b₁, a₁b₀, a₁b₁]ᵀ
    
    NumPy: np.kron(a, b)
    """)
    
    # 3. Product vs Entangled
    print("=" * 70)
    print("3. PRODUCT VS ENTANGLED STATES")
    print("=" * 70)
    print("""
    PRODUCT (Separable):
        Can be written as |ψ⟩ = |a⟩ ⊗ |b⟩
        Example: |00⟩, |+⟩⊗|0⟩
    
    ENTANGLED (Non-separable):
        CANNOT be factored into single-qubit states
        Example: (|00⟩ + |11⟩)/√2
    
    Test: Reshape to matrix, check if rank = 1 (separable)
    """)
    
    # 4. Bell States
    print("=" * 70)
    print("4. BELL STATES")
    print("=" * 70)
    print("""
    Four maximally entangled two-qubit states:
    
        |Φ⁺⟩ = (|00⟩ + |11⟩)/√2   (correlated)
        |Φ⁻⟩ = (|00⟩ - |11⟩)/√2   (correlated)
        |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2   (anti-correlated)
        |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2   (anti-correlated)
    
    These form an orthonormal basis for 2-qubit space.
    """)
    
    # 5. CNOT Gate
    print("=" * 70)
    print("5. CNOT GATE")
    print("=" * 70)
    print("""
    Controlled-NOT gate:
    
        CNOT|c,t⟩ = |c, t⊕c⟩
    
    Truth table:
        |00⟩ → |00⟩
        |01⟩ → |01⟩
        |10⟩ → |11⟩
        |11⟩ → |10⟩
    
    Matrix:
        [[1,0,0,0],
         [0,1,0,0],
         [0,0,0,1],
         [0,0,1,0]]
    """)
    
    # 6. Creating Entanglement
    print("=" * 70)
    print("6. CREATING ENTANGLEMENT")
    print("=" * 70)
    print("""
    Bell state creation circuit:
    
        |00⟩ → H⊗I → CNOT → |Φ⁺⟩
    
    Step by step:
        |00⟩ → (|0⟩+|1⟩)/√2 ⊗ |0⟩ → (|00⟩+|11⟩)/√2
    
    Key: Hadamard creates superposition, CNOT creates correlation
    """)
    
    # 7. Measurement
    print("=" * 70)
    print("7. MEASUREMENT")
    print("=" * 70)
    print("""
    Born rule:
        P(k) = |αₖ|²
    
    For |ψ⟩ = Σ αₖ|k⟩:
        - Probability of outcome k is |αₖ|²
        - State collapses to |k⟩ after measurement
        - Σ|αₖ|² = 1 (normalization)
    """)
    
    # 8. Partial Trace
    print("=" * 70)
    print("8. PARTIAL TRACE")
    print("=" * 70)
    print("""
    Reduced density matrix:
        ρ_A = Tr_B(ρ_AB)
    
    Entanglement indicator:
        - Pure reduced state → Product state
        - Mixed reduced state → Entangled state
    
    Purity: Tr(ρ²)
        = 1 for pure states
        < 1 for mixed states
    """)
    
    # 9. GHZ State
    print("=" * 70)
    print("9. GHZ STATE")
    print("=" * 70)
    print("""
    Three-qubit GHZ state:
        |GHZ⟩ = (|000⟩ + |111⟩)/√2
    
    Properties:
        - Genuinely tripartite entangled
        - All qubits perfectly correlated
        - Fragile: losing one qubit destroys entanglement
    
    General: |GHZ_n⟩ = (|0...0⟩ + |1...1⟩)/√2
    """)
    
    # Quick Reference Code
    print("=" * 70)
    print("QUICK REFERENCE: PYTHON CODE")
    print("=" * 70)
    
    code = '''
import numpy as np

# === BASIC STATES ===
ket_0 = np.array([1, 0])
ket_1 = np.array([0, 1])
ket_plus = np.array([1, 1]) / np.sqrt(2)

# === GATES ===
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
CNOT = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]])

# === TENSOR PRODUCT ===
ket_00 = np.kron(ket_0, ket_0)

# === BELL STATE ===
bell = CNOT @ np.kron(H, I) @ ket_00

# === MEASUREMENT ===
probs = np.abs(state)**2

# === SEPARABILITY CHECK ===
def is_separable(state):
    matrix = state.reshape(2, 2)
    _, S, _ = np.linalg.svd(matrix)
    return np.sum(S > 1e-10) == 1

# === PARTIAL TRACE ===
def partial_trace_B(rho):
    rho_A = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                rho_A[i, j] += rho[2*i + k, 2*j + k]
    return rho_A
'''
    print(code)
    
    # Key Insights
    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
    1. EXPONENTIAL GROWTH
       n qubits → 2^n dimensional space
       Source of quantum computational power
    
    2. ENTANGLEMENT
       Non-classical correlations
       Cannot be explained by local hidden variables
       Key resource for quantum computing
    
    3. MEASUREMENT
       Collapses superposition
       Reveals correlations in entangled states
       Probabilistic outcomes
    
    4. UNIVERSAL GATE SET
       H + CNOT can create any quantum state
       Foundation of quantum circuits
    """)
    
    print("=" * 70)
    print("Congratulations! You've completed the Multiple-Qubit States guide!")
    print("=" * 70)

if __name__ == "__main__":
    main()
