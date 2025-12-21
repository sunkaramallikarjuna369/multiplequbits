"""
Multiple-Qubit States - Concept 12: Exercises
=============================================
Practice problems with solutions.
"""

import numpy as np

def exercise_1():
    """Tensor Product Calculation"""
    print("\n" + "=" * 60)
    print("EXERCISE 1: Tensor Product Calculation")
    print("=" * 60)
    print("\nProblem: Calculate |+⟩ ⊗ |1⟩ where |+⟩ = (|0⟩ + |1⟩)/√2")
    print()
    input("Press Enter to see solution...")
    
    ket_plus = np.array([1, 1]) / np.sqrt(2)
    ket_1 = np.array([0, 1])
    result = np.kron(ket_plus, ket_1)
    
    print("\nSolution:")
    print("  |+⟩ ⊗ |1⟩ = (|0⟩ + |1⟩)/√2 ⊗ |1⟩")
    print("           = (|0⟩ ⊗ |1⟩ + |1⟩ ⊗ |1⟩)/√2")
    print("           = (|01⟩ + |11⟩)/√2")
    print(f"\n  As a vector: {np.round(result, 4)}")
    print(f"  = [0, 1/√2, 0, 1/√2]ᵀ")

def exercise_2():
    """Separability Test"""
    print("\n" + "=" * 60)
    print("EXERCISE 2: Separability Test")
    print("=" * 60)
    print("\nProblem: Is |ψ⟩ = (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2 entangled?")
    print()
    input("Press Enter to see solution...")
    
    state = np.array([1, 1, 1, 1]) / 2
    
    # Check separability using SVD
    matrix = state.reshape(2, 2)
    U, S, Vh = np.linalg.svd(matrix)
    rank = np.sum(S > 1e-10)
    
    print("\nSolution:")
    print("  Reshape state into 2x2 matrix and compute SVD:")
    print(f"  Matrix = \n{matrix}")
    print(f"  Singular values: {np.round(S, 4)}")
    print(f"  Rank = {rank}")
    print()
    print("  Since rank = 1, the state IS SEPARABLE!")
    print()
    print("  Factorization:")
    print("  |ψ⟩ = (|0⟩ + |1⟩)/√2 ⊗ (|0⟩ + |1⟩)/√2 = |+⟩ ⊗ |+⟩")

def exercise_3():
    """CNOT Action"""
    print("\n" + "=" * 60)
    print("EXERCISE 3: CNOT Action")
    print("=" * 60)
    print("\nProblem: What is CNOT applied to (|0⟩ + |1⟩)/√2 ⊗ |1⟩?")
    print()
    input("Press Enter to see solution...")
    
    ket_plus = np.array([1, 1]) / np.sqrt(2)
    ket_1 = np.array([0, 1])
    initial = np.kron(ket_plus, ket_1)
    
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])
    
    result = CNOT @ initial
    
    print("\nSolution:")
    print("  Input: |ψ⟩ = (|0⟩ + |1⟩)/√2 ⊗ |1⟩ = (|01⟩ + |11⟩)/√2")
    print(f"  Input vector: {np.round(initial, 4)}")
    print()
    print("  Apply CNOT:")
    print("    CNOT|01⟩ = |01⟩  (control=0, no flip)")
    print("    CNOT|11⟩ = |10⟩  (control=1, target flips)")
    print()
    print(f"  Result: {np.round(result, 4)}")
    print("        = (|01⟩ + |10⟩)/√2 = |Ψ⁺⟩ (Bell state!)")

def exercise_4():
    """Measurement Probabilities"""
    print("\n" + "=" * 60)
    print("EXERCISE 4: Measurement Probabilities")
    print("=" * 60)
    print("\nProblem: For |ψ⟩ = (|00⟩ + 2|01⟩ + |10⟩)/√6,")
    print("         calculate all measurement probabilities.")
    print()
    input("Press Enter to see solution...")
    
    state = np.array([1, 2, 1, 0]) / np.sqrt(6)
    probs = np.abs(state)**2
    
    print("\nSolution:")
    print(f"  Amplitudes: α₀₀ = 1/√6, α₀₁ = 2/√6, α₁₀ = 1/√6, α₁₁ = 0")
    print()
    print("  Probabilities P = |α|²:")
    print(f"    P(00) = |1/√6|² = 1/6 = {probs[0]:.4f} ≈ 16.67%")
    print(f"    P(01) = |2/√6|² = 4/6 = {probs[1]:.4f} ≈ 66.67%")
    print(f"    P(10) = |1/√6|² = 1/6 = {probs[2]:.4f} ≈ 16.67%")
    print(f"    P(11) = |0|² = 0 = {probs[3]:.4f}")
    print()
    print(f"  Verification: Σ P = {np.sum(probs):.4f} ✓")

def exercise_5():
    """Bell State Identification"""
    print("\n" + "=" * 60)
    print("EXERCISE 5: Bell State Identification")
    print("=" * 60)
    print("\nProblem: Which Bell state gives opposite measurement results?")
    print()
    input("Press Enter to see solution...")
    
    print("\nSolution:")
    print("  The four Bell states:")
    print()
    print("  |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 → Same outcomes (00 or 11)")
    print("  |Φ⁻⟩ = (|00⟩ - |11⟩)/√2 → Same outcomes (00 or 11)")
    print("  |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2 → OPPOSITE outcomes (01 or 10)")
    print("  |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2 → OPPOSITE outcomes (01 or 10)")
    print()
    print("  Answer: |Ψ⁺⟩ and |Ψ⁻⟩ give opposite (anti-correlated) results!")

def exercise_6():
    """Dimension Calculation"""
    print("\n" + "=" * 60)
    print("EXERCISE 6: Dimension Calculation")
    print("=" * 60)
    print("\nProblem: How many complex numbers describe a 7-qubit pure state?")
    print()
    input("Press Enter to see solution...")
    
    n = 7
    dim = 2**n
    
    print("\nSolution:")
    print(f"  For n qubits: dim(H) = 2ⁿ")
    print(f"  For n = {n}: dim = 2⁷ = {dim}")
    print()
    print(f"  Answer: {dim} complex numbers")
    print()
    print("  Note: Due to normalization and global phase,")
    print(f"  only 2×{dim} - 2 = {2*dim - 2} real parameters are independent.")

def exercise_7():
    """Creating Entanglement"""
    print("\n" + "=" * 60)
    print("EXERCISE 7: Creating Entanglement")
    print("=" * 60)
    print("\nProblem: Starting from |01⟩, create |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2")
    print()
    input("Press Enter to see solution...")
    
    I = np.eye(2)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])
    
    ket_01 = np.array([0, 1, 0, 0])
    
    # Apply H⊗I then CNOT
    H_I = np.kron(H, I)
    after_H = H_I @ ket_01
    result = CNOT @ after_H
    
    print("\nSolution:")
    print("  Apply H ⊗ I followed by CNOT:")
    print()
    print(f"  Step 1: |01⟩ = {ket_01}")
    print(f"  Step 2: (H⊗I)|01⟩ = {np.round(after_H, 4)}")
    print("                    = (|0⟩ + |1⟩)/√2 ⊗ |1⟩ = (|01⟩ + |11⟩)/√2")
    print(f"  Step 3: CNOT = {np.round(result, 4)}")
    print("               = (|01⟩ + |10⟩)/√2 = |Ψ⁺⟩ ✓")

def exercise_8():
    """Partial Trace"""
    print("\n" + "=" * 60)
    print("EXERCISE 8: Partial Trace")
    print("=" * 60)
    print("\nProblem: What is ρ_A for the state |01⟩?")
    print()
    input("Press Enter to see solution...")
    
    ket_01 = np.array([0, 1, 0, 0])
    rho = np.outer(ket_01, ket_01)
    
    # Partial trace over B
    rho_A = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                rho_A[i, j] += rho[2*i + k, 2*j + k]
    
    print("\nSolution:")
    print("  |01⟩ = |0⟩_A ⊗ |1⟩_B")
    print()
    print("  ρ_AB = |01⟩⟨01|")
    print()
    print("  ρ_A = Tr_B(ρ_AB) = |0⟩⟨0|")
    print()
    print(f"  ρ_A = \n{rho_A}")
    print()
    print("  This is a pure state (as expected for a product state).")

def main():
    print("=" * 60)
    print("MULTIPLE-QUBIT STATES: EXERCISES")
    print("=" * 60)
    print("\nThis program contains 8 practice exercises.")
    print("Work through each problem before revealing the solution!")
    
    exercises = [
        exercise_1,
        exercise_2,
        exercise_3,
        exercise_4,
        exercise_5,
        exercise_6,
        exercise_7,
        exercise_8
    ]
    
    while True:
        print("\n" + "-" * 40)
        print("Select an exercise (1-8) or 'q' to quit:")
        choice = input("> ").strip().lower()
        
        if choice == 'q':
            print("\nGoodbye! Keep practicing quantum mechanics!")
            break
        
        try:
            ex_num = int(choice)
            if 1 <= ex_num <= 8:
                exercises[ex_num - 1]()
            else:
                print("Please enter a number between 1 and 8.")
        except ValueError:
            print("Invalid input. Enter a number 1-8 or 'q' to quit.")

if __name__ == "__main__":
    main()
