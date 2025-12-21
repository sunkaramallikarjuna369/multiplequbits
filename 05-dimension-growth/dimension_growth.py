"""
Multiple-Qubit States - Concept 5: Dimension Growth
====================================================
Exponential scaling of Hilbert space dimension.
"""

import numpy as np

def main():
    print("=" * 60)
    print("DIMENSION GROWTH - EXPONENTIAL SCALING")
    print("=" * 60)
    
    # Dimension formula
    print("\n1. HILBERT SPACE DIMENSION")
    print("-" * 40)
    print("For n qubits: dim(H) = 2^n")
    print()
    print(f"{'n qubits':<12} {'Dimension':<20} {'Basis States'}")
    print("-" * 50)
    for n in range(1, 11):
        dim = 2**n
        if n <= 4:
            basis = f"|{'0'*n}⟩ ... |{'1'*n}⟩"
        else:
            basis = f"{dim:,} states"
        print(f"{n:<12} {dim:<20,} {basis}")
    
    # Memory requirements
    print("\n2. CLASSICAL SIMULATION MEMORY")
    print("-" * 40)
    print("Each amplitude: 16 bytes (complex128)")
    print()
    
    for n in [10, 20, 30, 40, 50, 60]:
        dim = 2**n
        bytes_needed = dim * 16
        
        if bytes_needed < 1024:
            mem = f"{bytes_needed} B"
        elif bytes_needed < 1024**2:
            mem = f"{bytes_needed/1024:.1f} KB"
        elif bytes_needed < 1024**3:
            mem = f"{bytes_needed/1024**2:.1f} MB"
        elif bytes_needed < 1024**4:
            mem = f"{bytes_needed/1024**3:.1f} GB"
        elif bytes_needed < 1024**5:
            mem = f"{bytes_needed/1024**4:.1f} TB"
        elif bytes_needed < 1024**6:
            mem = f"{bytes_needed/1024**5:.1f} PB"
        else:
            mem = f"{bytes_needed/1024**6:.1f} EB"
        
        print(f"{n:3} qubits: 2^{n} = {dim:>25,} amplitudes → {mem}")
    
    # Quantum supremacy threshold
    print("\n3. QUANTUM SUPREMACY THRESHOLD")
    print("-" * 40)
    print("Around 50-60 qubits, classical simulation becomes")
    print("practically impossible:")
    print()
    print("• 50 qubits: ~16 PB of RAM needed")
    print("• Google's Sycamore (2019): 53 qubits")
    print("• IBM's Eagle (2021): 127 qubits")
    print("• IBM's Condor (2023): 1,121 qubits")
    
    # Computational basis enumeration
    print("\n4. COMPUTATIONAL BASIS ENUMERATION")
    print("-" * 40)
    
    def binary_string(k, n):
        """Convert integer k to n-bit binary string."""
        return format(k, f'0{n}b')
    
    n = 3
    print(f"For n = {n} qubits, the 2^{n} = {2**n} basis states are:")
    print()
    for k in range(2**n):
        binary = binary_string(k, n)
        print(f"  |{binary}⟩ = |{k}⟩  (decimal notation)")
    
    # General state representation
    print("\n5. GENERAL n-QUBIT STATE")
    print("-" * 40)
    print("A general n-qubit pure state:")
    print()
    print("  |ψ⟩ = Σ_{k=0}^{2^n - 1} α_k |k⟩")
    print()
    print("where:")
    print("  • α_k are complex amplitudes")
    print("  • Σ|α_k|² = 1 (normalization)")
    print("  • |k⟩ is the k-th computational basis state")
    
    # Example: uniform superposition
    print("\n6. EXAMPLE: UNIFORM SUPERPOSITION")
    print("-" * 40)
    
    n = 3
    dim = 2**n
    uniform = np.ones(dim) / np.sqrt(dim)
    
    print(f"For n = {n} qubits:")
    print(f"|ψ⟩ = (1/√{dim}) Σ|k⟩ = H⊗{n} |{'0'*n}⟩")
    print()
    print("State vector:", np.round(uniform, 4))
    print(f"Each amplitude: 1/√{dim} = {1/np.sqrt(dim):.4f}")
    print(f"Each probability: 1/{dim} = {1/dim:.4f}")
    
    # Verify normalization
    norm_sq = np.sum(np.abs(uniform)**2)
    print(f"\nNormalization check: Σ|α_k|² = {norm_sq:.4f}")
    
    # Number of parameters
    print("\n7. DEGREES OF FREEDOM")
    print("-" * 40)
    print("For n qubits:")
    print(f"  • 2^n complex amplitudes = 2^{n+1} real numbers")
    print(f"  • Minus 1 for normalization")
    print(f"  • Minus 1 for global phase")
    print(f"  • = 2^{n+1} - 2 independent real parameters")
    print()
    for n in range(1, 6):
        params = 2**(n+1) - 2
        print(f"  {n} qubit(s): {params} real parameters")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: Exponential growth enables quantum parallelism")
    print("but makes classical simulation intractable!")
    print("=" * 60)

if __name__ == "__main__":
    main()
