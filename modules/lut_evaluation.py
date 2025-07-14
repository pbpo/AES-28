from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
import desilofhe  # type: ignore – external homomorphic‑encryption back‑end


class LUTEvaluator:
    """Handles efficient LUT evaluation using pre‑computed monomial bases."""

    # ---------------------------------------------------------------------
    # 0) Construction & frequently‑used constants
    # ---------------------------------------------------------------------

    def __init__(self, engine: desilofhe.Engine, keys: Dict, *, verbose: bool = False):
        self.engine = engine
        self.pk = keys["pk"]
        self.relin_key = keys["relin_key"]
        self.conj_key = keys.get("conj_key")  # may be *None* if backend lacks support
        self.verbose = verbose
        self.slot_count = engine.slot_count

        # Cache for plaintext/​ciphertext that represents the all‑ones vector.
        one_vec = np.ones(self.slot_count, dtype=complex)
        self._one_pt = self.engine.encode(one_vec)
        one_ct = self.engine.encrypt(one_vec, self.pk)  # encrypted at highest level
        self._one_ct_cache: Dict[int, desilofhe.Ciphertext] = {one_ct.level: one_ct}

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:  # pragma: no cover – debug utility
        if self.verbose:
            print(f"[LUT] {msg}")

    def _get_one_ct(self, level: int) -> desilofhe.Ciphertext:
        """Return an encrypted constant 1 matched to *level* (with caching)."""
        if level in self._one_ct_cache:
            return self._one_ct_cache[level]

        # Downgrade from the highest‑level copy already cached.
        base_level = max(self._one_ct_cache)
        base_ct = self._one_ct_cache[base_level]
        if base_level > level:
            ct = self.engine.level_down(base_ct, level)
            self._one_ct_cache[level] = ct
            return ct

        # (Rare) request for *higher* level than we currently have.
        # If the backend lacks level‑up we simply return the highest available.
        self._log("Warning: requested higher level than cached – returning best effort")
        return base_ct

    # ------------------------------------------------------------------
    # 1) Shared monomial basis computation
    # ------------------------------------------------------------------

    def compute_monomial_basis_optimized(
        self, input_cts: List[desilofhe.Ciphertext]
    ) -> Dict[int, desilofhe.Ciphertext]:
        """
        Compute all monomials up to degree n for n = len(input_cts).
        
        Uses conjugation optimization from the paper (Section 3.4):
        - For code C = {ζ^j}, we have x^(n-j) = conjugate(x^j) for all x ∈ C
        - This reduces depth and uses less expensive Galois operations instead of multiplications
        """
        n = len(input_cts)
        basis: Dict[int, desilofhe.Ciphertext] = {}
        
        # Store the constant 1 ciphertext
        basis[0] = self._get_one_ct(input_cts[0].level)
        
        # Stage 1: Single variable monomials (degree 1)
        for i, ct in enumerate(input_cts):
            basis[1 << i] = ct
        
        # Stage 2: Compute monomials up to degree n/2 using multiplication
        # We only need to compute up to n/2 because higher degrees can be obtained via conjugation
        max_direct_degree = n // 2 if self.conj_key is not None else n
        
        # Pre-compute which monomials need conjugation
        full_mask = (1 << n) - 1
        conjugate_pairs = {}
        for idx in range(1 << n):
            conj_idx = full_mask ^ idx
            if idx < conj_idx:  # Store only one direction to avoid duplicates
                conjugate_pairs[idx] = conj_idx
        
        # Compute lower degree monomials first (for conjugation later)
        for degree in range(2, n + 1):
            # Skip higher degrees if we can use conjugation
            if self.conj_key is not None and degree > max_direct_degree:
                continue
                
            for subset in itertools.combinations(range(n), degree):
                idx = sum(1 << v for v in subset)
                
                # Skip if already computed
                if idx in basis:
                    continue
                    
                # Skip if we'll compute this via conjugation later
                if self.conj_key is not None and bin(idx).count('1') > max_direct_degree:
                    continue
                
                # Find optimal factorization using dynamic programming approach
                factors = self._find_optimal_factorization(idx, basis)
                
                if factors:
                    idx1, idx2 = factors
                    ct1, ct2 = basis[idx1], basis[idx2]
                    
                    # Level synchronization
                    if ct1.level != ct2.level:
                        if ct1.level > ct2.level:
                            ct1 = self.engine.level_down(ct1, ct2.level)
                        else:
                            ct2 = self.engine.level_down(ct2, ct1.level)
                    
                    # Multiply and relinearize
                    prod = self.engine.multiply(ct1, ct2)
                    prod = self.engine.relinearize(prod, self.relin_key)
                    prod = self.engine.rescale(prod)
                    basis[idx] = prod
        
        # Stage 3: Use conjugation for remaining monomials
        if self.conj_key is not None:
            self._log(f"Applying conjugation optimization for high-degree monomials")
            
            # Sort indices by degree to ensure we compute lower degrees first
            remaining_indices = []
            for idx in range(1 << n):
                if idx not in basis and bin(idx).count('1') > max_direct_degree:
                    remaining_indices.append(idx)
            
            remaining_indices.sort(key=lambda x: bin(x).count('1'))
            
            for idx in remaining_indices:
                if idx in basis:
                    continue
                    
                conj_idx = full_mask ^ idx
                
                if conj_idx in basis:
                    try:
                        basis[idx] = self._conjugate_ciphertext(basis[conj_idx])
                        self._log(f"Computed monomial {bin(idx)} (deg={bin(idx).count('1')}) via conjugation")
                    except Exception as e:
                        self._log(f"Conjugation failed for {bin(idx)}: {e}")
                        # Fallback to multiplication
                        basis[idx] = self._compute_monomial_via_multiplication(idx, basis)
                else:
                    # Both idx and conj_idx are missing - compute the lower degree one first
                    if bin(idx).count('1') <= bin(conj_idx).count('1'):
                        basis[idx] = self._compute_monomial_via_multiplication(idx, basis)
                        if conj_idx not in basis:
                            basis[conj_idx] = self._conjugate_ciphertext(basis[idx])
                    else:
                        basis[conj_idx] = self._compute_monomial_via_multiplication(conj_idx, basis)
                        basis[idx] = self._conjugate_ciphertext(basis[conj_idx])
        
        # Stage 4: Fill any remaining gaps
        for idx in range(1 << n):
            if idx not in basis:
                self._log(f"Computing missing monomial {bin(idx)} via multiplication")
                basis[idx] = self._compute_monomial_via_multiplication(idx, basis)
        
        self._log(f"Monomial basis complete: {len(basis)} monomials")
        self._log(f"Used conjugation: {self.conj_key is not None}")
        
        return basis


    def _find_optimal_factorization(
        self, 
        target_idx: int, 
        basis: Dict[int, desilofhe.Ciphertext]
    ) -> Optional[Tuple[int, int]]:
        """
        Find the optimal factorization of target_idx to minimize multiplicative depth.
        Returns (idx1, idx2) such that idx1 | idx2 = target_idx and both are in basis.
        """
        if target_idx == 0:
            return None
            
        # Strategy 1: Try balanced factorization (similar depths)
        target_weight = bin(target_idx).count('1')
        best_balance = float('inf')
        best_factors = None
        
        # Check all possible factorizations
        for i in range(1, target_idx):
            if (i & target_idx) == i and i in basis:  # i is a subset of target_idx
                rest = target_idx ^ i
                if rest in basis:
                    # Compute balance score (difference in bit counts)
                    balance = abs(bin(i).count('1') - bin(rest).count('1'))
                    if balance < best_balance:
                        best_balance = balance
                        best_factors = (i, rest)
        
        if best_factors:
            return best_factors
        
        # Strategy 2: Use lowest bit factorization if balanced approach fails
        low_bit = target_idx & -target_idx
        rest = target_idx ^ low_bit
        
        if low_bit in basis and rest in basis:
            return (low_bit, rest)
        
        return None




    def _compute_monomial_via_multiplication(
        self, 
        target_idx: int, 
        basis: Dict[int, desilofhe.Ciphertext]
    ) -> desilofhe.Ciphertext:
        """
        Helper function to compute a monomial via multiplication when conjugation is not available.
        Uses the most efficient factorization to minimize multiplicative depth.
        """
        # Special case: if target_idx is 0, return the constant 1
        if target_idx == 0:
            return basis[0]
        
        # Find the best factorization
        # Strategy: use the lowest bit and the rest to minimize depth
        low_bit = target_idx & -target_idx  # Isolate the lowest set bit
        rest = target_idx ^ low_bit
        
        if low_bit not in basis or rest not in basis:
            # If standard factorization doesn't work, try other factorizations
            for i in range(1, target_idx):
                if (i & target_idx) == i and i in basis:  # i is a subset of target_idx
                    rest = target_idx ^ i
                    if rest in basis:
                        low_bit, rest = i, rest
                        break
        
        if low_bit not in basis or rest not in basis:
            raise ValueError(f"Cannot compute monomial {bin(target_idx)}: missing factors")
        
        ct1, ct2 = basis[low_bit], basis[rest]
        
        # Level synchronization
        if ct1.level != ct2.level:
            if ct1.level > ct2.level:
                ct1 = self.engine.level_down(ct1, ct2.level)
            else:
                ct2 = self.engine.level_down(ct2, ct1.level)
        
        # Multiply and relinearize
        prod = self.engine.multiply(ct1, ct2)
        prod = self.engine.relinearize(prod, self.relin_key)
        prod = self.engine.rescale(prod)
        
        return prod



    # ------------------------------------------------------------------
    # 2) Multivariate polynomial evaluation
    # ------------------------------------------------------------------

    def evaluate_multivariate_polynomial(
        self,
        basis: Dict[int, desilofhe.Ciphertext],
        coeffs: np.ndarray,
        slot_coeffs: Optional[np.ndarray] = None,
    ) -> desilofhe.Ciphertext:
        """Evaluate ∑ cᵢ·x^i given its monomial *basis*.

        *slot_coeffs* allows different coefficients per SIMD slot (Fix #3).
        """
        acc: Optional[desilofhe.Ciphertext] = None

        for idx, mon_ct in basis.items():
            if idx >= len(coeffs):
                continue

            if slot_coeffs is None:
                c = coeffs[idx]
                if abs(c) < 1e-12:
                    continue
                pt = self.engine.encode(np.full(self.slot_count, c, dtype=complex))
            else:
                vec = slot_coeffs[:, idx]
                if np.all(np.abs(vec) < 1e-12):
                    continue
                pt = self.engine.encode(vec)

            term = self.engine.multiply(mon_ct, pt)

            if acc is None:
                acc = term
            else:
                # Level alignment on the fly.
                if acc.level != term.level:
                    if acc.level > term.level:
                        acc = self.engine.level_down(acc, term.level)
                    else:
                        term = self.engine.level_down(term, acc.level)
                acc = self.engine.add(acc, term)

        assert acc is not None, "Polynomial evaluation produced no non‑zero terms"
        return acc

    # ------------------------------------------------------------------
    # 3) Bit extraction (4‑to‑1 LUT)
    # ------------------------------------------------------------------

    def extract_bit_from_nibble(
        self,
        nibble_ct: desilofhe.Ciphertext,
        bit_idx: int,
        bit_extract_coeffs: List[np.ndarray],
    ) -> desilofhe.Ciphertext:
        coeffs = bit_extract_coeffs[bit_idx][:4]  # degree ≤ 3
        lvl = nibble_ct.level
        one = self._get_one_ct(lvl)

        # Pre‑compute x, x², x³.
        x1 = nibble_ct
        x2 = self.engine.multiply(x1, x1)
        x2 = self.engine.relinearize(x2, self.relin_key)
        x2 = self.engine.rescale(x2)
        x3 = self.engine.multiply(x2, x1)
        x3 = self.engine.relinearize(x3, self.relin_key)
        x3 = self.engine.rescale(x3)
        powers = [one, x1, x2, x3]

        acc: Optional[desilofhe.Ciphertext] = None
        for d, c in enumerate(coeffs):
            if abs(c) < 1e-12:
                continue
            pt = self.engine.encode(np.full(self.slot_count, c, dtype=complex))
            term = self.engine.multiply(powers[d], pt)
            acc = term if acc is None else self.engine.add(acc, term)
        assert acc is not None
        return acc

    # ------------------------------------------------------------------
    # 4) GF(2^8) multiplication LUT evaluation (Fix #5)
    # ------------------------------------------------------------------

    def evaluate_gf28_multiplication(
        self, ct: desilofhe.Ciphertext, mult_coeffs: np.ndarray
    ) -> desilofhe.Ciphertext:
        lvl = ct.level
        one = self._get_one_ct(lvl)
        powers: List[Optional[desilofhe.Ciphertext]] = [one, ct] + [None] * 14

        # Lazily compute only the powers whose coefficient ≠ 0.
        for i in range(2, 16):
            if abs(mult_coeffs[i]) < 1e-12:
                continue
            if i % 2 == 0 and powers[i // 2] is not None:
                base = powers[i // 2]
                p = self.engine.multiply(base, base)
            else:
                # Ensure previous power available.
                prev = powers[i - 1]
                if prev is None:
                    prev = self.evaluate_gf28_multiplication(ct, np.eye(1, 16, i - 1)[0])  # recursion for safety
                p = self.engine.multiply(ct, prev)
            p = self.engine.relinearize(p, self.relin_key)
            p = self.engine.rescale(p)
            powers[i] = p

        acc: Optional[desilofhe.Ciphertext] = None
        for i, c in enumerate(mult_coeffs[:16]):
            if abs(c) < 1e-12:
                continue
            pt = self.engine.encode(np.full(self.slot_count, c, dtype=complex))
            term = self.engine.multiply(powers[i], pt)  # type: ignore[arg-type]
            acc = term if acc is None else self.engine.add(acc, term)

        assert acc is not None
        return acc

    # ------------------------------------------------------------------
    # 5) Internal helpers
    # ------------------------------------------------------------------

    def _conjugate_ciphertext(self, ct: desilofhe.Ciphertext) -> desilofhe.Ciphertext:
        """Apply complex conjugation via Galois automorphism (if available)."""
        if self.conj_key is None:
            self._log("Conjugation key missing – returning input unmodified")
            return ct
        try:
            return self.engine.apply_galois(ct, -1, self.conj_key)
        except Exception as exc:  # pragma: no cover – backend‑specific
            self._log(f"Conjugation failed ({exc}) – returning input unmodified")
            return ct
