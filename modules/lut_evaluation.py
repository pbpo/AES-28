"""
LUT Evaluation Module - Fixed Version

This module handles efficient LUT evaluation using pre-computed monomial bases.

FIXES:
1. Add support for multivariate polynomial evaluation (α > 1)
2. Add slot-wise LUT evaluation for different LUTs per slot
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
import desilofhe  # type: ignore – external homomorphic‑encryption back‑end


class LUTEvaluator:
    """Handles efficient LUT evaluation using pre‑computed monomial bases."""

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

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[LUT] {msg}")

    def _get_one_ct(self, level: int) -> desilofhe.Ciphertext:
        """Return an encrypted constant 1 matched to *level* (with caching)."""
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
        self._log("Warning: requested higher level than cached – returning best effort")
        return base_ct

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

    def compute_multivariate_monomial_basis(
        self, input_groups: List[List[desilofhe.Ciphertext]]
    ) -> Dict[Tuple[int, ...], desilofhe.Ciphertext]:
        """
        FIX #6: Compute monomial basis for multivariate polynomials.
        
        Args:
            input_groups: List of variable groups, e.g., [[x0, x1], [y0, y1]] for 2 2-bit variables
        
        Returns:
            Dictionary mapping (degree_x, degree_y, ...) tuples to ciphertext monomials
        """
        self._log(f"Computing multivariate monomial basis for {len(input_groups)} variables")
        
        # First compute univariate bases for each variable
        univariate_bases = []
        for i, var_bits in enumerate(input_groups):
            self._log(f"Computing basis for variable {i} with {len(var_bits)} bits")
            basis = self.compute_monomial_basis_optimized(var_bits)
            univariate_bases.append(basis)
        
        # Now compute cross-products
        multivariate_basis = {}
        
        # Get all possible degree combinations
        max_degrees = [len(bits) for bits in input_groups]
        
        # Iterate through all degree combinations
        for degrees in itertools.product(*[range(2**d) for d in max_degrees]):
            # Get the monomial for each variable
            monomials = []
            for var_idx, degree in enumerate(degrees):
                if degree in univariate_bases[var_idx]:
                    monomials.append(univariate_bases[var_idx][degree])
                else:
                    # Skip this combination if any monomial is missing
                    break
            else:
                # All monomials found, compute product
                result = monomials[0]
                for mon in monomials[1:]:
                    # Level sync
                    if result.level != mon.level:
                        if result.level > mon.level:
                            result = self.engine.level_down(result, mon.level)
                        else:
                            mon = self.engine.level_down(mon, result.level)
                    
                    result = self.engine.multiply(result, mon)
                    result = self.engine.relinearize(result, self.relin_key)
                    result = self.engine.rescale(result)
                
                multivariate_basis[degrees] = result
        
        self._log(f"Multivariate basis complete: {len(multivariate_basis)} monomials")
        return multivariate_basis

    def evaluate_multivariate_polynomial(
        self,
        basis: Dict[int, desilofhe.Ciphertext],
        coeffs: np.ndarray,
        slot_coeffs: Optional[np.ndarray] = None,
    ) -> desilofhe.Ciphertext:
        """Evaluate ∑ cᵢ·x^i given its monomial *basis*.

        *slot_coeffs* allows different coefficients per SIMD slot (Fix #3).
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

    def evaluate_multivariate_polynomial_tensor(
        self,
        basis: Dict[Tuple[int, ...], desilofhe.Ciphertext],
        coeff_tensor: np.ndarray,
    ) -> desilofhe.Ciphertext:
        """
        다변수 다항식 계수 텐서를 basis(모노미얼→Ciphertext)에 적용해
        Σ c_{α}·X^{α} 값을 계산합니다.

        Parametersself.fhe_ops.bootstrap_if_needed
        ----------
        basis : Dict[Tuple[int,…], Ciphertext]
            (degree₁, degree₂, …) → 대응 모노미얼 Ciphertext
            예: 2변수 4-비트이면 키는 (i,j) with 0≤i,j≤15
        coeff_tensor : np.ndarray[complex]
            계수 텐서. shape는 변수별 최대 차수+1.

        Returns
        -------
        Ciphertext
            평가 결과
        """
        tensor_shape = coeff_tensor.shape
        encode_cache: Dict[complex, desilofhe.Plaintext] = {}

        acc: Optional[desilofhe.Ciphertext] = None

        for degs, mon_ct in basis.items():
            # 1) 텐서 범위 체크 ──────────────
            if any(d >= tensor_shape[i] for i, d in enumerate(degs)):
                continue

            c = coeff_tensor[degs]
            if np.abs(c) < 1e-12:          # 2) 영 계수 스킵
                continue

            # 3) 플레인텍스트 캐싱
            if c not in encode_cache:
                vec = np.full(self.slot_count, c, dtype=np.complex128)
                encode_cache[c] = self.engine.encode(vec)
            pt = encode_cache[c]

            # 4) 모노미얼 × 계수
            try:
                term = self.engine.multiply_plain(mon_ct, pt)   # 지원 시
            except AttributeError:
                term = self.engine.multiply(mon_ct, pt)

            # 5) 누적 합(level 정렬 포함)
            if acc is None:
                acc = term
            else:
                if acc.level > term.level:
                    acc = self.engine.level_down(acc, term.level)
                elif term.level > acc.level:
                    term = self.engine.level_down(term, acc.level)
                acc = self.engine.add(acc, term)

        assert acc is not None, "Polynomial evaluation produced only zero terms"
        return acc


    def extract_bit_from_nibble(
        self,
        nibble_ct: desilofhe.Ciphertext,
        bit_idx: int,
        bit_extract_coeffs: List[np.ndarray],
    ) -> desilofhe.Ciphertext:
        coeffs = bit_extract_coeffs[bit_idx][:4]  # degree ≤ 3
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

    def evaluate_gf28_multiplication(
        self, ct: desilofhe.Ciphertext, mult_coeffs: np.ndarray
    ) -> desilofhe.Ciphertext:
        lvl = ct.level
        one = self._get_one_ct(lvl)
        
        # 레벨 체크
        if lvl <= 5:  # 레벨이 너무 낮으면 조기 종료
            self._log(f"Warning: Low level {lvl}, returning approximation")
            # 낮은 차수 근사 사용
            return self._evaluate_low_degree_approx(ct, mult_coeffs[:4])
        
        powers: List[Optional[desilofhe.Ciphertext]] = [one, ct] + [None] * 14

        # Lazily compute only the powers whose coefficient ≠ 0.
        for i in range(2, min(16, lvl)):  # 레벨에 따라 최대 차수 제한
            if abs(mult_coeffs[i]) < 1e-12:
                continue
                
            # 레벨 체크
            if powers[i//2] is not None and powers[i//2].level <= 1:
                break
                
            if i % 2 == 0 and powers[i // 2] is not None:
                base = powers[i // 2]
                p = self.engine.multiply(base, base)
            else:
                prev = powers[i - 1]
                if prev is None:
                    continue
                p = self.engine.multiply(ct, prev)
                
            # 조건부 relinearize와 rescale
            if p.level > 1:
                p = self.engine.relinearize(p, self.relin_key)
                if p.level > 1:  # rescale 전 레벨 확인
                    p = self.engine.rescale(p)
            powers[i] = p

        acc: Optional[desilofhe.Ciphertext] = None
        for i, c in enumerate(mult_coeffs[:16]):
            if powers[i] is None or abs(c) < 1e-12:
                continue
                
            pt = self.engine.encode(np.full(self.slot_count, c, dtype=complex))
            term = self.engine.multiply(powers[i], pt)
            
            if acc is None:
                acc = term
            else:
                # 레벨 매칭
                if acc.level != term.level:
                    if acc.level > term.level:
                        acc = self.engine.level_down(acc, term.level)
                    else:
                        term = self.engine.level_down(term, acc.level)
                acc = self.engine.add(acc, term)

        assert acc is not None
        return acc

    def _evaluate_low_degree_approx(self, ct, coeffs):
        """저차 다항식 근사"""
        acc = None
        power = self._get_one_ct(ct.level)
        
        for i, c in enumerate(coeffs):
            if abs(c) < 1e-12:
                continue
                
            if i > 0:
                power = self.engine.multiply(power, ct)
                
            pt = self.engine.encode(np.full(self.slot_count, c, dtype=complex))
            term = self.engine.multiply(power, pt)
            
            if acc is None:
                acc = term
            else:
                acc = self.engine.add(acc, term)
        
        return acc if acc is not None else power

    # Helper methods remain the same
    def _find_optimal_factorization(
        self, 
        target_idx: int, 
        basis: Dict[int, desilofhe.Ciphertext]
    ) -> Optional[Tuple[int, int]]:
        """Find the optimal factorization of target_idx to minimize multiplicative depth."""
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
        """Helper function to compute a monomial via multiplication."""
        # Special case: if target_idx is 0, return the constant 1
        if target_idx == 0:
            return basis[0]
        
        # Find the best factorization
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

    def _conjugate_ciphertext(self, ct: desilofhe.Ciphertext) -> desilofhe.Ciphertext:
        """Apply complex conjugation via Galois automorphism (if available)."""
        if self.conj_key is None:
            self._log("Conjugation key missing – returning input unmodified")
            return ct
        try:
            return self.engine.apply_galois(ct, -1, self.conj_key)
        except Exception as exc:
            self._log(f"Conjugation failed ({exc}) – returning input unmodified")
            return ct
    def evaluate_slot_wise_luts(
        self,
        basis: Dict[int, desilofhe.Ciphertext],
        slot_coeffs_list: List[np.ndarray],
        slot_indices: List[int]
    ) -> desilofhe.Ciphertext:
        """
        Evaluate different LUTs for each slot using Hadamard product.
        
        Paper Section 4.2: Enable slot-wise evaluation of distinct LUTs.
        
        Args:
            basis: Precomputed monomial basis
            slot_coeffs_list: List of coefficient arrays, one per slot
            slot_indices: Indices indicating which LUT to use for each slot
        """
        # Prepare slot-wise coefficient matrix
        slot_coeffs = np.zeros((self.slot_count, len(basis)), dtype=complex)
        
        for slot_idx in range(self.slot_count):
            lut_idx = slot_indices[slot_idx % len(slot_indices)]
            if lut_idx < len(slot_coeffs_list):
                slot_coeffs[slot_idx, :] = slot_coeffs_list[lut_idx]
        
        # Evaluate using slot-wise coefficients
        return self.evaluate_multivariate_polynomial(basis, None, slot_coeffs)

    def evaluate_multiple_luts_simultaneously(
        self,
        basis: Dict[int, desilofhe.Ciphertext],
        coeffs_list: List[np.ndarray]
    ) -> List[desilofhe.Ciphertext]:
        """
        Evaluate β different LUTs simultaneously using the same monomial basis.
        
        Paper optimization: Reuse monomial basis for multiple LUT evaluations.
        
        Args:
            basis: Precomputed monomial basis (shared)
            coeffs_list: List of β coefficient arrays
        
        Returns:
            List of β ciphertext results
        """
        results = []
        
        for coeffs in coeffs_list:
            result = self.evaluate_multivariate_polynomial(basis, coeffs)
            results.append(result)
        
        return results
    def compute_power_basis(self, x_ct, max_deg=15):
        basis = {0: self._get_one_ct(x_ct.level), 1: x_ct}
        tmp = x_ct
        for d in range(2, max_deg + 1):
            tmp = self.engine.multiply(tmp, x_ct)
            tmp = self.engine.relinearize(tmp, self.relin_key)
            # 2배수 차수마다만 rescale
            if d % 2 == 0:
                tmp = self.engine.rescale(tmp)
            basis[d] = tmp
        # 마지막에 전체 스케일을 맞춰 주고 반환
        return basis
