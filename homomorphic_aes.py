"""
Homomorphic AES Operations Module - Fixed Version

This module implements the homomorphic evaluation of AES operations including
SubBytes, ShiftRows, MixColumns, and AddRoundKey using the paper's methodology.

FIXES:
1. Use XOR LUT for AddRoundKey instead of simple addition
2. Force bootstrapping after every LUT evaluation
3. Apply noise reduction after each round
"""

import numpy as np
import desilofhe
from typing import Tuple, List, Dict, Optional
try:
    from .lut_evaluation import LUTEvaluator
    from .fhe_operations import FHEOperations
except ImportError:
    from modules.lut_evaluation import LUTEvaluator
    from modules.fhe_operations import FHEOperations


class HomomorphicAES:
    """Implements homomorphic AES operations using LUT evaluations."""
    
    def __init__(self, engine: desilofhe.Engine, keys: Dict, coeffs: object,
                lut_evaluator: LUTEvaluator, fhe_ops: FHEOperations,
                slot_gap: int, batch_size: int, verbose: bool = False):
        self.engine       = engine
        self.keys         = keys
        self.coeffs       = coeffs
        self.lut_eval     = lut_evaluator
        self.fhe_ops      = fhe_ops
        self.slot_gap     = slot_gap
        self.batch_size   = batch_size
        self.verbose      = verbose
        self.slot_count   = engine.slot_count

        # Pre-compute linear-layer LUTs once
        self.linear_luts: Dict[Tuple[int, int], np.ndarray] = (
            self.coeffs.compute_linear_layer_luts()
            if hasattr(self.coeffs, "compute_linear_layer_luts")
            else {}
        )

        # Quick access to rotation keys
        self.rot_keys = self.keys.get("gap_rotation_keys", {})
        
        # Cache relinearization key
        self.relin_key = self.keys.get("relin_key")

    def _log(self, msg):
        if self.verbose:
            print(f"[HOM-AES] {msg}")
    
    def _get_one_ct(self, level: int) -> desilofhe.Ciphertext:
        """Get encrypted constant 1 at specified level."""
        # Delegate to LUT evaluator which has this functionality
        return self.lut_eval._get_one_ct(level)

    def homomorphic_sub_bytes(
        self,
        state: List[desilofhe.Ciphertext],
    ) -> List[desilofhe.Ciphertext]:
        """
        AES SubBytes – 멀티버리어블 S-box 다항식으로 16바이트 상태를 변환합니다.
        입력·출력 모두 16 개의 8-비트 Ciphertext.
        """
        out_state: List[desilofhe.Ciphertext] = []

        # ── 상수 Plaintext 캐시 ───────────────────────────────────────────
        const_cache: Dict[int, desilofhe.Plaintext] = {}
        def _const_pt(val: int) -> desilofhe.Plaintext:
            if val not in const_cache:
                const_cache[val] = self.engine.encode(
                    np.full(self.slot_count, val, dtype=np.complex128)
                )
            return const_cache[val]

        for byte_ct in state:
            # 1. 8→4 LUT로 두 니블 추출
            rnib = self.lut_eval.apply_univariate_lut(
                byte_ct, self.coeffs.sbox_lower_coeffs
            )               # 하위 4비트
            lnib = self.lut_eval.apply_univariate_lut(
                byte_ct, self.coeffs.sbox_upper_coeffs
            )               # 상위 4비트

            # 2. 파워-베이시스 생성 (x⁰ … x¹⁵)
            r_basis = self.lut_eval.compute_power_basis(rnib, max_deg=15)
            l_basis = self.lut_eval.compute_power_basis(lnib, max_deg=15)

            # 3. 2-변수 단항 X^i · Y^j 생성
            mv_basis: Dict[Tuple[int, int], desilofhe.Ciphertext] = {(0, 0): _const_pt(1)}
            for i, x_ct in r_basis.items():
                for j, y_ct in l_basis.items():
                    if i == 0 and j == 0:
                        continue
                    mv_basis[(i, j)] = self.engine.multiply(x_ct, y_ct)

            # 4. 상위·하위 니블 각각 계산
            upper_ct = self.lut_eval.evaluate_multivariate_polynomial_tensor(
                mv_basis, self.coeffs.sbox_upper_mv_coeffs
            )
            lower_ct = self.lut_eval.evaluate_multivariate_polynomial_tensor(
                mv_basis, self.coeffs.sbox_lower_mv_coeffs
            )

            # 5. (upper << 4) + lower  ── plain 16(L=4) 곱 후 더하기
            upper_shifted = self.engine.multiply_plain(upper_ct, _const_pt(16))
            byte_out = self.engine.add(upper_shifted, lower_ct)

            out_state.append(byte_out)

        return out_state


    def homomorphic_linear_layer(self, left_ct, right_ct):
        """
        Apply unified ShiftRows + MixColumns transformation.
        """
        self._log("Linear layer – unified LUT")
        outL, outR = [None]*16, [None]*16

        for inp in range(16):
            start = inp * self.slot_gap
            mask = np.zeros(self.slot_count, dtype=complex)
            for b in range(self.batch_size):
                s = start + b
                if s < self.slot_count:
                    mask[s] = 1.0
            mask_pt = self.engine.encode(mask)
            lnib = self.engine.multiply(left_ct,  mask_pt)
            rnib = self.engine.multiply(right_ct, mask_pt)

            for out in range(16):
                coeffs = self.linear_luts.get((inp, out))
                if coeffs is None:
                    continue
                tl = self.lut_eval.evaluate_gf28_multiplication(lnib, coeffs)
                tr = self.lut_eval.evaluate_gf28_multiplication(rnib, coeffs)

                # FIX #2: Force bootstrapping after LUT evaluation
                tl = self.fhe_ops.bootstrap_if_needed(tl)
                tr = self.fhe_ops.bootstrap_if_needed(tr)

                rot = (out - inp) * self.slot_gap
                if rot in self.rot_keys:
                    tl = self.engine.rotate(tl, self.rot_keys[rot])
                    tr = self.engine.rotate(tr, self.rot_keys[rot])

                outL[out] = tl if outL[out] is None else self.engine.add(outL[out], tl)
                outR[out] = tr if outR[out] is None else self.engine.add(outR[out], tr)

        final_left, final_right = outL[0], outR[0]
        for i in range(1, 16):
            if outL[i] is None:
                continue
            # Level matching
            if final_left.level != outL[i].level:
                if final_left.level > outL[i].level:
                    final_left = self.engine.level_down(final_left, outL[i].level)
                else:
                    outL[i] = self.engine.level_down(outL[i], final_left.level)
            if final_right.level != outR[i].level:
                if final_right.level > outR[i].level:
                    final_right = self.engine.level_down(final_right, outR[i].level)
                else:
                    outR[i] = self.engine.level_down(outR[i], final_right.level)

            final_left  = self.engine.add(final_left,  outL[i])
            final_right = self.engine.add(final_right, outR[i])

        final_left  = self.fhe_ops.apply_noise_reduction(final_left)
        final_right = self.fhe_ops.apply_noise_reduction(final_right)
        return final_left, final_right

    def homomorphic_add_round_key(
        self,
        left_ct, right_ct,
        rk_left_ct, rk_right_ct
    ):
        """
        FIX #3: Apply AddRoundKey using XOR LUT instead of simple addition.
        """
        self._log("AddRoundKey – XOR LUT")

        resL, resR = [], []
        for idx in range(16):
            start = idx * self.slot_gap
            mask = np.zeros(self.slot_count, dtype=complex)
            for b in range(self.batch_size):
                s = start + b
                if s < self.slot_count:
                    mask[s] = 1.0
            mp = self.engine.encode(mask)

            # Extract nibbles
            sl = self.engine.multiply(left_ct,     mp)
            sr = self.engine.multiply(right_ct,    mp)
            kl = self.engine.multiply(rk_left_ct,  mp)
            kr = self.engine.multiply(rk_right_ct, mp)

            # Evaluate XOR using 4-bit XOR LUT
            xl = self._evaluate_xor_lut(sl, kl)
            xr = self._evaluate_xor_lut(sr, kr)

            # FIX #2: Force bootstrapping after LUT evaluation
            xl = self.fhe_ops.bootstrap_if_needed(xl)
            xr = self.fhe_ops.bootstrap_if_needed(xr)

            if start in self.rot_keys:
                xl = self.engine.rotate(xl, self.rot_keys[start])
                xr = self.engine.rotate(xr, self.rot_keys[start])

            resL.append(xl)
            resR.append(xr)

        left  = resL[0]
        right = resR[0]
        for i in range(1, 16):
            left  = self.engine.add(left,  resL[i])
            right = self.engine.add(right, resR[i])

        left  = self.fhe_ops.apply_noise_reduction(left)
        right = self.fhe_ops.apply_noise_reduction(right)
        return left, right
    def _evaluate_xor_lut(self, a_ct: desilofhe.Ciphertext, b_ct: desilofhe.Ciphertext) -> desilofhe.Ciphertext:
        """
        Evaluate 4-bit XOR using modular-reduced polynomial coefficients.
        Production-ready 최적화 버전
        """
        # 두 4-bit 입력을 8-bit으로 결합
        if not hasattr(self, '_sixteen_pt'):
            self._sixteen_pt = self.engine.encode(np.full(self.slot_count, 16.0))
        
        a_shifted = self.engine.multiply_plain(a_ct, self._sixteen_pt)
        combined = self.engine.add(a_shifted, b_ct)
        
        # XOR 계수 초기화 (한 번만)
        if not hasattr(self, '_xor_reduced_data'):
            self._precompute_xor_data(combined.scale)  # 현재 스케일 전달
        
        # Sparse polynomial evaluation
        result = self._evaluate_sparse_polynomial_final(combined)
        
        if self.verbose:
            self._log(f"XOR result: level={result.level}, scale={result.scale:.2e}")
        
        return result

    def _precompute_xor_data(self, reference_scale: float):
        """XOR 관련 모든 사전계산"""
        # 1. 차수 환원 (mod 16)
        reduced = np.zeros(16, dtype=complex)
        for i in range(256):
            if abs(self.coeffs.xor_4bit_coeffs[i]) > 1e-12:
                reduced[i % 16] += self.coeffs.xor_4bit_coeffs[i]
        
        # 2. 데이터 구조 초기화
        self._xor_reduced_data = {
            'sparse_coeffs': {},
            'plaintexts': {},
            'conjugate_pairs': {},
            'constant_term': None,
            'scale': reference_scale  # 스케일 저장
        }
        
        # 3. Non-zero 계수 처리 및 plaintext 생성
        for deg, coeff in enumerate(reduced):
            if abs(coeff) > 1e-12:
                if deg == 0:
                    # 상수항은 별도로 저장 (나중에 add_plain)
                    self._xor_reduced_data['constant_term'] = coeff
                    self._xor_reduced_data['constant_pt'] = self.engine.encode(
                        np.full(self.slot_count, coeff)
                    )
                else:
                    self._xor_reduced_data['sparse_coeffs'][deg] = coeff
                    # Plaintext를 참조 스케일로 생성
                    pt = self.engine.encode(
                        np.full(self.slot_count, coeff),
                        scale=reference_scale  # 스케일 맞춤
                    )
                    self._xor_reduced_data['plaintexts'][deg] = pt
                
                # Conjugate 관계 확인 (16-deg), 자기 자신 제외
                conj_deg = 16 - deg
                if 0 < conj_deg < 16 and conj_deg != deg:
                    self._xor_reduced_data['conjugate_pairs'][deg] = conj_deg
        
        # 4. 개발 단계에서만 검증
        if self.verbose:
            self._verify_xor_reduction()
        
        self._log(f"XOR initialized: {len(self._xor_reduced_data['sparse_coeffs'])} terms, "
                f"constant={'yes' if self._xor_reduced_data['constant_term'] else 'no'}")

    def _evaluate_sparse_polynomial_final(self, x_ct: desilofhe.Ciphertext) -> desilofhe.Ciphertext:
        """최종 최적화된 sparse polynomial 평가"""
        data = self._xor_reduced_data
        sparse_coeffs = data['sparse_coeffs']
        
        if not sparse_coeffs and not data['constant_term']:
            # 빈 다항식
            return self.engine.encrypt(np.zeros(self.slot_count), self.pk)
        
        # 필요한 monomial 계산 (상수항 제외)
        needed_degrees = set(sparse_coeffs.keys())
        monomials = self._compute_sparse_monomials_final(x_ct, needed_degrees)
        
        # 다항식 평가 (첫 항으로 초기화)
        result = None
        for deg, coeff in sparse_coeffs.items():
            term = self.engine.multiply_plain(monomials[deg], data['plaintexts'][deg])
            
            if result is None:
                result = term
            else:
                # 레벨만 맞추고 덧셈 (스케일은 이미 동일)
                if result.level > term.level:
                    result = self.engine.level_down(result, term.level)
                elif term.level > result.level:
                    term = self.engine.level_down(term, result.level)
                result = self.engine.add(result, term)
        
        # 상수항이 있으면 마지막에 한 번만 추가
        if data['constant_term']:
            if result is None:
                # 상수항만 있는 경우
                result = self.engine.encrypt(
                    np.full(self.slot_count, data['constant_term']), 
                    self.pk
                )
                # 입력과 레벨 맞춤
                if result.level > x_ct.level:
                    result = self.engine.level_down(result, x_ct.level)
            else:
                result = self.engine.add_plain(result, data['constant_pt'])
        
        return result

    def _compute_sparse_monomials_final(self, x_ct: desilofhe.Ciphertext, needed_degrees: set) -> dict:
        """최종 최적화된 monomial 계산"""
        monomials = {1: x_ct}
        data = self._xor_reduced_data
        
        # deg=8 같은 self-conjugate 처리
        self_conjugates = {deg for deg in needed_degrees if deg == 16 - deg}
        
        # Conjugate로 얻을 수 있는 차수 분리
        direct_needed = needed_degrees.copy()
        for deg in needed_degrees:
            if deg in data['conjugate_pairs']:
                conj_deg = data['conjugate_pairs'][deg]
                # 더 낮은 차수만 직접 계산 (self-conjugate 제외)
                if deg > conj_deg and conj_deg in needed_degrees and deg not in self_conjugates:
                    direct_needed.discard(deg)
        
        # 로그 정보
        if self.verbose:
            self._log(f"Computing {len(direct_needed)} monomials directly, "
                    f"{len(needed_degrees) - len(direct_needed)} via conjugation")
        
        # Binary exponentiation
        computed = {1}
        max_deg = max(direct_needed) if direct_needed else 0
        
        # 2의 거듭제곱만 계산
        power = 2
        while power <= max_deg:
            if any(deg & power for deg in direct_needed if deg >= power):
                prev = monomials[power // 2]
                squared = self.engine.multiply(prev, prev)
                squared = self.engine.relinearize(squared, self.relin_key)
                squared = self.engine.rescale(squared)  # 한 번만 rescale
                monomials[power] = squared
                computed.add(power)
                
                if self.verbose:
                    self._log(f"  x^{power}: level={squared.level}, scale={squared.scale:.2e}")
            power *= 2
        
        # 나머지 차수는 최소 곱셈으로 구성
        for deg in sorted(direct_needed):
            if deg in computed:
                continue
            
            # Binary decomposition
            factors = []
            temp = deg
            bit = 1
            while temp > 0:
                if temp & 1 and bit in computed:
                    factors.append(bit)
                temp >>= 1
                bit <<= 1
            
            if factors:
                result = monomials[factors[0]]
                for i in range(1, len(factors)):
                    factor = monomials[factors[i]]
                    # 레벨 정렬 후 곱셈
                    if result.level > factor.level:
                        result = self.engine.level_down(result, factor.level)
                    elif factor.level > result.level:
                        factor = self.engine.level_down(factor, result.level)
                    
                    result = self.engine.multiply(result, factor)
                    result = self.engine.relinearize(result, self.relin_key)
                    result = self.engine.rescale(result)
                
                monomials[deg] = result
                computed.add(deg)
        
        # Conjugation으로 나머지 생성
        if self.conj_key:
            for deg in needed_degrees:
                if deg not in monomials and deg in data['conjugate_pairs']:
                    conj_deg = data['conjugate_pairs'][deg]
                    if conj_deg in monomials:
                        monomials[deg] = self.engine.conjugate(monomials[conj_deg], self.conj_key)
                        if self.verbose:
                            self._log(f"  x^{deg} via conjugation from x^{conj_deg}")
        
        return monomials

    def _verify_xor_reduction(self):
        """개발 단계용 XOR 검증 (verbose 모드에서만)"""
        errors = []
        for a in range(16):
            for b in range(16):
                expected = a ^ b
                
                # 환원된 다항식 평가
                input_val = a * 16 + b
                zeta = np.exp(2j * np.pi / 16)
                x = zeta ** (input_val % 16)
                
                result = 0
                if self._xor_reduced_data['constant_term']:
                    result = self._xor_reduced_data['constant_term']
                
                for deg, coeff in self._xor_reduced_data['sparse_coeffs'].items():
                    result += coeff * (x ** deg)
                
                # 디코딩
                angle = np.angle(result)
                if angle < 0:
                    angle += 2 * np.pi
                decoded = int(round(angle * 16 / (2 * np.pi))) % 16
                
                if decoded != expected:
                    errors.append(f"XOR({a},{b})={expected} but got {decoded}")
        
        if errors:
            raise ValueError(f"XOR reduction failed:\n" + "\n".join(errors))
        else:
            self._log("XOR reduction verified successfully")

    def _square_and_rescale(self, ct: desilofhe.Ciphertext) -> desilofhe.Ciphertext:
        """제곱 후 일관된 rescale 적용"""
        result = self.engine.multiply(ct, ct)
        result = self.engine.relinearize(result, self.relin_key)
        # 항상 rescale (일관성 유지)
        result = self.engine.rescale(result)
        return result

    def _multiply_and_rescale(self, ct1: desilofhe.Ciphertext, ct2: desilofhe.Ciphertext) -> desilofhe.Ciphertext:
        """곱셈 후 일관된 rescale 적용"""
        # 먼저 스케일 확인
        ct1, ct2 = self._align_for_multiplication(ct1, ct2)
        
        result = self.engine.multiply(ct1, ct2)
        result = self.engine.relinearize(result, self.relin_key)
        result = self.engine.rescale(result)
        return result

    def _align_for_multiplication(self, ct1: desilofhe.Ciphertext, ct2: desilofhe.Ciphertext):
        """곱셈을 위한 스케일/레벨 정렬"""
        # 레벨 맞추기
        if ct1.level > ct2.level:
            ct1 = self.engine.level_down(ct1, ct2.level)
        elif ct2.level > ct1.level:
            ct2 = self.engine.level_down(ct2, ct1.level)
        
        # 스케일 확인 (2의 거듭제곱으로 정규화)
        target_scale = 2 ** round(np.log2(ct1.scale))
        if abs(ct1.scale - target_scale) > 1:
            ct1 = self.engine.rescale_to(ct1, target_scale)
        if abs(ct2.scale - target_scale) > 1:
            ct2 = self.engine.rescale_to(ct2, target_scale)
        
        return ct1, ct2

    def _align_for_addition(self, ct1: desilofhe.Ciphertext, ct2: desilofhe.Ciphertext, target_scale: float):
        """덧셈을 위한 스케일/레벨 정렬"""
        # 레벨 맞추기
        min_level = min(ct1.level, ct2.level)
        if ct1.level > min_level:
            ct1 = self.engine.level_down(ct1, min_level)
        if ct2.level > min_level:
            ct2 = self.engine.level_down(ct2, min_level)
        
        # 스케일 맞추기
        if abs(ct1.scale - target_scale) > target_scale * 0.01:
            ct1 = self.engine.rescale_to(ct1, target_scale)
        if abs(ct2.scale - target_scale) > target_scale * 0.01:
            ct2 = self.engine.rescale_to(ct2, target_scale)
        
        return ct1, ct2

    def _match_scale_and_level(self, ct: desilofhe.Ciphertext, reference: desilofhe.Ciphertext):
        """참조 ciphertext와 스케일/레벨 일치"""
        if ct.level > reference.level:
            ct = self.engine.level_down(ct, reference.level)
        if abs(ct.scale - reference.scale) > reference.scale * 0.01:
            ct = self.engine.rescale_to(ct, reference.scale)
        return ct

    def _find_optimal_decomposition(self, target: int, available: set) -> list:
        """최소 곱셈 횟수로 target을 만드는 분해 찾기"""
        # Greedy approach: 가장 큰 available부터 사용
        factors = []
        remaining = target
        
        for power in sorted(available, reverse=True):
            while remaining >= power and (remaining & power):
                factors.append(power)
                remaining ^= power
        
        return factors if remaining == 0 else None
    
    def homomorphic_shift_rows_only(self, left_ct: desilofhe.Ciphertext,
                                   right_ct: desilofhe.Ciphertext) -> Tuple[desilofhe.Ciphertext, desilofhe.Ciphertext]:
        """Apply ShiftRows only for the final round."""
        self._log("Applying ShiftRows only (final round)...")
        
        shift_amounts = [0, -1, -2, -3]  # Shift amounts for each row
        
        result_left = None
        result_right = None
        
        for row in range(4):
            if shift_amounts[row] == 0:
                continue  # No shift for row 0
            
            # Bytes in this row
            row_bytes = [row + col*4 for col in range(4)]
            
            # Create mask for this row
            mask = np.zeros(self.slot_count, dtype=complex)
            for byte_idx in row_bytes:
                for block_idx in range(self.batch_size):
                    slot_idx = byte_idx * self.slot_gap + block_idx
                    if slot_idx < self.slot_count:
                        mask[slot_idx] = 1.0
            
            mask_pt = self.engine.encode(mask)
            
            # Extract row
            row_left = self.engine.multiply(left_ct, mask_pt)
            row_right = self.engine.multiply(right_ct, mask_pt)
            
            # Apply rotation
            rotation_amount = shift_amounts[row] * self.slot_gap
            rotation_key = self.keys.get('gap_rotation_keys', {}).get(rotation_amount)
            if rotation_key:
                row_left = self.engine.rotate(row_left, rotation_key)
                row_right = self.engine.rotate(row_right, rotation_key)
            
            # Add to result
            if result_left is None:
                result_left = row_left
                result_right = row_right
            else:
                result_left = self.engine.add(result_left, row_left)
                result_right = self.engine.add(result_right, row_right)
        
        # Add unshifted row 0
        row0_mask = np.zeros(self.slot_count, dtype=complex)
        for col in range(4):
            byte_idx = col * 4  # Row 0 bytes
            for block_idx in range(self.batch_size):
                slot_idx = byte_idx * self.slot_gap + block_idx
                if slot_idx < self.slot_count:
                    row0_mask[slot_idx] = 1.0
        
        row0_mask_pt = self.engine.encode(row0_mask)
        row0_left = self.engine.multiply(left_ct, row0_mask_pt)
        row0_right = self.engine.multiply(right_ct, row0_mask_pt)
        
        result_left = self.engine.add(result_left, row0_left)
        result_right = self.engine.add(result_right, row0_right)
        
        return result_left, result_right
    def homomorphic_sub_bytes_optimized(self, left_ct, right_ct):
        """
        Optimized S-box using shared monomial basis and parallel evaluation.
        """
        self._log("SubBytes – optimized with shared basis")
        
        # Step 1: Extract all nibbles at once
        all_left_nibbles = []
        all_right_nibbles = []
        
        for idx in range(16):
            start = idx * self.slot_gap
            mask = np.zeros(self.slot_count, dtype=complex)
            for b in range(self.batch_size):
                s = start + b
                if s < self.slot_count:
                    mask[s] = 1.0
            mask_pt = self.engine.encode(mask)
            
            all_left_nibbles.append(self.engine.multiply(left_ct, mask_pt))
            all_right_nibbles.append(self.engine.multiply(right_ct, mask_pt))
        
        # Step 2: Compute shared monomial basis once
        # Extract bits for all 16 bytes
        all_bits = []
        for idx in range(16):
            right_bits = [self.lut_eval.extract_bit_from_nibble(
                all_right_nibbles[idx], i, self.coeffs.bit_extract_coeffs
            ) for i in range(4)]
            left_bits = [self.lut_eval.extract_bit_from_nibble(
                all_left_nibbles[idx], i, self.coeffs.bit_extract_coeffs
            ) for i in range(4)]
            all_bits.extend(right_bits + left_bits)
        
        # Compute basis for all 128 bits (16 bytes * 8 bits)
        shared_basis = self.lut_eval.compute_monomial_basis_optimized(all_bits)
        
        # Step 3: Evaluate both upper and lower nibbles simultaneously
        upper_results = []
        lower_results = []
        
        for byte_idx in range(16):
            # Select relevant subset of basis for this byte
            byte_basis = self._extract_byte_basis(shared_basis, byte_idx)
            
            # Evaluate both outputs at once
            results = self.lut_eval.evaluate_multiple_luts_simultaneously(
                byte_basis,
                [self.coeffs.sbox_upper_coeffs, self.coeffs.sbox_lower_coeffs]
            )
            
            upper, lower = results[0], results[1]
            
            # Bootstrapping
            if self.fhe_ops.enable_bootstrapping:
                upper = self.fhe_ops.bootstrap_if_needed(upper)
                lower = self.fhe_ops.bootstrap_if_needed(lower)
            
            # Rotation
            if byte_idx * self.slot_gap in self.rot_keys:
                rot_key = self.rot_keys[byte_idx * self.slot_gap]
                upper = self.engine.rotate(upper, rot_key)
                lower = self.engine.rotate(lower, rot_key)
            
            upper_results.append(upper)
            lower_results.append(lower)
        
        # Combine results
        return self._combine_results(upper_results, lower_results)