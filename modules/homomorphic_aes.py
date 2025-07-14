"""
Homomorphic AES Operations Module

This module implements the homomorphic evaluation of AES operations including
SubBytes, ShiftRows, MixColumns, and AddRoundKey using the paper's methodology.
"""

import numpy as np
import desilofhe
from typing import Tuple, List, Dict, Optional
from .lut_evaluation import LUTEvaluator
from .fhe_operations import FHEOperations


class HomomorphicAES:
    """Implements homomorphic AES operations using LUT evaluations."""
    
# ────────────────────────────────────────────────────────────────────
# 1)  __init__  ──  Linear-layer LUT 을 한 번만 미리 계산‧캐시
# ────────────────────────────────────────────────────────────────────
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

        # ▶︎ linear-layer LUT 은 무겁게 계산되므로 한 번만
        self.linear_luts: Dict[Tuple[int, int], np.ndarray] = (
            self.coeffs.compute_linear_layer_luts()
            if hasattr(self.coeffs, "compute_linear_layer_luts")
            else {}
        )

        # 빠른 접근을 위해 미리 꺼내 둠
        self.rot_keys = self.keys.get("gap_rotation_keys", {})

    def _log(self, msg):                             # 그대로 유지
        if self.verbose:
            print(f"[HOM-AES] {msg}")


    # ────────────────────────────────────────────────────────────────────
    # 2)  homomorphic_sub_bytes  ──  bootstrap 호출 시그니처 & 레벨매칭 개선
    # ────────────────────────────────────────────────────────────────────
    def homomorphic_sub_bytes(self, left_ct, right_ct):
        self._log("SubBytes – shared power basis")
        left_out, right_out = [], []

        for idx in range(16):
            start = idx * self.slot_gap
            mask = np.zeros(self.slot_count, dtype=complex)
            for b in range(self.batch_size):
                s = start + b
                if s < self.slot_count:
                    mask[s] = 1.0
            mask_pt = self.engine.encode(mask)

            lnib = self.engine.multiply(left_ct,  mask_pt)
            rnib = self.engine.multiply(right_ct, mask_pt)

            bits = [self.lut_eval.extract_bit_from_nibble(rnib, i, self.coeffs.bit_extract_coeffs) for i in range(4)] + \
                [self.lut_eval.extract_bit_from_nibble(lnib, i, self.coeffs.bit_extract_coeffs) for i in range(4)]

            monomials = self.lut_eval.compute_monomial_basis_optimized(bits)
            upper = self.lut_eval.evaluate_multivariate_polynomial(monomials, self.coeffs.sbox_upper_coeffs)
            lower = self.lut_eval.evaluate_multivariate_polynomial(monomials, self.coeffs.sbox_lower_coeffs)

            if start in self.rot_keys:
                upper = self.engine.rotate(upper, self.rot_keys[start])
                lower = self.engine.rotate(lower, self.rot_keys[start])

            left_out.append(upper)
            right_out.append(lower)

        # ─ combine 16 bytes
        final_left, final_right = left_out[0], right_out[0]
        for i in range(1, 16):
            if final_left.level != left_out[i].level:
                if final_left.level > left_out[i].level:
                    final_left = self.engine.level_down(final_left, left_out[i].level)
                else:
                    left_out[i] = self.engine.level_down(left_out[i], final_left.level)
            if final_right.level != right_out[i].level:
                if final_right.level > right_out[i].level:
                    final_right = self.engine.level_down(final_right, right_out[i].level)
                else:
                    right_out[i] = self.engine.level_down(right_out[i], final_right.level)

            final_left  = self.engine.add(final_left,  left_out[i])
            final_right = self.engine.add(final_right, right_out[i])

        final_left  = self.fhe_ops.apply_noise_reduction(final_left)
        final_right = self.fhe_ops.apply_noise_reduction(final_right)

        # ▶︎ 바뀐 시그니처: bootstrap_if_needed(ct)  (key를 내부에서 사용)
        final_left  = self.fhe_ops.bootstrap_if_needed(final_left)
        final_right = self.fhe_ops.bootstrap_if_needed(final_right)

        return final_left, final_right


    # ────────────────────────────────────────────────────────────────────
    # 3)  homomorphic_linear_layer  ──  캐시된 LUT 사용 & 안전한 레벨매칭
    # ────────────────────────────────────────────────────────────────────
    def homomorphic_linear_layer(self, left_ct, right_ct):
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


    # ────────────────────────────────────────────────────────────────────
    # 4)  homomorphic_add_round_key  ──  간단화 + 레벨매칭
    # ────────────────────────────────────────────────────────────────────
    def homomorphic_add_round_key(
        self,
        left_ct,  right_ct,
        rk_left_ct, rk_right_ct
    ):
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

            sl = self.engine.multiply(left_ct,     mp)
            sr = self.engine.multiply(right_ct,    mp)
            kl = self.engine.multiply(rk_left_ct,  mp)
            kr = self.engine.multiply(rk_right_ct, mp)

            xl = self.engine.add(sl, kl)   # a ⊕ b  (mod-2 → 근사치로 덧셈)
            xr = self.engine.add(sr, kr)

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