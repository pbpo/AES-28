"""
Coefficient Computation Module for FHE-AES - Fixed Version

This module handles all polynomial coefficient pre-computations using FFT-based methods.
Implements Fix #1 from the paper alignment.

FIXES:
1. Add multivariate polynomial support for α > 1 LUTs
2. Add LUT decomposition for αℓ→βℓ transformations
"""

import numpy as np
from scipy.fft import ifft, ifftn
from typing import Dict, List, Tuple


class CoefficientComputer:
    """Handles FFT-based coefficient computation for various LUTs."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._precompute_all_coefficients()
    
    def _log(self, message: str):
        if self.verbose:
            print(f"[COEFF] {message}")
    
    def _precompute_all_coefficients(self):
        """Pre-compute coefficients for all LUTs used in AES."""
        self._log("Pre-computing all LUT coefficients...")
        self._compute_sbox_coefficients()
        self._compute_xor_coefficients()
        self._compute_gf28_multiplication_coefficients()
        self._compute_bit_extraction_coefficients()
        
        # FIX #4: Add multivariate polynomial coefficients
        self._compute_multivariate_coefficients()
    
    def _compute_sbox_coefficients(self):
        """
        S-Box LUT(8→4)·다변수 LUT(4+4→4) 계수를 FFT로 계산합니다.
        - IFFT 정규화는 'backward'(기본값) → 별도 스케일링 불필요
        - 상·하위 니블을 각각 독립된 다항식으로 처리
        """
        self._log("Computing S-Box coefficients (fixed normalization)…")

        # ── AES S-Box 테이블 ─────────────────────────────────────────────
        self.sbox = [
            0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
            0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
            0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
            0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
            0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
            0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
            0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
            0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
            0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
            0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
            0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
            0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
            0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
            0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
            0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
            0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
        ]

        # ── Univariate 8→4 LUT 계수 ────────────────────────────────
        upper_vals = np.array([(s >> 4) & 0xF for s in self.sbox], dtype=np.complex128)
        lower_vals = np.array([s & 0xF for s in self.sbox],      dtype=np.complex128)

        self.sbox_upper_coeffs = ifft(upper_vals)   # 256-complex
        self.sbox_lower_coeffs = ifft(lower_vals)

        self._log(f"Univariate S-Box coefficients ready: {self.sbox_upper_coeffs.shape[0]}")

        # ── Multivariate 4+4→4 LUT 계수 (16×16) ────────────────────
        self._log("Computing multivariate S-Box coefficients…")

        sbox_upper_mv = np.empty((16, 16), dtype=np.complex128)
        sbox_lower_mv = np.empty((16, 16), dtype=np.complex128)

        for r in range(16):         # right nibble
            for l in range(16):     # left  nibble
                v = self.sbox[(l << 4) | r]
                sbox_upper_mv[r, l] = (v >> 4) & 0xF
                sbox_lower_mv[r, l] =  v        & 0xF

        self.sbox_upper_mv_coeffs = ifftn(sbox_upper_mv)  # shape (16,16)
        self.sbox_lower_mv_coeffs = ifftn(sbox_lower_mv)

        self._log(f"Multivariate tensor shape: {self.sbox_upper_mv_coeffs.shape}")

        if self.verbose:
            self._log(f"Max |univar| = {np.abs(self.sbox_upper_coeffs).max():.2e}")
            self._log(f"Max |multivar| = {np.abs(self.sbox_upper_mv_coeffs).max():.2e}")

    
    def _compute_xor_coefficients(self):
        """Compute XOR LUT coefficients."""
        self._log("Computing XOR LUT coefficients using FFT...")
        
        # 4-bit XOR (두 4-bit 입력 → 4-bit 출력)
        # 입력: 8비트 (상위 4비트 = a, 하위 4비트 = b)
        xor_4bit_values = np.zeros(256, dtype=complex)
        for i in range(256):
            a = (i >> 4) & 0xF
            b = i & 0xF
            xor_4bit_values[i] = a ^ b
        
        self.xor_4bit_coeffs = ifft(xor_4bit_values, n=256, norm='forward') * 256
        
        self._log("XOR coefficients computed")
    
    def _compute_gf28_multiplication_coefficients(self):
        """
        ## PAPER ALIGNMENT (Fix #5): True GF(2^8) multiplication LUTs
        """
        self._log("Computing GF(2^8) multiplication LUT coefficients…")

        # ---------- 1)  Multiplication by 2  ----------
        mult2_uint = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            x = i << 1
            if x & 0x100:          # overflow → reduce by AES poly x⁸+x⁴+x³+x+1 (0x11B)
                x ^= 0x11B
            mult2_uint[i] = x & 0xFF

        # store the raw LUT (정수형) – 선형층에서 그대로 재사용
        self.gf28_mult2_values = mult2_uint

        # FFT 계수는 복소수형이 필요하므로 여기서만 캐스팅
        mult2_complex = mult2_uint.astype(np.complex128)
        self.gf28_mult2_coeffs = ifft(mult2_complex, n=256, norm='forward') * 256


        # ---------- 2)  Multiplication by 3 (= 2·x  XOR  x) ----------
        mult3_uint = mult2_uint ^ np.arange(256, dtype=np.uint8)
        self.gf28_mult3_values = mult3_uint

        mult3_complex = mult3_uint.astype(np.complex128)
        self.gf28_mult3_coeffs = ifft(mult3_complex, n=256, norm='forward') * 256

        self._log("GF(2^8) multiplication coefficients computed")

    
    def _compute_bit_extraction_coefficients(self):
        """
        ## PAPER ALIGNMENT (Fix #4): Pre-compute bit extraction LUT coefficients
        """
        self._log("Computing bit extraction LUT coefficients...")
        
        self.bit_extract_coeffs = []
        for bit_idx in range(4):
            # LUT that extracts bit 'bit_idx' from a 4-bit nibble
            values = np.zeros(16, dtype=complex)
            for nibble in range(16):
                values[nibble] = (nibble >> bit_idx) & 1
            
            coeffs = ifft(values, n=16, norm='forward') * 16
            self.bit_extract_coeffs.append(coeffs)
        
        self._log("Bit extraction coefficients computed")
    
    def _compute_multivariate_coefficients(self):
        """
        FIX #4: Compute coefficients for multivariate polynomials (α > 1).
        
        For 2-variable LUTs, we use 2D FFT to compute coefficient tensors.
        """
        self._log("Computing multivariate polynomial coefficients...")
        
        # Example: 2-variable 4-bit AND operation (for testing)
        and_2var_table = np.zeros((16, 16), dtype=complex)
        for i in range(16):
            for j in range(16):
                and_2var_table[i, j] = i & j
        
        # 2D IFFT to get coefficient tensor
        self.and_2var_coeffs = ifftn(and_2var_table, norm='forward') * 256
        
        # Example: 2-variable 4-bit OR operation
        or_2var_table = np.zeros((16, 16), dtype=complex)
        for i in range(16):
            for j in range(16):
                or_2var_table[i, j] = i | j
        
        self.or_2var_coeffs = ifftn(or_2var_table, norm='forward') * 256
        
        self._log("Multivariate coefficients computed")
    
    def decompose_lut(self, lut_table: np.ndarray, input_bits: int, output_bits: int) -> List[np.ndarray]:
        """
        FIX #5: Decompose αℓ→βℓ LUT into β separate αℓ→ℓ LUTs.
        
        Args:
            lut_table: The full LUT as a 1D array of size 2^input_bits
            input_bits: Number of input bits (α*ℓ)
            output_bits: Number of output bits (β*ℓ)
        
        Returns:
            List of β coefficient arrays, one for each output bit
        """
        self._log(f"Decomposing {input_bits}→{output_bits} LUT into {output_bits} sub-LUTs")
        
        decomposed_coeffs = []
        
        for out_bit in range(output_bits):
            # Extract the out_bit-th bit from each LUT entry
            bit_values = np.zeros(len(lut_table), dtype=complex)
            for i, val in enumerate(lut_table):
                bit_values[i] = (int(val) >> out_bit) & 1
            
            # Compute coefficients for this bit's LUT
            coeffs = ifft(bit_values, norm='forward') * len(bit_values)
            decomposed_coeffs.append(coeffs)
        
        return decomposed_coeffs
    
    def compute_linear_layer_luts(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Compute coefficients for the unified ShiftRows+MixColumns LUT
        (16 input bytes × 16 output bytes ⇒ 256 개 8→8 LUT).
        Returns
        -------
        Dict[(in_pos, out_pos), np.ndarray]
            key : (0‥15, 0‥15)
            value : 256-complex-coeff 다항식 계수 벡터
        """
        self._log("Computing unified linear-layer LUT coefficients…")

        # ─── 고정 파라미터 ─────────────────────────────────────────────
        shift_rows_perm = [0, 5, 10, 15, 4, 9, 14, 3,
                        8, 13, 2, 7, 12, 1, 6, 11]

        mixcol = np.array([[0x02, 0x03, 0x01, 0x01],
                        [0x01, 0x02, 0x03, 0x01],
                        [0x01, 0x01, 0x02, 0x03],
                        [0x03, 0x01, 0x01, 0x02]], dtype=np.uint8)

        # 빠른 참조용
        mult2 = self.gf28_mult2_values.astype(np.int64)
        mult3 = self.gf28_mult3_values.astype(np.int64)

        luts: Dict[Tuple[int, int], np.ndarray] = {}

        # ─── 각 입력 바이트에 대해 LUT 생성 ────────────────────────────
        for in_pos in range(16):
            s_pos = shift_rows_perm[in_pos]        # ShiftRows 이후 위치
            col, row = divmod(s_pos, 4)            # MixColumns용 (열, 행)

            for out_row in range(4):               # 해당 열의 4개 출력 바이트
                out_pos = col * 4 + out_row
                coeff = mixcol[out_row, row]

                # 8→8 LUT 진리표 생성 (256 항목)
                if coeff == 0x01:
                    lut_values = np.arange(256, dtype=np.int64)
                elif coeff == 0x02:
                    lut_values = mult2
                elif coeff == 0x03:
                    lut_values = mult3
                else:                              # 이론상 나오지 않음
                    lut_values = np.zeros(256, dtype=np.int64)

                # FFT → 계수 (complex128)
                coeff_vec = ifft(lut_values.astype(np.complex128),
                                n=256, norm='forward') * 256

                luts[(in_pos, out_pos)] = coeff_vec

        self.linear_layer_luts = luts
        self._log(f"✓ Generated {len(luts)} LUTs for unified linear layer")
        return luts