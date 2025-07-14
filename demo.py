"""
FHE-AES-128 — Modular, Paper-Aligned Implementation
===================================================
This file wires together all specialised sub-modules so that a *single* class
(`FheAES128Modular`) offers a ready-to-run, fully-optimised homomorphic AES-128
engine that follows every optimisation described in “Amortized Large Look-up
Table Evaluation with Multivariate Polynomials for Homomorphic Encryption”
(ePrint 2024/274).

Key design points
-----------------
* Clear separation of concerns – each responsibility lives in its own module
  (`coefficients`, `fhe_ops`, `aes_ops`, `lut_eval`, `hom_aes`).
* The main class only orchestrates: initialise, preprocess, encrypt,
  post-process.
* No library globals or side-effects: everything is passed explicitly.
* All parameters come straight from the paper (N = 2¹⁶, log Q ≈ 1658, …).

USAGE
-----
```bash
python fhe_aes128_modular.py [--enable-bootstrapping]
```

The script at the bottom will run a demo on four test vectors and print a small
performance summary.
"""
from __future__ import annotations

import argparse
import time
from typing import List, Tuple, Optional

import numpy as np
import desilofhe  
# ──────────────────────────────────────────────────────────────────────────────
# Local modules (relative imports).  These must live in the same Python package
# directory – adapt the import paths if you keep them elsewhere.
# ──────────────────────────────────────────────────────────────────────────────
from modules.coefficient_computation import CoefficientComputer
from modules.fhe_operations        import FHEOperations
from modules.aes_operations        import AESOperations
from modules.lut_evaluation         import LUTEvaluator
from modules.homomorphic_aes       import HomomorphicAES


class FheAES128Modular:
    """High-level façade that glues all sub-modules together."""

    # ─────────────────────────────────────────── initialisation ──┐
    def __init__(
        self,
        key: bytes,
        *,
        batch_size: int = 1,
        verbose: bool = False,
        enable_bootstrapping: bool = False,
    ) -> None:
        if len(key) != 16:
            raise ValueError("AES-128 key must be exactly 16 bytes")

        self.batch_size = batch_size
        self.verbose     = verbose
        self.slot_gap    = 2048  # <-- paper-prescribed gap between bytes

        self._log("Bootstrapping is %s", "ENABLED" if enable_bootstrapping else "disabled")

        # 1)  Low-level CKKS/CKG setup ─────────────────────────────────────────
        self.fhe  = FHEOperations(verbose=verbose, enable_bootstrapping=enable_bootstrapping)
        self.keys = self.fhe.generate_keys()  # secret/public + aux keys

        # 2)  Static data / coefficient generation ────────────────────────────
        self.coeffs = CoefficientComputer(verbose=verbose)

        # 3)  Rotation keys for gapped SIMD layout ────────────────────────────
        self.keys["gap_rotation_keys"] = self.fhe.generate_rotation_keys_for_gaps(
            sk=self.keys["sk"], slot_gap=self.slot_gap, batch_size=batch_size
        )

        # 4)  Plain AES helpers (key schedule, TF conversions, …) ─────────────
        self.aes = AESOperations()

        # 5)  LUT evaluator (handles monomial-basis reuse + polynomial eval) ───
        self.lut = LUTEvaluator(self.fhe.engine, self.keys, verbose=verbose)

        # 6)  Homomorphic AES round operations module ─────────────────────────
        self.haes = HomomorphicAES(
            engine      = self.fhe.engine,
            keys        = self.keys,
            coeffs      = self.coeffs,
            lut_evaluator = self.lut,
            fhe_ops     = self.fhe,
            slot_gap    = self.slot_gap,
            batch_size  = self.batch_size,
            verbose     = verbose,
        )

        # 7)  Expand AES key and encrypt round keys once  ─────────────────────
        self._log("Pre-encrypting round keys …")
        self.round_keys   = self.aes.generate_round_keys(key)
        self.round_key_ct = self._encrypt_round_keys()

        self._log("Initialisation complete! Ready to encrypt 🔒")

    # ─────────────────────────────────────────────── helpers ────┘
    # Utility: formatted logging only if verbose flag is on.
    def _log(self, fmt: str, *args) -> None:
        if self.verbose:
            print("[MAIN] " + fmt % args if args else fmt)

    # ---------------------------------------------------------------------
    # Round-key preprocessing (encrypt each nibble into gapped layout)
        # ---------------------------------------------------------------------
    def _encrypt_round_keys(
        self,
    ) -> List[Tuple[desilofhe.Ciphertext, desilofhe.Ciphertext]]:
        """
        • 각 라운드키를 두 개의 CKKS 암호문(left/right nibble)으로 변환한다.
        • ‘가파른(gapped) SIMD 레이아웃’에 맞춰 슬롯을 배치한다.
        • bytes / bytearray / 0-d bytes ndarray / uint8 배열 등
        어떤 형태로 들어와도 안전하게 처리한다.
        """
        ct_keys: List[Tuple[desilofhe.Ciphertext, desilofhe.Ciphertext]] = []

        for rk in self.round_keys:

            # ── 1. 라운드키를 길이 16 uint8 벡터로 변환 ──────────────────
            if isinstance(rk, (bytes, bytearray)):
                #   rk 자체가 bytes
                bytes_iter = np.frombuffer(rk, dtype=np.uint8)

            elif isinstance(rk, np.ndarray):
                if rk.dtype == np.uint8 and rk.ndim == 1 and rk.size == 16:
                    #   이미 올바른 uint8 벡터
                    bytes_iter = rk
                else:
                    #   0-d ndarray(바이트열) 혹은 다른 형태
                    bytes_iter = np.frombuffer(rk.tobytes(), dtype=np.uint8)
            else:
                #   리스트·튜플 등 → uint8 배열로 캐스팅
                bytes_iter = np.asarray(rk, dtype=np.uint8).ravel()

            if bytes_iter.size != 16:
                raise ValueError("round-key must have exactly 16 bytes")

            # ── 2. gapped layout 배열 초기화 ─────────────────────────────
            left  = np.zeros(self.fhe.slot_count, dtype=complex)
            right = np.zeros(self.fhe.slot_count, dtype=complex)

            # ── 3. 각 바이트를 두 니블로 나눠 루트-오브-유니티로 인코딩 ───
            for byte_idx, byte_val in enumerate(bytes_iter):
                ln, rn = (byte_val >> 4) & 0xF, byte_val & 0xF
                ln_enc = self.fhe.encode_to_roots_of_unity(ln)
                rn_enc = self.fhe.encode_to_roots_of_unity(rn)

                # 모든 블록(batch) 슬롯에 동일 키 삽입
                for blk in range(self.batch_size):
                    slot = byte_idx * self.slot_gap + blk
                    if slot < self.fhe.slot_count:
                        left[slot]  = ln_enc
                        right[slot] = rn_enc

            # ── 4. 평문 배열 → CKKS 암호문 ───────────────────────────────
            ct_left  = self.fhe.engine.encrypt(
                left,  self.keys["pk"], level=self.fhe.max_level
            )
            ct_right = self.fhe.engine.encrypt(
                right, self.keys["pk"], level=self.fhe.max_level
            )
            ct_keys.append((ct_left, ct_right))

        return ct_keys


    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def preprocess_plaintexts(self, pts: List[bytes]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert cleartexts → Tower Field bits → gapped nibble layout."""
        tf_bits = self.aes.preprocess_plaintext_batch(pts)
        return self.fhe.prepare_gapped_layout(tf_bits, self.slot_gap, self.batch_size)

    def encrypt(
        self,
        left_nibbles: np.ndarray,
        right_nibbles: np.ndarray,
        *,
        slot_lut_indices: Optional[List[int]] = None,  # reserved for future work
    ) -> Tuple[desilofhe.Ciphertext, desilofhe.Ciphertext]:
        self._log("⇢  Homomorphic encryption of %d block(s)", self.batch_size)
        eng = self.fhe.engine

        left_ct  = eng.encrypt(left_nibbles,  self.keys["pk"], level=self.fhe.max_level)
        right_ct = eng.encrypt(right_nibbles, self.keys["pk"], level=self.fhe.max_level)

        # ——— ROUND 0 ————————————————————————————————————————————————
        left_ct, right_ct = self.haes.homomorphic_add_round_key(
            left_ct, right_ct, *self.round_key_ct[0]
        )

        # ——— ROUNDS 1 … 9 ————————————————————————————————————————————
        for rnd in range(1, 10):
            self._log("  • round %d", rnd)
            left_ct, right_ct = self.haes.homomorphic_sub_bytes(left_ct, right_ct)
            left_ct, right_ct = self.haes.homomorphic_linear_layer(left_ct, right_ct)
            left_ct, right_ct = self.haes.homomorphic_add_round_key(
                left_ct, right_ct, *self.round_key_ct[rnd]
            )

        # ——— ROUND 10 (final) ————————————————————————————————————————
        self._log("  • round 10 (final)")
        left_ct, right_ct = self.haes.homomorphic_sub_bytes(left_ct, right_ct)
        left_ct, right_ct = self.haes.homomorphic_shift_rows_only(left_ct, right_ct)
        left_ct, right_ct = self.haes.homomorphic_add_round_key(
            left_ct, right_ct, *self.round_key_ct[10]
        )
        self._log("⇠  Encryption finished")
        return left_ct, right_ct

    def decrypt_and_decode(self, left_ct: desilofhe.Ciphertext, right_ct: desilofhe.Ciphertext) -> List[bytes]:
        self._log("Decoding ciphertext(s) back to cleartext …")
        left_plain  = self.fhe.engine.decrypt(left_ct,  self.keys["sk"])
        right_plain = self.fhe.engine.decrypt(right_ct, self.keys["sk"])
        return self.fhe.extract_from_gapped_layout(left_plain, right_plain, self.slot_gap, self.batch_size)


# ──────────────────────────────────────────────────────────────────────────────
# CLI demonstration
# ──────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    parser = argparse.ArgumentParser(description="Run a quick demo of FHE-AES-128 (modular)")
    parser.add_argument("--enable-bootstrapping", action="store_true", help="turn on CKKS bootstrapping")
    opts = parser.parse_args()

    key = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    pts = [
        bytes.fromhex("00112233445566778899aabbccddeeff"),
 
    ]

    print("\n→ Bootstrapping:", "enabled" if opts.enable_bootstrapping else "disabled")
    t0 = time.time()
    cipher = FheAES128Modular(key, batch_size=len(pts), verbose=True, enable_bootstrapping=opts.enable_bootstrapping)
    print(f"Initialisation done in {time.time() - t0:.1f}s\n")

    left_n, right_n = cipher.preprocess_plaintexts(pts)
    t0 = time.time()
    ct_l, ct_r      = cipher.encrypt(left_n, right_n)
    enc_sec         = time.time() - t0
    print(f"✦ Encrypted in {enc_sec:.1f}s  ({len(pts)/enc_sec:.2f} blk/s)\n")

    # Decrypt & verify against AES-ECB reference
    from Crypto.Cipher import AES  # PyCryptodome
    ref = [AES.new(key, AES.MODE_ECB).encrypt(pt) for pt in pts]
    dec = cipher.decrypt_and_decode(ct_l, ct_r)

    ok = all(a == b for a, b in zip(ref, dec))
    for i, (a, b) in enumerate(zip(ref, dec)):
        print(f"block {i}:", "✓" if a == b else "✗", "—", b.hex())
    print("\nSUCCESS" if ok else "MISMATCH – numerical precision needs tuning")


if __name__ == "__main__":
    _demo()
