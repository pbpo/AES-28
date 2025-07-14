"""
FHE Operations Module

This module handles core FHE operations including noise reduction, bootstrapping,
and key generation with proper parameters from the paper.
"""

import numpy as np
import desilofhe
from typing import Dict, Optional, Tuple, List


class FHEOperations:
    """Manages FHE-specific operations and optimizations."""
    
    def __init__(self, verbose: bool = False, enable_bootstrapping: bool = False):
        self.verbose = verbose
        self.enable_bootstrapping = enable_bootstrapping
        self._initialize_engine()
    
    def _log(self, message: str):
        if self.verbose:
            print(f"[FHE-OPS] {message}")
    
    def _initialize_engine(self):
        """
        ## PAPER ALIGNMENT (Fix #9): Explicitly set FHE parameters from paper
        """
        self.N = 2**16  # Polynomial degree for 128-bit security
        self.log_Q = 1658  # Total modulus bitsize
        self.log_delta = 59  # Scaling factor bitsize
        self.levels = self.log_Q // self.log_delta  # Approximately 28 levels
        
        self._log(f"Paper parameters: N={self.N}, log(Q)={self.log_Q}, log(Δ)={self.log_delta}")
        
        # Create FHE engine with explicit parameters
        engine_params = {
            'polynomial_degree': self.N,
            'coefficient_modulus_bits': [60] + [self.log_delta] * (self.levels - 2) + [60],
            'scale': 2**self.log_delta,
            'security_level': 128
        }
        
        try:
            # Try to pass explicit parameters if supported
            self.engine = desilofhe.Engine(**engine_params)
        except:
            # Fallback with approximation if exact params not supported
            self._log("Warning: Using approximated parameters")
            self.engine = desilofhe.Engine(max_level=self.levels, mode='cpu')
        
        # Get actual engine parameters
        self.slot_count = self.engine.slot_count
        self.max_level = self.engine.max_level
        
        # Noise reduction parameters
        self.noise_reduction_degree = 16
        
        # Complex encoding
        self.zeta = np.exp(2j * np.pi / self.N)
        
    def generate_keys(self) -> Dict[str, object]:
        """Generate and cache all FHE keys needed by the module."""
        self._log("Generating FHE keys …")

        keys: Dict[str, object] = {}
        keys["sk"]         = self.engine.create_secret_key()
        keys["pk"]         = self.engine.create_public_key(keys["sk"])
        keys["relin_key"]  = self.engine.create_relinearization_key(keys["sk"])
        keys["rotation_key"] = self.engine.create_rotation_key(keys["sk"])

        # cache on the instance for later direct use
        self.sk          = keys["sk"]
        self.pk          = keys["pk"]
        self.relin_key   = keys["relin_key"]
        self.rotation_key = keys["rotation_key"]

        # ── 부트스트래핑 키 (선택) ──────────────────────────────────────
        self.bootstrap_key = None
        if self.enable_bootstrapping:
            self._log("Generating bootstrap key …")
            try:
                self.bootstrap_key = self.engine.create_bootstrap_key(self.sk)
                keys["bootstrap_key"] = self.bootstrap_key
                self._log("Bootstrap key OK")
            except Exception as e:
                self._log(f"⚠️  bootstrap key gen failed → disable: {e}")
                self.enable_bootstrapping = False

        # ── 컨쥬게이션(복소 켤레) Galois 키 ───────────────────────────
        self.conj_key = None
        try:
            self.conj_key = self.engine.create_galois_key(self.sk, -1)
            keys["conj_key"] = self.conj_key
        except Exception:
            self._log("⚠️  conj key not supported in this backend")

        return keys
        
    def generate_rotation_keys_for_gaps(self, sk, slot_gap: int, batch_size: int) -> Dict[int, object]:
        """Generate rotation keys needed for gapped layout operations."""
        self._log("Generating rotation keys for gapped layout...")
        gap_rotation_keys = {}
        
        # For byte-level operations with gaps
        for byte_offset in range(16):
            for bit_offset in range(8):
                rotation = byte_offset * slot_gap + bit_offset * batch_size
                if 0 < rotation < self.slot_count // 2:
                    try:
                        gap_rotation_keys[rotation] = \
                            self.engine.create_fixed_rotation_key(sk, rotation)
                    except:
                        pass
        
        # For shifting between bytes (used in ShiftRows)
        for shift in range(1, 16):
            rotation = shift * slot_gap
            if rotation < self.slot_count // 2:
                try:
                    gap_rotation_keys[rotation] = \
                        self.engine.create_fixed_rotation_key(sk, rotation)
                    gap_rotation_keys[-rotation] = \
                        self.engine.create_fixed_rotation_key(sk, -rotation)
                except:
                    pass
        
        self._log(f"Generated {len(gap_rotation_keys)} rotation keys")
        return gap_rotation_keys
    
    def apply_noise_reduction(
        self,
        ct: "desilofhe.Ciphertext",
        relin_key: Optional[object] = None,
        force: bool = False
    ) -> "desilofhe.Ciphertext":
        """
        Apply f(t) = -(t^{n+1})/n + (1+1/n)·t as in the paper (§4.3).
        Parameters
        ----------
        ct        : ciphertext to refresh
        relin_key : override relinearization key if desired
        force     : bypass the level-threshold skip
        """
        rk = relin_key or getattr(self, "relin_key", None)
        if rk is None:
            raise ValueError("relinearization key not set; call generate_keys() first")

        if not force and ct.level > int(self.max_level * 0.70):
            return ct  # noise still low, skip

        n = self.noise_reduction_degree
        self._log(f"Noise-reduction (deg={n+1}) on level {ct.level}")

        # 1) t^{n+1}
        t_power = ct
        for _ in range(n):
            t_power = self.engine.multiply(t_power, ct)
            t_power = self.engine.relinearize(t_power, rk)
            t_power = self.engine.rescale(t_power)

        # 2) -(1/n)·t^{n+1}
        neg_factor_pt = self.engine.encode(np.full(self.slot_count, -1.0 / n))
        term1 = self.engine.multiply(t_power, neg_factor_pt)

        # 3) (1+1/n)·t
        pos_factor_pt = self.engine.encode(np.full(self.slot_count, 1.0 + 1.0 / n))
        term2 = self.engine.multiply(ct, pos_factor_pt)

        # 4) Sum (level match handled automatically by add)
        result = self.engine.add(term1, term2)
        self._log(f" → new level {result.level}")
        return result
    
    def bootstrap_if_needed(self, ct: "desilofhe.Ciphertext") -> "desilofhe.Ciphertext":
        """
        Perform CKKS bootstrapping when level is critically low (≤3 by default).
        Requires self.enable_bootstrapping = True and self.bootstrap_key set.
        """
        if not self.enable_bootstrapping or self.bootstrap_key is None:
            return ct

        if ct.level > 3:                       # 아직 여유 있음
            return ct

        self._log(f"Bootstrapping (level {ct.level}) …")
        try:
            refreshed = self.engine.bootstrap(ct, self.bootstrap_key)
            self._log(f"Bootstrap OK → level {refreshed.level}")
            return refreshed
        except Exception as e:
            self._log(f"⚠️  bootstrap failed: {e} (continue unrefreshed)")
            return ct
        
    def encode_to_roots_of_unity(self, value: int) -> complex:
        """Encode integer value as root of unity: encode(x) = ζ^x"""
        return self.zeta ** value
    
    def decode_from_roots_of_unity(self, encoded: complex) -> int:
        """Decode from root of unity to nearest integer."""
        angle = np.angle(encoded)
        if angle < 0:
            angle += 2 * np.pi
        x = int(round(angle * self.N / (2 * np.pi))) % self.N
        return x
    
    def prepare_gapped_layout(self, batch_data: np.ndarray, slot_gap: int, 
                            batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data with byte splitting and gapped layout.
        
        Returns two arrays: left nibbles and right nibbles, each with gaps.
        """
        batch_data_size, total_bits = batch_data.shape
        
        left_nibbles = np.zeros(self.slot_count, dtype=complex)
        right_nibbles = np.zeros(self.slot_count, dtype=complex)
        
        for block_idx in range(min(batch_data_size, batch_size)):
            for byte_idx in range(16):
                byte_bits = batch_data[block_idx, byte_idx*8:(byte_idx+1)*8]
                byte_val = sum(int(byte_bits[i]) * (2**i) for i in range(8))
                
                left_nibble = (byte_val >> 4) & 0xF
                right_nibble = byte_val & 0xF
                
                left_encoded = self.encode_to_roots_of_unity(left_nibble)
                right_encoded = self.encode_to_roots_of_unity(right_nibble)
                
                slot_idx = byte_idx * slot_gap + block_idx
                if slot_idx < self.slot_count:
                    left_nibbles[slot_idx] = left_encoded
                    right_nibbles[slot_idx] = right_encoded
        
        return left_nibbles, right_nibbles
    
    def extract_from_gapped_layout(self, left_plain: np.ndarray, right_plain: np.ndarray,
                                 slot_gap: int, batch_size: int) -> List[bytes]:
        """Extract and decode results from gapped layout."""
        results = []
        
        for block_idx in range(batch_size):
            block_bytes = []
            
            for byte_idx in range(16):
                slot_idx = byte_idx * slot_gap + block_idx
                
                if slot_idx < len(left_plain):
                    left_val = self.decode_from_roots_of_unity(left_plain[slot_idx])
                    right_val = self.decode_from_roots_of_unity(right_plain[slot_idx])
                    
                    byte_val = ((left_val & 0xF) << 4) | (right_val & 0xF)
                    block_bytes.append(byte_val)
            
            results.append(bytes(block_bytes))
        
        return results