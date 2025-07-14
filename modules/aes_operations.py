"""
AES Operations Module

This module handles AES-specific operations including key expansion,
basis transformations, and round operations.
"""

import numpy as np
from typing import List, Tuple


class AESOperations:
    """Handles AES-specific operations and transformations."""
    
    def __init__(self):
        self._init_basis_matrices()
        self.num_rounds = 10  # Default to 1 round for testing; can be set to 10 for full AES-128
    
    def _init_basis_matrices(self):
        """Initialize basis transformation matrices for Tower Field representation."""
        # M: Standard polynomial basis -> Tower Field basis
        self.M = np.array([
            [1, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0]
        ], dtype=np.uint8)
        
        # M_inv: Tower Field basis -> Standard polynomial basis
        self.M_inv = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.uint8)
    
    def generate_round_keys(self, key: bytes) -> List[np.ndarray]:
        """
        표준 AES-128 키 스케줄 (4 × 32-bit words, 총 44 words) 구현.
        반환: 11 개의 round-key 바이트 배열 (numpy array 형태).
        """
        if len(key) != 16:
            raise ValueError("Key must be exactly 16 bytes")

        # ─── S-Box & Rcon ─────────────────────────────────────────────
        sbox = [
            0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
            0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
            0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
            0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
            0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
            0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
            0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
            0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
            0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
            0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
            0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
            0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
            0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
            0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
            0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
            0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16]
        rcon = [0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36]

        # ─── 초기 4 개 word (W0‥W3) ──────────────────────────────────
        words: List[List[int]] = [list(key[i*4:(i+1)*4]) for i in range(4)]

        round_keys: List[np.ndarray] = [np.array(key)]

        # ─── W4‥W43 생성 ─────────────────────────────────────────────
        for i in range(4, 44):
            temp = words[i-1].copy()

            if i % 4 == 0:                  # RotWord + SubWord + Rcon
                temp = temp[1:] + temp[:1]  # RotWord
                temp = [sbox[b] for b in temp]
                temp[0] ^= rcon[(i//4) - 1]

            # Wi = Wi-4 ⊕ temp
            new_word = [a ^ b for a, b in zip(words[i-4], temp)]
            words.append(new_word)

            # 매 4 word 마다 16-byte 라운드키 저장
            if i % 4 == 3:
                round_key_bytes = sum(words[i-3:i+1], [])  # flatten 4 words
                round_keys.append(np.array(round_key_bytes, dtype=np.uint8))

        return round_keys  # 총 11 개 (Round 0 ~ Round 10)

    
    def byte_to_tower_field(self, byte_val: int) -> np.ndarray:
        """Convert a byte to its Tower Field representation."""
        bits = np.array([(byte_val >> i) & 1 for i in range(8)], dtype=np.uint8)
        return self.M @ bits % 2
    
    def tower_field_to_byte(self, tower_val: np.ndarray) -> int:
        """Convert Tower Field representation back to byte."""
        bits = self.M_inv @ tower_val % 2
        return sum(int(bits[i]) << i for i in range(8))
    
    def preprocess_plaintext_batch(self, plaintexts: List[bytes]) -> np.ndarray:
        """
        Preprocess a batch of plaintexts by converting to Tower Field representation.
        
        Returns array ready for FHE processing.
        """
        if any(len(pt) != 16 for pt in plaintexts):
            raise ValueError("All plaintexts must be exactly 16 bytes")
        
        batch_data = []
        for plaintext in plaintexts:
            block_tower = []
            for byte in plaintext:
                tower_bits = self.byte_to_tower_field(byte)
                block_tower.extend(tower_bits)
            batch_data.append(block_tower)
        
        return np.array(batch_data, dtype=float)
    
    def get_shift_rows_permutation(self) -> List[int]:
        """Get the ShiftRows permutation mapping."""
        return [
            0, 5, 10, 15,   # Row 0: no shift
            4, 9, 14, 3,    # Row 1: shift left by 1
            8, 13, 2, 7,    # Row 2: shift left by 2
            12, 1, 6, 11    # Row 3: shift left by 3
        ]
    
    def get_mixcolumns_matrix(self) -> List[List[int]]:
        """Get the MixColumns transformation matrix."""
        return [
            [0x02, 0x03, 0x01, 0x01],
            [0x01, 0x02, 0x03, 0x01],
            [0x01, 0x01, 0x02, 0x03],
            [0x03, 0x01, 0x01, 0x02]
        ]