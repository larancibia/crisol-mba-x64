"""Independent native x86-64 and Unicorn execution harnesses for Sanity Gate.

Executes exact slice bytes on real x86-64 hardware (via mmap RX trampoline)
and in Unicorn CPU emulator, returning (rax_out, cf_out) for each concrete vector.
"""

from __future__ import annotations

import ctypes
import mmap
import random
from dataclasses import dataclass
from typing import Iterable

from unicorn import Uc, UC_ARCH_X86, UC_MODE_64
from unicorn.x86_const import (
    UC_X86_REG_RAX,
    UC_X86_REG_RBX,
    UC_X86_REG_RCX,
    UC_X86_REG_RDX,
    UC_X86_REG_RSI,
    UC_X86_REG_RDI,
    UC_X86_REG_R8,
    UC_X86_REG_R9,
    UC_X86_REG_R10,
    UC_X86_REG_R11,
    UC_X86_REG_R12,
    UC_X86_REG_R13,
    UC_X86_REG_R14,
    UC_X86_REG_R15,
    UC_X86_REG_RSP,
    UC_X86_REG_RBP,
    UC_X86_REG_EFLAGS,
)


BOUNDARY_VALUES_64: tuple[int, ...] = (
    0,
    1,
    2,
    0x7F,
    0x80,
    0xFF,
    0x7FFF,
    0x8000,
    0xFFFF,
    0x7FFFFFFF,
    0x80000000,
    0xFFFFFFFF,
    0x100000000,
    0x7FFFFFFFFFFFFFFF,
    0x8000000000000000,
    0xFFFFFFFFFFFFFFFE,
    0xFFFFFFFFFFFFFFFF,
    0x5555555555555555,
    0xAAAAAAAAAAAAAAAA,
)


@dataclass(frozen=True)
class ConcreteVector:
    rdi: int
    rsi: int
    rdx: int
    rcx: int
    cf: int  # 0 or 1


def generate_test_vectors(seed: int = 20261123, n_total: int = 1024) -> list[ConcreteVector]:
    """Generate >= 1024 deterministic vectors covering boundaries, one-hot bits, and CF in {0,1}."""
    rng = random.Random(seed)
    vectors: list[ConcreteVector] = []

    # 1. Boundary combinations with both CF=0 and CF=1
    for cf in (0, 1):
        for v in BOUNDARY_VALUES_64:
            vectors.append(ConcreteVector(v, v, v, v, cf))
            vectors.append(ConcreteVector(v, 0, 1, 0xFFFFFFFFFFFFFFFF, cf))
            vectors.append(ConcreteVector(0, v, 0x80000000, 0x7FFFFFFF, cf))

    # 2. One-hot bits across all 64 bit positions
    for bit in range(64):
        oh = 1 << bit
        vectors.append(ConcreteVector(oh, 0, 0, 0, bit & 1))
        vectors.append(ConcreteVector(0, oh, oh, (~oh) & 0xFFFFFFFFFFFFFFFF, (bit + 1) & 1))

    # 3. Fill remainder with deterministic pseudo-random 64-bit values (half CF=0, half CF=1)
    while len(vectors) < n_total:
        cf = len(vectors) & 1
        vectors.append(
            ConcreteVector(
                rng.getrandbits(64),
                rng.getrandbits(64),
                rng.getrandbits(64),
                rng.getrandbits(64),
                cf,
            )
        )
    return vectors[:n_total] if len(vectors) >= n_total else vectors


# Native x86-64 trampoline:
#   uint64_t fn(uint64_t rdi, uint64_t rsi, uint64_t rdx, uint64_t rcx, uint64_t cf_in, uint64_t *cf_out)
# System V ABI args: rdi, rsi, rdx, rcx, r8 (cf_in), r9 (cf_out)
# Prologue:
#   push rbx; push rbp; push r12; push r13; push r14; push r15; push r9
#   xor eax, eax
#   xor r10d, r10d; xor r11d, r11d; xor ebx, ebx; xor ebp, ebp
#   xor r12d, r12d; xor r13d, r13d; xor r14d, r14d; xor r15d, r15d
#   xor r9d, r9d
#   bt r8, 0           ; sets CF = cf_in & 1
#   pushfq             ; save CF
#   xor r8d, r8d       ; zero r8 deterministically
#   popfq              ; restore CF
#   <SLICE BYTES>
# Epilogue:
#   pushfq
#   pop rdx
#   and rdx, 1
#   pop r9
#   mov [r9], rdx
#   pop r15; pop r14; pop r13; pop r12; pop rbp; pop rbx
#   ret

_PROLOGUE = bytes([
    0x53,                   # push rbx
    0x55,                   # push rbp
    0x41, 0x54,             # push r12
    0x41, 0x55,             # push r13
    0x41, 0x56,             # push r14
    0x41, 0x57,             # push r15
    0x41, 0x51,             # push r9  (save cf_out ptr)
    0x31, 0xC0,             # xor eax, eax
    0x45, 0x31, 0xD2,       # xor r10d, r10d
    0x45, 0x31, 0xDB,       # xor r11d, r11d
    0x31, 0xDB,             # xor ebx, ebx
    0x31, 0xED,             # xor ebp, ebp
    0x45, 0x31, 0xE4,       # xor r12d, r12d
    0x45, 0x31, 0xED,       # xor r13d, r13d
    0x45, 0x31, 0xF6,       # xor r14d, r14d
    0x45, 0x31, 0xFF,       # xor r15d, r15d
    0x45, 0x31, 0xC9,       # xor r9d, r9d
    0x49, 0x0F, 0xBA, 0xE0, 0x00,  # bt r8, 0
    0x9C,                   # pushfq
    0x45, 0x31, 0xC0,       # xor r8d, r8d
    0x9D,                   # popfq
])

_EPILOGUE = bytes([
    0x9C,                   # pushfq
    0x5A,                   # pop rdx
    0x48, 0x83, 0xE2, 0x01, # and rdx, 1
    0x41, 0x59,             # pop r9
    0x49, 0x89, 0x11,       # mov [r9], rdx
    0x41, 0x5F,             # pop r15
    0x41, 0x5E,             # pop r14
    0x41, 0x5D,             # pop r13
    0x41, 0x5C,             # pop r12
    0x5D,                   # pop rbp
    0x5B,                   # pop rbx
    0xC3,                   # ret
])

_FUNC_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_uint64),
)


class NativeSliceRunner:
    """Executes raw x86-64 slice bytes directly on the host CPU."""

    def __init__(self, code: bytes) -> None:
        blob = _PROLOGUE + code + _EPILOGUE
        size = ((len(blob) + 4095) // 4096) * 4096
        self._buf = mmap.mmap(
            -1,
            size,
            prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
        )
        self._buf.write(blob)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buf))
        self._fn = _FUNC_TYPE(addr)

    def run(self, vec: ConcreteVector) -> tuple[int, int]:
        cf_out = ctypes.c_uint64(0)
        rax = self._fn(vec.rdi, vec.rsi, vec.rdx, vec.rcx, vec.cf & 1, ctypes.byref(cf_out))
        return int(rax), int(cf_out.value & 1)

    def run_batch(self, vectors: Iterable[ConcreteVector]) -> list[tuple[int, int]]:
        return [self.run(v) for v in vectors]

    def close(self) -> None:
        if self._buf is not None:
            self._buf.close()
            self._buf = None

    def __enter__(self) -> "NativeSliceRunner":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class UnicornSliceRunner:
    """Executes raw x86-64 slice bytes inside Unicorn CPU emulator."""

    CODE_ADDR = 0x100000
    STACK_ADDR = 0x200000
    MEM_SIZE = 0x10000

    def __init__(self, code: bytes) -> None:
        self._code = code
        self._mu = Uc(UC_ARCH_X86, UC_MODE_64)
        self._mu.mem_map(self.CODE_ADDR, self.MEM_SIZE)
        self._mu.mem_map(self.STACK_ADDR, self.MEM_SIZE)
        self._mu.mem_write(self.CODE_ADDR, code)

    def run(self, vec: ConcreteVector) -> tuple[int, int]:
        mu = self._mu
        # Initialize all scratch regs identically to NativeSliceRunner (zero)
        for reg in (
            UC_X86_REG_RAX, UC_X86_REG_RBX, UC_X86_REG_RBP,
            UC_X86_REG_R8, UC_X86_REG_R9, UC_X86_REG_R10, UC_X86_REG_R11,
            UC_X86_REG_R12, UC_X86_REG_R13, UC_X86_REG_R14, UC_X86_REG_R15,
        ):
            mu.reg_write(reg, 0)
        mu.reg_write(UC_X86_REG_RSP, self.STACK_ADDR + self.MEM_SIZE - 0x100)
        mu.reg_write(UC_X86_REG_RDI, vec.rdi)
        mu.reg_write(UC_X86_REG_RSI, vec.rsi)
        mu.reg_write(UC_X86_REG_RDX, vec.rdx)
        mu.reg_write(UC_X86_REG_RCX, vec.rcx)
        mu.reg_write(UC_X86_REG_EFLAGS, 0x2 | (vec.cf & 1))
        mu.emu_start(self.CODE_ADDR, self.CODE_ADDR + len(self._code))
        rax = int(mu.reg_read(UC_X86_REG_RAX))
        cf_out = int(mu.reg_read(UC_X86_REG_EFLAGS)) & 1
        return rax, cf_out

    def run_batch(self, vectors: Iterable[ConcreteVector]) -> list[tuple[int, int]]:
        return [self.run(v) for v in vectors]
