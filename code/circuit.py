"""Logical circuit -> EO pulse execution, with merging of adjacent single-qubit gates.

Ops: ('u', k, U2x2) | ('rz', k, phi) | ('cnot', c, t) with |c-t|==1 | ('swap', k, k+1)
Compilation: CNOT(c->t) = (W on t) G (W^dag on t)            if t==c-1,
             CNOT(c->t) = H-conjugated G with control/target exchanged  if t==c+1  (G is the native 20-pulse gate
             with control on the right block, |S>_12|up>_3 encoding).  All single-qubit unitaries between two-qubit gates are multiplied together
             per qubit and synthesised once (<=4 pulses; exact Rz = 1 pulse).
"""
import numpy as np
from eo_sim import EOState, TimeNoise, CM_W, _rot, synth_single, CX_MODE, CHAD_L

H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], complex)
I2 = np.eye(2, dtype=complex)


def rz(phi):
    return np.diag([np.exp(-1j * phi / 2), np.exp(1j * phi / 2)])


def ry(th):
    return _rot([0, 1, 0], th)


def rx(th):
    return _rot([1, 0, 0], th)


class Circuit:
    def __init__(self, n):
        self.n = n
        self.ops = []

    def u(self, k, U):
        self.ops.append(('u', k, np.asarray(U, complex)))
        return self

    def rz(self, k, phi): return self.u(k, rz(phi))
    def rx(self, k, th): return self.u(k, rx(th))
    def ry(self, k, th): return self.u(k, ry(th))
    def h(self, k): return self.u(k, H)
    def x(self, k): return self.u(k, X)

    def cnot(self, c, t):
        """CNOT; a non-adjacent pair is routed by moving the target next to the control with logical
        SWAPs (9 SWAP pulses each) and moving it back afterwards."""
        if abs(c - t) == 1:
            self.ops.append(('cnot', c, t)); return self
        step = 1 if t < c else -1
        path = list(range(t, c, step))              # t, t+step, ..., c-step
        for a in path[:-1]:
            self.swap(a, a + step)
        self.ops.append(('cnot', c, c - step))
        for a in reversed(path[:-1]):
            self.swap(a, a + step)
        return self

    def swap(self, k, l):
        assert abs(k - l) == 1
        self.ops.append(('swap', min(k, l), max(k, l))); return self

    def cz(self, c, t):
        return self.h(t).cnot(c, t).h(t)

    def cp(self, c, t, lam):
        """controlled-phase diag(1,1,1,e^{i lam}) = Rz(lam/2) x Rz(lam/2) . exp(-i(-lam/4) Z Z)"""
        return self.rz(c, lam / 2).rz(t, lam / 2).zz(c, t, -lam / 4)

    def ccz(self, a, b, c):
        """CCZ = exp(i pi/8 (1 - Za - Zb - Zc + ZaZb + ZbZc + ZaZc - ZaZbZc)) for a line a-b-c (b in the middle)."""
        for q in (a, b, c):
            self.rz(q, np.pi / 4)
        self.zz(a, b, -np.pi / 8).zz(b, c, -np.pi / 8)
        self.zz(a, c, -np.pi / 8)                       # routed
        self.cnot(a, b).zz(b, c, np.pi / 8).cnot(a, b)   # Z_b -> Z_a Z_b inside the sandwich
        return self

    def zz(self, c, t, gamma):
        """exp(-i gamma Z_c Z_t)"""
        return self.cnot(c, t).rz(t, 2 * gamma).cnot(c, t)

    # ---- compile to primitive stream: ('u1',k,U) merged, ('cm',k), ('swapL',k)
    def compile(self):
        pend = [None] * self.n          # pending single-qubit unitary per qubit
        prim = []

        def push(k, U):
            pend[k] = U if pend[k] is None else U @ pend[k]

        def flush(k):
            if pend[k] is not None:
                if not np.allclose(pend[k], pend[k][0, 0] * I2, atol=1e-12) or abs(pend[k][0, 0]) < 0.5:
                    prim.append(('u1', k, pend[k]))
                pend[k] = None

        for op in self.ops:
            if op[0] == 'u':
                push(op[1], op[2])
            elif op[0] == 'cnot':
                c, t = op[1], op[2]
                if CX_MODE != 'zb':
                    # Chadwick CZ-type entangler, an exact CZ in our basis and symmetric in the two qubits:
                    # CNOT(c,t) = H_t CZ H_t
                    push(t, H); flush(c); flush(t)
                    prim.append(('chad', min(c, t)))
                    push(t, H)
                elif t == c - 1:            # control on the right block: native orientation
                    push(t, CM_W.conj().T); flush(c); flush(t)
                    prim.append(('cm', t)); push(t, CM_W)
                else:                       # t == c+1 : control on the left block
                    push(c, H); push(t, H); push(c, CM_W.conj().T); flush(c); flush(t)
                    prim.append(('cm', c)); push(c, CM_W); push(c, H); push(t, H)
            elif op[0] == 'swap':
                k = op[1]; flush(k + 1); flush(k); prim.append(('swapL', k))
        for k in reversed(range(self.n)):   # (order fixes the jitter draws; kept from v1 under the mirror map)
            flush(k)
        return prim

    def run(self, noise=None, state0=None):
        """Execute on an EOState; returns the state."""
        st = EOState(self.n, noise)
        if state0 is not None:
            st.set_logical(state0)
        for p in self.compile():
            if p[0] == 'u1':
                st.u1(p[1], p[2])
            elif p[0] == 'cm':
                st.native_cm(p[1])
            elif p[0] == 'chad':
                st.native_chadwick(p[1], 'cz_inplace' if CX_MODE == 'cz' else 'lccz_inplace')
            elif p[0] == 'swapL':
                st.swap_logical(p[1])
        return st

    def pulse_count(self):
        st = self.run(TimeNoise())
        return len(st.pulse_log), sum(t for _, _, t in st.pulse_log)

    # ---- ideal logical reference (numpy, 2^n)
    def ideal_state(self, state0=None):
        n = self.n
        psi = np.zeros(2 ** n, complex); psi[0] = 1
        if state0 is not None:
            psi = np.asarray(state0, complex).copy()
        psi = psi.reshape([2] * n)
        for op in self.ops:
            if op[0] == 'u':
                k, U = op[1], op[2]
                psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [k])), 0, k)
            elif op[0] == 'cnot':
                c, t = op[1], op[2]
                idx = [slice(None)] * n; idx[c] = 1
                sub = psi[tuple(idx)]
                sub = np.flip(sub, axis=t if t < c else t - 1)
                psi[tuple(idx)] = sub
            elif op[0] == 'swap':
                psi = np.swapaxes(psi, op[1], op[2])
        return psi.reshape(-1)


# ---------------------------------------------------------------- observables
def expect_pauli(circ, pauli, noise=None, state0=None):
    """<P> for a Pauli string like 'XZIY' (qubit 0 first), measured physically:
    basis-change gates (noisy pulses) appended to the circuit, then pair singlet/triplet readout."""
    c2 = Circuit(circ.n); c2.ops = list(circ.ops)
    qs = []
    for k, p in enumerate(pauli):
        if p == 'I':
            continue
        qs.append(k)
        if p == 'X':
            c2.h(k)
        elif p == 'Y':
            c2.u(k, H @ rz(-np.pi / 2))      # S^dag then H
    if not qs:
        return 1.0
    st = c2.run(noise, state0)
    return st.expect_z_string(qs)


def ideal_expect_pauli(circ, pauli, state0=None):
    psi = circ.ideal_state(state0).reshape([2] * circ.n)
    Z = np.diag([1, -1]).astype(complex); Y = np.array([[0, -1j], [1j, 0]])
    mats = {'X': X, 'Y': Y, 'Z': Z}
    phi = psi
    for k, p in enumerate(pauli):
        if p != 'I':
            phi = np.moveaxis(np.tensordot(mats[p], phi, axes=([1], [k])), 0, k)
    return float(np.vdot(psi.reshape(-1), phi.reshape(-1)).real)


if __name__ == '__main__':
    import time
    rng = np.random.default_rng(0)
    # random circuit test: EO execution vs ideal numpy
    for n in [2, 3, 4]:
        c = Circuit(n)
        for _ in range(6):
            k = rng.integers(n); c.ry(k, rng.uniform(0, 2 * np.pi)); c.rz(k, rng.uniform(0, 2 * np.pi))
            a = rng.integers(n - 1)
            if rng.random() < 0.5: c.cnot(a, a + 1)
            else: c.cnot(a + 1, a)
            if rng.random() < 0.3: c.swap(a, a + 1)
        t0 = time.time(); st = c.run(); dt = time.time() - t0
        amp, leak = st.logical_amplitudes()
        ideal = c.ideal_state()
        ov = abs(np.vdot(ideal, amp)) ** 2
        paulis = ['XZ', 'YY', 'ZX'] if n == 2 else ['X' + 'Z' * (n - 1), 'Y' * n]
        pe = [(expect_pauli(c, p), ideal_expect_pauli(c, p)) for p in paulis]
        print(f"n={n}: overlap={ov:.12f} leak={leak:.1e} pulses={len(st.pulse_log)} time={dt:.2f}s  paulis={[(round(a,6),round(b,6)) for a,b in pe]}")
