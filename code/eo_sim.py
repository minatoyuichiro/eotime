"""Exchange-only (EO) 3-spin qubit simulator with pulse-time errors.

Conventions
-----------
* n logical qubits -> N = 3n physical spin-1/2.  Qubit k lives on spins (3k, 3k+1, 3k+2).
  Spin 0 is the most significant index of the state tensor.  |up> = index 0.
* Logical basis (S = 1/2, m = +1/2 sector, gauge fixed by polarisation):
    |0_L> = |S>_{3k,3k+1} (x) |up>_{3k+2}
    |1_L> = state with S=1/2, m=+1/2 orthogonal to |0_L> and to the S=3/2 state |W>.
  A pulse on the PAIR (3k,3k+1) is an exact Rz; a pulse on (3k+1,3k+2) rotates about an
  axis tilted 120 deg from z in the xz plane.
* Convention change (v2, 2026-10-12): v1 used |0_L> = |up>_{3k} (x) |S>_{3k+1,3k+2}.  v2 is the exact mirror
  image of v1 (spin i <-> N-1-i, qubit k <-> n-1-k, both basis states of a block times -1): every v1 result is
  reproduced by the mirrored circuit, and the experiment scripts build the mirrored circuits.
* Exchange pulse on spins (i,j) of duration t (units of pi/J, i.e. t=1 is a SWAP):
      U_ij(t) = exp(-i pi t S_i.S_j)  ~  cos(pi t/2) 1 - i sin(pi t/2) SWAP_ij   (global phase dropped)
  so that in the (singlet, triplet) basis  U = diag(1, e^{-i pi t})  (Zeuch-Bonesteel convention).
  Durations are physical: nominal t is always taken in [0, 2).
* Time-error model: actual duration  t' = t (1 + eps) + delta + (random terms).
"""
import numpy as np
from dataclasses import dataclass, field

SQ2, SQ3, SQ6 = np.sqrt(2.0), np.sqrt(3.0), np.sqrt(6.0)

# ---------------------------------------------------------------- encoding
def _enc_matrix():
    """8x2 matrix: columns are |0_L>, |1_L> in the 3-spin computational basis (index bits = spins 0,1,2; 1 = down)."""
    e = np.eye(8, dtype=complex)
    zero = (e[2] - e[4]) / SQ2             # |up dn up> - |dn up up> = |S>_{01} |up>_2   (singlet on pair 1,2)
    one = (e[2] + e[4] - 2 * e[1]) / SQ6   # (|up dn up> + |dn up up> - 2|up up dn>)/sqrt6  (sign: mirror image of v1)
    return np.column_stack([zero, one])

ENC = _enc_matrix()                       # 8 x 2
W_LEAK = np.array([0, 1, 1, 0, 1, 0, 0, 0], dtype=complex) / SQ3  # S=3/2, Sz=+1/2


# ---------------------------------------------------------------- noise model
@dataclass
class TimeNoise:
    """Pulse-duration error model.  All quantities in units of the SWAP time (pi/J).

    eps        : systematic multiplicative error  t -> t(1+eps)     (J / clock miscalibration)
    delta      : systematic additive offset       t -> t + delta    (rise/fall-time error)
    sigma_mult : per-pulse random multiplicative jitter, N(0, sigma_mult)
    sigma_add  : per-pulse random additive jitter,       N(0, sigma_add)
    pair_eps   : optional dict {(i,j): eps_ij} per-pair static miscalibration
    grid       : >0 rounds every nominal duration to a multiple of grid (time quantisation)
    seed       : rng seed
    """
    eps: float = 0.0
    delta: float = 0.0
    sigma_mult: float = 0.0
    sigma_add: float = 0.0
    pair_eps: dict = field(default_factory=dict)
    grid: float = 0.0          # if >0, nominal duration is rounded to this grid (no amplitude compensation)
    fp_bits: int = 0           # if >0, nominal duration rounded to a binary floating-point mantissa of fp_bits bits
    sigma_h: float = 0.0       # quasi-static Zeeman gradient: per-spin detuning h_i ~ N(0, sigma_h), units of J (pi/t_SWAP)
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.n_pulses = 0
        self.h = None              # drawn lazily per register size

    def gradient(self, N):
        if self.h is None:
            # drawn from the right end of the chain (reversed), so that a seed gives the same realisation as in v1
            self.h = self.rng.normal(0, self.sigma_h, N)[::-1].copy() if self.sigma_h else np.zeros(N)
        return self.h

    def actual(self, t, i, j):
        self.n_pulses += 1
        e = self.eps + self.pair_eps.get((i, j), 0.0)
        if self.grid:
            t = round(t / self.grid) * self.grid
        if self.fp_bits and t > 0:
            ex = np.floor(np.log2(t)); q = 2.0 ** (ex - (self.fp_bits - 1))
            t = round(t / q) * q
        tt = t * (1.0 + e) + self.delta
        if self.sigma_mult:
            tt += t * self.sigma_mult * self.rng.standard_normal()
        if self.sigma_add:
            tt += self.sigma_add * self.rng.standard_normal()
        return tt

    def is_noiseless(self):
        return (self.eps == 0 and self.delta == 0 and self.sigma_mult == 0
                and self.sigma_add == 0 and not self.pair_eps and not self.grid and not self.sigma_h and not self.fp_bits)


IDEAL = TimeNoise()


# ---------------------------------------------------------------- state (fixed-Sz sector)
import itertools
from functools import lru_cache
import scipy.sparse as sp


@lru_cache(maxsize=None)
def sector_basis(N, n_down):
    """Sorted array of N-bit integers (bit N-1-i = spin i, 1 = down) with exactly n_down ones."""
    out = []
    for downs in itertools.combinations(range(N), n_down):
        v = 0
        for i in downs:
            v |= 1 << (N - 1 - i)
        out.append(v)
    return np.array(sorted(out), dtype=np.int64)


@lru_cache(maxsize=None)
def sector_perm(N, n_down, i, j):
    """Index array p such that (SWAP_ij psi)[a] = psi[p[a]] in the sector basis."""
    basis = sector_basis(N, n_down)
    bi, bj = 1 << (N - 1 - i), 1 << (N - 1 - j)
    v = basis.copy()
    xi = (v & bi) != 0; xj = (v & bj) != 0
    diff = xi != xj
    v[diff] ^= (bi | bj)
    return np.searchsorted(basis, v)


@lru_cache(maxsize=None)
def sector_encoder(n):
    """Sparse (dim_sector x 2^n) matrix whose columns are the encoded logical basis states."""
    N = 3 * n
    basis = sector_basis(N, n)
    pats = [1, 2, 4]                       # 3-spin patterns in the Sz=+1/2 sector (bit2 bit1 bit0 = spins 3k,3k+1,3k+2)
    rows, cols, vals = [], [], []
    for b in range(2 ** n):
        bits = [(b >> (n - 1 - k)) & 1 for k in range(n)]
        for combo in itertools.product(range(3), repeat=n):
            amp = 1.0 + 0j; v = 0
            for k, c in enumerate(combo):
                amp *= ENC[pats[c], bits[k]]
                v |= pats[c] << (3 * (n - 1 - k))
            if abs(amp) > 1e-15:
                rows.append(np.searchsorted(basis, v)); cols.append(b); vals.append(amp)
    return sp.csc_matrix((vals, (rows, cols)), shape=(len(basis), 2 ** n))


class EOState:
    """State of n encoded qubits on N=3n spins, stored in the Sz = +n/2 sector (n spins down)."""

    def __init__(self, n_qubits, noise=None):
        self.n = n_qubits
        self.N = 3 * n_qubits
        self.noise = noise if noise is not None else TimeNoise()
        self.enc = sector_encoder(self.n)
        self.dim = self.enc.shape[0]
        self.pulse_log = []        # (i, j, t_nominal)
        self.set_logical(np.eye(1, 2 ** self.n, 0, dtype=complex).ravel())

    # --- encode / decode
    def set_logical(self, c):
        c = np.asarray(c, dtype=complex).reshape(-1)
        self.psi = self.enc @ c

    def logical_amplitudes(self):
        """Project onto the code space; returns (c (2^n,), leakage = 1 - |c|^2)."""
        c = self.enc.conj().T @ self.psi
        return c, 1.0 - float(np.vdot(c, c).real)

    # --- physics
    def _sz_field(self):
        """sum_i h_i S_z^i on the sector basis (S_z = +1/2 for up = bit 0)."""
        if not hasattr(self, '_szf'):
            basis = sector_basis(self.N, self.n); h = self.noise.gradient(self.N)
            f = np.zeros(len(basis))
            for i in range(self.N):
                down = ((basis >> (self.N - 1 - i)) & 1).astype(bool)
                f += h[i] * np.where(down, -0.5, 0.5)
            self._szf = f
        return self._szf

    def _sz_pair(self, i, j):
        """h_i S_z^i + h_j S_z^j on the sector basis."""
        basis = sector_basis(self.N, self.n); h = self.noise.gradient(self.N)
        f = np.zeros(len(basis))
        for k in (i, j):
            down = ((basis >> (self.N - 1 - k)) & 1).astype(bool)
            f += h[k] * np.where(down, -0.5, 0.5)
        return f

    def _perm(self, i, j):
        return sector_perm(self.N, self.n, i, j)

    def pulse(self, i, j, t):
        """Exchange pulse of nominal duration t (units pi/J) on spins i<j (absolute indices)."""
        t = float(t) % 2.0
        self.pulse_log.append((i, j, t))
        ta = self.noise.actual(t, i, j)
        c, s = np.cos(np.pi * ta / 2), np.sin(np.pi * ta / 2)
        if not self.noise.sigma_h:
            self.psi = c * self.psi - 1j * s * self.psi[self._perm(i, j)]
            return
        # Exact evolution under pi*t*(S_i.S_j + sum_k h_k S_z^k).  Fields on spins other than i,j commute with
        # the exchange and give phases; on the pair, the |up up>, |dn dn> states are SWAP eigenstates (phase), and the
        # {|up dn>, |dn up>} block is  (1/2) sigma_x + (Delta/2) sigma_z  with Delta = h_i - h_j  (up to a constant).
        h = self.noise.gradient(self.N); Delta = h[i] - h[j]
        field = self._sz_field() - self._sz_pair(i, j)            # fields of the other spins (+ constant part)
        basis = sector_basis(self.N, self.n)
        bi = ((basis >> (self.N - 1 - i)) & 1).astype(bool); bj = ((basis >> (self.N - 1 - j)) & 1).astype(bool)
        diff = bi != bj
        Om = np.sqrt(0.25 + 0.25 * Delta * Delta)
        cO, sO = np.cos(np.pi * ta * Om), np.sin(np.pi * ta * Om)
        # a_x = +Delta/2 when spin i is up (bit 0) and j down, -Delta/2 otherwise (within diff)
        a_x = np.where(bi, -0.5 * Delta, 0.5 * Delta)
        sw = self.psi[self._perm(i, j)]
        same = (c * self.psi - 1j * s * sw) * np.exp(-1j * np.pi * ta * self._sz_pair(i, j))   # equal-spin states: SWAP=1, phases
        mixed = cO * self.psi - 1j * (sO / Om) * (a_x * self.psi + 0.5 * sw)
        self.psi = np.where(diff, mixed, same) * np.exp(-1j * np.pi * ta * field)

    def swap_exact(self, i, j):
        self.psi = self.psi[self._perm(i, j)]

    # --- logical-level helpers
    def spins(self, k):
        return 3 * k, 3 * k + 1, 3 * k + 2

    def rz(self, k, phi):
        """Exact Rz(phi) = diag(e^{-i phi/2}, e^{i phi/2}) via one pair pulse."""
        a, b, c = self.spins(k)
        self.pulse(a, b, (-phi / np.pi) % 2.0)

    def seq(self, k, pulses):
        """Apply [(pair_code, t), ...] where pair_code 'p' = pair (3k,3k+1), 'l' = (3k+1,3k+2)."""
        a, b, c = self.spins(k)
        for code, t in pulses:
            if code == 'p':
                self.pulse(a, b, t)
            else:
                self.pulse(b, c, t)

    def u1(self, k, U):
        """Arbitrary single-qubit SU(2) U via <=4 pulses (synthesised, cached)."""
        self.seq(k, synth_single(U))

    def native_cm(self, k):
        """Zeuch-Bonesteel 20-pulse controlled-(n.sigma), mirrored for the |S>_12|up>_3 encoding:
        control = qubit k+1 (spins 3k+3..3k+5, 1-based 4-6), target = qubit k (spins 3k..3k+2, 1-based 1-3)."""
        s = 3 * k
        t1, t2 = 0.5, 1.5

        def R():
            self.pulse(s + 2, s + 3, t1)
            self.pulse(s + 1, s + 2, t2)
            self.pulse(s + 2, s + 3, 1.0)
            self.pulse(s + 0, s + 1, 1.0)
            self.pulse(s + 1, s + 2, 2 - t2)
            self.pulse(s + 2, s + 3, 2 - t1)
        R(); self.pulse(s + 3, s + 4, 1.0); R(); self.pulse(s + 3, s + 4, 1.0); R()

    def native_chadwick(self, k, key):
        """Sequence from the Chadwick et al. library (linear six-dot topology), spins 3k..3k+5."""
        import json
        if not hasattr(EOState, '_chad'):
            EOState._chad = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chadwick_linear.json')))
        e = EOState._chad[key]
        for (a, b), ang in zip(e['pairs'], e['angles']):
            i, j = sorted([a - 1, b - 1])            # library orientation as published: their basis (singlet on the first pair) is ours
            self.pulse(3 * k + i, 3 * k + j, (2 - ang) % 2.0)      # their U = cos(th/2) + i sin(th/2) SWAP  ->  t = 2 - th/pi

    def cnot(self, k):
        """CNOT control k+1 -> target k (the native gate has its control on the right block).
        CX_MODE 'zb'   : (1 x W) G_ZB (1 x W^dag)                       (20 + 8 pulses)
        CX_MODE 'cz'   : H_t G_CZ,Chadwick H_t  (26 + 2 H; their CZ is exact in our basis)
        CX_MODE 'lccz' : same with the leakage-controlled CZ of Chadwick (44 + 2 H)"""
        if CX_MODE == 'zb':
            self.u1(k, CM_W.conj().T); self.native_cm(k); self.u1(k, CM_W)
        else:
            Hd = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
            self.u1(k, Hd)
            self.native_chadwick(k, 'cz_inplace' if CX_MODE == 'cz' else 'lccz_inplace')
            self.u1(k, Hd)

    def swap_logical(self, k):
        """Logical SWAP of qubits k, k+1 by 9 physical nearest-neighbour SWAP pulses."""
        s = 3 * k
        for (i, j) in [(2, 3), (3, 4), (1, 2), (4, 5), (2, 3), (0, 1), (3, 4), (1, 2), (2, 3)]:
            self.pulse(s + i, s + j, 1.0)

    # --- measurement
    def expect_z_string(self, qubits):
        """<prod_k Z_k> with physical Z_k = P_singlet - P_triplet = -SWAP on the pair (3k,3k+1) of qubit k.
        Leaked (S=3/2) population reads as triplet, i.e. Z=-1."""
        phi = self.psi
        sign = 1.0
        for k in qubits:
            a, b, c = self.spins(k)
            phi = phi[self._perm(a, b)]
            sign = -sign
        return sign * float(np.vdot(self.psi, phi).real)

    def expect_all_zz(self, pairs):
        """<Z_i Z_j> for a list of qubit pairs from the current state (no extra gates)."""
        return [self.expect_z_string([i, j]) for i, j in pairs]


# ---------------------------------------------------------------- single-qubit synthesis
def _rot(axis, ang):
    ax = np.asarray(axis, float); ax = ax / np.linalg.norm(ax)
    X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1]).astype(complex)
    n_sig = ax[0] * X + ax[1] * Y + ax[2] * Z
    return np.cos(ang / 2) * np.eye(2) - 1j * np.sin(ang / 2) * n_sig


def logical_pulse_matrix(code, t):
    """2x2 logical action (up to global phase) of a single pulse on one encoded qubit."""
    st = EOState(1)
    out = np.zeros((2, 2), complex)
    for b in range(2):
        st.set_logical(np.eye(2)[b]); st.seq(0, [(code, t)])
        c, _ = st.logical_amplitudes(); out[:, b] = c
    return out


def _fid(U, V):
    return abs(np.trace(U.conj().T @ V)) ** 2 / 4.0


_AXIS_L = None
def axis_l():
    """Rotation axis of the (3k+1,3k+2) pulse in the logical Bloch sphere."""
    global _AXIS_L
    if _AXIS_L is None:
        t = 0.5; a = np.pi * t / 2
        M = logical_pulse_matrix('l', t)
        tr = np.trace(M); M = M / (tr / abs(tr))          # now M = cos a - i sin a (n.sigma)
        A = (M - np.cos(a) * np.eye(2)) / (-1j * np.sin(a))
        n = np.array([np.trace(A @ np.array([[0, 1], [1, 0]])).real,
                      np.trace(A @ np.array([[0, -1j], [1j, 0]])).real,
                      np.trace(A @ np.diag([1, -1])).real]) / 2
        _AXIS_L = n / np.linalg.norm(n)
    return _AXIS_L


def _seq_matrix(ts):
    """Logical matrix of pulses alternating p,l,p,l... with durations ts (analytic, fast)."""
    n = axis_l()
    U = np.eye(2, dtype=complex)
    for idx, t in enumerate(ts):
        if idx % 2 == 0:
            U = _rot([0, 0, -1], np.pi * t) @ U    # pair pulse: diag(1,e^{-i pi t}) = e^{-i pi t/2} Rz(-pi t)
        else:
            U = _rot(n, np.pi * t) @ U
    return U


import os
CX_MODE = os.environ.get('EOTIME_CX', 'zb')          # two-qubit sequence: 'zb' | 'cz' | 'lccz'
CHAD_L = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chadwick_L.npy')) if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chadwick_L.npy')) else None
SYNTH_MODE = os.environ.get('EOTIME_SYNTH', 'search')      # 'search': 4-pulse least-squares (shortest of 16 starts); 'fixed': fixed calibrated sequences
_FIXED = {}


def _fixed_sequences():
    """Fixed, calibrated pulse sequences: H and S (= Rz(pi/2), 1 pulse) are the only non-trivial blocks.
    Any U = e^{i a} Rz(al) Ry(be) Rz(ga) is realised as Rz(al) [S H Rz(be) H S^dag] Rz(ga), since
    Ry(be) = S H Rz(be) H S^dag (time order: S^dag, H, Rz(be), H, S); every Rz is one pair pulse."""
    if not _FIXED:
        H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
        _FIXED['H'] = synth_single_search(H)
    return _FIXED


def synth_single(U, tol=1e-10):
    if SYNTH_MODE == 'fixed':
        return synth_single_fixed(U)
    return synth_single_search(U, tol)


def _zyz(U):
    """U (up to phase) = Rz(al) Ry(be) Rz(ga)."""
    Un = np.asarray(U, complex); Un = Un / np.lib.scimath.sqrt(np.linalg.det(Un))
    be = 2 * np.arctan2(abs(Un[1, 0]), abs(Un[0, 0]))
    if abs(Un[0, 0]) < 1e-12:            # be = pi: U10 = e^{i(al-ga)/2}; take al = 0 -> ga = -2 arg U10
        al = 0.0; ga = -2 * np.angle(Un[1, 0])
    elif abs(Un[1, 0]) < 1e-12:
        al = 0.0; ga = 2 * np.angle(Un[1, 1])
    else:
        # Un = [[e^{-i(al+ga)/2} c, -e^{-i(al-ga)/2} s], [e^{i(al-ga)/2} s, e^{i(al+ga)/2} c]]
        p = np.angle(Un[1, 1])          # (al+ga)/2
        q = np.angle(Un[1, 0])          # (al-ga)/2
        al = p + q; ga = p - q
    return al, be, ga


def synth_single_fixed(U):
    """Pulse list for U using only fixed blocks: Rz -> 1 pulse; Ry(be) -> S H Rz(be) H S^dag (time order S^dag, H, Rz, H, S) with H a fixed 4-pulse block."""
    Un = np.asarray(U, complex); Un = Un / np.lib.scimath.sqrt(np.linalg.det(Un))
    if abs(Un[0, 1]) < 1e-12 and abs(Un[1, 0]) < 1e-12:
        phi = 2 * np.angle(Un[1, 1])
        return [('p', (-phi / np.pi) % 2.0)]
    al, be, ga = _zyz(Un)
    Hseq = _fixed_sequences()['H']
    def rz(phi):
        t = (-phi / np.pi) % 2.0
        return [('p', t)] if t > 1e-9 else []
    # time order: Rz(ga) first, then Ry(be) = S H Rz(be) H S^dag  (time order: S^dag, H, Rz(be), H, S), then Rz(al)
    seq = rz(ga) + rz(-np.pi / 2) + list(Hseq) + rz(be) + list(Hseq) + rz(np.pi / 2) + rz(al)
    return seq


_SYNTH_CACHE = {}
def synth_single_search(U, tol=1e-10):
    """Return pulse list [(code,t),...] (<=4 pulses) realising U (up to global phase).
    Exact Rz handled with 1 pulse; otherwise numerical 4-pulse Euler-like solve with multistart."""
    from scipy.optimize import least_squares
    Un = np.asarray(U, complex)
    Un = Un / np.lib.scimath.sqrt(np.linalg.det(Un))
    key = tuple(np.round(Un.ravel(), 9))
    if key in _SYNTH_CACHE:
        return _SYNTH_CACHE[key]
    # diagonal -> single pair pulse
    if abs(Un[0, 1]) < 1e-12 and abs(Un[1, 0]) < 1e-12:
        phi = 2 * np.angle(Un[1, 1])         # Rz(phi): diag(e^{-i phi/2}, e^{i phi/2})
        res = [('p', (-phi / np.pi) % 2.0)]
        _SYNTH_CACHE[key] = res
        return res

    def resid(ts):
        M = _seq_matrix(ts)
        # fix phase: compare M to Un up to phase via trace
        tr = np.trace(Un.conj().T @ M)
        ph = tr / abs(tr) if abs(tr) > 1e-12 else 1.0
        d = (M / ph - Un).ravel()
        return np.concatenate([d.real, d.imag])

    # multistart; among exact solutions keep the one with the shortest total pulse time
    # (shortest exposure to timing error).  Durations are folded into [0,2); t~0 or t~2 pulses are dropped.
    best = None
    rng = np.random.default_rng(1)
    for trial in range(16):
        x0 = rng.uniform(0, 2, 4)
        sol = least_squares(resid, x0, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=400)
        ts = np.mod(sol.x, 2.0); ts[np.abs(ts - 2.0) < 1e-9] = 0.0
        f = _fid(Un, _seq_matrix(ts))
        if 1 - f > tol:
            continue
        tot = ts.sum()
        if best is None or tot < best[0] - 1e-9:
            best = (tot, ts)
    if best is None:
        raise RuntimeError("single-qubit synthesis failed")
    _, ts = best
    codes = ['p', 'l', 'p', 'l']
    res = [(c, float(t)) for c, t in zip(codes, ts) if t > 1e-9]
    _SYNTH_CACHE[key] = res
    return res


# ---------------------------------------------------------------- native two-qubit gate characterisation
def logical_matrix_2q(apply_fn):
    """4x4 logical matrix (columns = images of |00>,|01>,|10>,|11>) and max leakage of a 2-qubit op."""
    out = np.zeros((4, 4), complex); leak = 0.0
    for b in range(4):
        st = EOState(2); st.set_logical(np.eye(4)[b]); apply_fn(st)
        c, l = st.logical_amplitudes(); out[:, b] = c; leak = max(leak, l)
    return out, leak


def _characterise_cm():
    G, leak = logical_matrix_2q(lambda st: st.native_cm(0))     # basis |q0 q1>, control = q1
    P = np.eye(4)[[0, 2, 1, 3]]
    G = P @ G @ P                                               # -> control-major ordering |c t>
    # G ~ phase * diag(1, M) in control-major ordering |c t>: blocks c=0 (rows/cols 0,1), c=1 (2,3)
    B0 = G[:2, :2]; B1 = G[2:, 2:]
    ph = B0[0, 0] / abs(B0[0, 0])
    M = B1 / ph
    # M = n.sigma with n real unit vector
    n = np.array([np.trace(M @ np.array([[0, 1], [1, 0]])).real,
                  np.trace(M @ np.array([[0, -1j], [1j, 0]])).real,
                  np.trace(M @ np.diag([1, -1])).real]) / 2
    # The axis is analytically (0, -sqrt3/2, -1/2).  It is fixed to the floating-point value obtained in v1, because the
    # 4-pulse synthesis of W picks one point of a one-parameter family of exact solutions and that choice is sensitive
    # to the last bit of n; with this value every synthesised sequence is identical to v1.
    n_v1 = np.array([-1.6228527075013995e-17, -0.86602540378443904, -0.50000000000000011])
    assert np.allclose(n, n_v1, atol=1e-12)
    n = n_v1
    # W with W (n.sigma) W^dag = X : rotate n -> x about axis n x x
    x = np.array([1.0, 0, 0])
    ax = np.cross(n, x); ang = np.arccos(np.clip(n @ x, -1, 1))
    W = _rot(ax, ang) if np.linalg.norm(ax) > 1e-12 else np.eye(2, dtype=complex)
    return P @ G @ P, leak, n, W                                # CM_G returned in the |q0 q1> basis

CM_G, CM_LEAK, CM_N, CM_W = _characterise_cm()


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True, linewidth=140)
    # 1. encoding sanity
    assert abs(np.vdot(ENC[:, 0], ENC[:, 1])) < 1e-12 and abs(np.vdot(W_LEAK, ENC[:, 1])) < 1e-12
    print("axis of (3k+1,3k+2) pulse:", axis_l(), " angle from z (deg):", np.degrees(np.arccos(axis_l()[2])))
    # 2. Rz check
    Mz = logical_pulse_matrix('p', 0.3); Mz /= np.lib.scimath.sqrt(np.linalg.det(Mz))
    print("pair pulse t=0.3 ~ Rz(-0.3 pi)? fidelity", _fid(Mz, _rot([0, 0, -1], 0.3 * np.pi)))
    # analytic sequence model vs physical simulation
    rng = np.random.default_rng(5)
    for _ in range(3):
        ts = rng.uniform(0, 2, 4)
        st = EOState(1); out = np.zeros((2, 2), complex)
        for b in range(2):
            st.set_logical(np.eye(2)[b]); st.seq(0, list(zip('plpl', ts))); out[:, b] = st.logical_amplitudes()[0]
        print("  seq model fidelity", _fid(_seq_matrix(ts), out))
    # 3. native CM gate
    print("native 2q gate: leakage", CM_LEAK, " n =", CM_N, "|n|=", np.linalg.norm(CM_N))
    CN, lk = logical_matrix_2q(lambda st: st.cnot(0))
    CNOT = np.eye(4)[[0, 3, 2, 1]].astype(complex)     # control q1 -> target q0, basis |q0 q1>
    print("CNOT fidelity", abs(np.trace(CNOT.conj().T @ CN)) ** 2 / 16, " leakage", lk,
          " pulses:", len(synth_single(CM_W)) * 2 + 20)
    # 4. logical swap
    SW, lk = logical_matrix_2q(lambda st: st.swap_logical(0))
    SWAP = np.eye(4)[[0, 2, 1, 3]].astype(complex)
    print("SWAP fidelity", abs(np.trace(SWAP.conj().T @ SW)) ** 2 / 16, " leakage", lk)
    # 5. single-qubit synthesis coverage
    H = np.array([[1, 1], [1, -1]]) / SQ2
    for name, U in [('H', H), ('X', _rot([1, 0, 0], np.pi)), ('Y', _rot([0, 1, 0], np.pi)),
                    ('Rx(0.7)', _rot([1, 0, 0], 0.7)), ('Ry(2.1)', _rot([0, 1, 0], 2.1)), ('W', CM_W)]:
        s = synth_single(U)
        M = logical_pulse_matrix('p', 0) ; st = EOState(1); out = np.zeros((2, 2), complex)
        for b in range(2):
            st.set_logical(np.eye(2)[b]); st.u1(0, U); out[:, b] = st.logical_amplitudes()[0]
        print(f"  {name:8s} pulses={len(s)}  fid={_fid(U/np.lib.scimath.sqrt(np.linalg.det(U)), out):.12f}  t={[round(t,4) for _,t in s]}")
