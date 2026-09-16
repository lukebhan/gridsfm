"""cuDSS-backed sparse direct solves for the optional GPU Newton closure.

Selected by ``closure.solver: cudss`` (default ``superlu`` = scipy SuperLU on
CPU). This wraps nvmath-python's high-level ``DirectSolver`` (cuDSS).

Two properties of the single-slack PF closure are exploited:

* Within one case's Newton solve the Jacobian keeps a FIXED sparsity pattern
  (same ``Ybus`` structure, same pvpq/pq index sets), so we ``plan`` (reorder +
  symbolic factorization -- the expensive, CPU-side step) once per case and then
  refactor IN PLACE as only the numeric values change across iterations.
* The adjoint needs ``J^T mu = g``; cuDSS has no transpose-solve flag, so we
  factor ``J^T`` directly (a different pattern from ``J``, so its own plan). The
  ``ClosurePool`` caches this ``J^T`` factor per case too, so both plans are
  reused across epochs (see ``ClosurePool.adjoint_factor`` in ``closure.py``).

cuDSS uses a single GPU context, so the cudss backend closes a batch SERIALLY in
the main process (see ``ClosurePool`` in ``closure.py``): for large grids the GPU
factorization is the win, not cross-case CPU parallelism.
"""
from __future__ import annotations

import glob
import logging
import os
import warnings

import numpy as np
import scipy.sparse as sp
import torch
from nvmath.sparse.advanced import DirectSolver, DirectSolverOptions

# cuDSS planning emits an info-level note when no multithreading layer is set; we
# supply one below, and silence the rest of nvmath's chatter in the training log.
logging.getLogger("nvmath").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*multithreading interface library.*")

_DEV = "cuda"
_F64 = torch.float64
_I32 = torch.int32


def _find_mtlayer():
    """Locate the cuDSS GOMP multithreading layer shipped with nvidia-cudss-cu12
    (speeds the CPU-side reorder/symbolic on large grids). Returns None if absent."""
    try:
        import nvidia
        for base in list(getattr(nvidia, "__path__", [])):
            hits = glob.glob(os.path.join(base, "**", "libcudss_mtlayer_gomp.so*"),
                             recursive=True)
            if hits:
                return sorted(hits)[0]
    except Exception:
        pass
    return None


_MTLAYER = _find_mtlayer()


def _options():
    return DirectSolverOptions(multithreading_lib=_MTLAYER) if _MTLAYER else None


def _csr_sorted(J: sp.spmatrix):
    Jc = J.tocsr()
    Jc.sum_duplicates()
    Jc.sort_indices()
    return Jc


class CudssFactor:
    """Persistent cuDSS LU factor for ONE case's Newton loop.

    ``plan`` is run once on the Jacobian's sparsity; ``refactor`` copies new
    values into the on-device buffer and re-factorizes without re-planning. The
    factored system is real, double precision, general (non-symmetric).
    """

    def __init__(self, J: sp.spmatrix):
        Jc = _csr_sorted(J)
        self.n = int(Jc.shape[0])
        self._indptr0 = Jc.indptr            # planned pattern (host copies for the
        self._indices0 = Jc.indices          # fast-path identity check in refactor)
        self._A = torch.sparse_csr_tensor(
            torch.as_tensor(Jc.indptr, dtype=_I32, device=_DEV),
            torch.as_tensor(Jc.indices, dtype=_I32, device=_DEV),
            torch.as_tensor(Jc.data, dtype=_F64, device=_DEV),
            size=(self.n, self.n))
        self._vals = self._A.values()        # exact buffer cuDSS reads at factorize()
        self._b = torch.empty(self.n, dtype=_F64, device=_DEV)
        self._slv = DirectSolver(self._A, self._b, options=_options())
        self._slv.plan()

    def refactor(self, J: sp.spmatrix):
        """Re-factorize with the current values of ``J`` (same sparsity pattern)."""
        Jc = _csr_sorted(J)
        if (Jc.nnz != self._vals.numel()
                or not np.array_equal(Jc.indptr, self._indptr0)
                or not np.array_equal(Jc.indices, self._indices0)):
            # Pattern drifted (should not happen within a single case); re-plan.
            self.free()
            self.__init__(J)
            Jc = _csr_sorted(J)
        self._vals.copy_(torch.as_tensor(Jc.data, dtype=_F64, device=_DEV))
        self._slv.factorize()

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve ``J x = rhs`` with the current factor; ``rhs``/return are host arrays."""
        self._b.copy_(torch.as_tensor(np.ascontiguousarray(rhs, dtype=np.float64),
                                      dtype=_F64, device=_DEV))
        x = self._slv.solve()
        return x.detach().to("cpu").numpy().ravel()

    def free(self):
        slv = getattr(self, "_slv", None)
        if slv is not None:
            try:
                slv.free()
            except Exception:
                pass
            self._slv = None
