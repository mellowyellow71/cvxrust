"""
Copyright 2025, the CVXPY authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import annotations

import struct

import numpy as np
import scipy.sparse as sp

from cvxpy.lin_ops import LinOp
from cvxpy.lin_ops.backends.base import CanonBackend

# Op type string -> int mapping for serialization (must match OpType::from_int in Rust)
_OP_TYPE_MAP = {
    "variable": 0, "scalar_const": 1, "dense_const": 2, "sparse_const": 3,
    "param": 4, "sum": 5, "neg": 6, "reshape": 7, "mul": 8, "rmul": 9,
    "mul_elem": 10, "div": 11, "index": 12, "transpose": 13, "promote": 14,
    "broadcast_to": 15, "hstack": 16, "vstack": 17, "concatenate": 18,
    "sum_entries": 19, "trace": 20, "diag_vec": 21, "diag_mat": 22,
    "upper_tri": 23, "conv": 24, "kron_r": 25, "kron_l": 26, "no_op": 27,
}

# Op types whose data field is itself a LinOp tree
_LINOP_DATA_OPS = {"mul", "rmul", "mul_elem", "div", "conv", "kron_l", "kron_r"}

_PACK_F64 = struct.Struct('<d').pack
_UNPACK_I64 = struct.Struct('<q').unpack


def _F64_BITS(v: float) -> int:
    """Bit-pattern of a float64 as a signed i64 (for the metadata stream)."""
    return _UNPACK_I64(_PACK_F64(v))[0]


def _append_axis_data(meta: list, axis, keepdims) -> None:
    """Append an AxisData payload (tag 7) to the metadata stream."""
    meta.append(7)
    if axis is None:
        meta.append(0)
    elif isinstance(axis, (int, np.integer)):
        meta.append(1)
        meta.append(int(axis))
    else:
        axes = list(axis)
        meta.append(2)
        meta.append(len(axes))
        meta.extend(int(a) for a in axes)
    meta.append(1 if keepdims else 0)


def serialize_linop_trees(lin_ops: list[LinOp]) -> tuple:
    """
    Serialize a list of LinOp trees into flat buffers for the Rust backend.

    Walks the trees in pre-order and packs:
    - node_meta: np.ndarray[i64] — all node metadata as one flat stream, so
      Rust deserialization is a single pass over a borrowed slice with no
      per-node Python object access at all
    - float_data: np.ndarray[f64] with all dense array / sparse value data concatenated
    - int_data: np.ndarray[i64] with all sparse indices / indptr data concatenated

    Per-node layout in node_meta (must stay in sync with
    DeserializationContext in cvxpy_rust/src/linop.rs):
      [op_type, ndim, *shape, num_args, data_tag, *payload]

    Payloads by data tag:
      0=None: ()                      1=Int: (value,)
      2=Float: (float64 bits as i64,) 3=DenseArray: (f_off, f_len, ndim, *shape)
      4=SparseArray: (f_off, f_len, i_off_idx, i_len_idx, i_off_ptr,
                      i_len_ptr, nrows, ncols)
      5=Slices: (n, *(start, stop, step) per slice)
      6=LinOpRef: () — the data LinOp follows inline, before this node's args
      7=AxisData: (kind 0|1|2, [value | n, *axes], keepdims)
      8=ConcatAxis: (has, value)
    """
    meta: list[int] = []
    float_chunks: list[np.ndarray] = []
    int_chunks: list[np.ndarray] = []
    float_offset = 0
    int_offset = 0

    # Hot path: bound-method/global lookups hoisted to locals; one extend
    # with a single tuple literal per node in the common cases; f64
    # bit-pattern via struct (much cheaper than np scalar .view()).
    extend = meta.extend
    op_map = _OP_TYPE_MAP
    linop_data_ops = _LINOP_DATA_OPS
    f64_bits = _F64_BITS

    def _serialize_node(lin_op):
        nonlocal float_offset, int_offset

        t = lin_op.type
        shape = lin_op.shape
        nargs = len(lin_op.args)
        data = lin_op.data

        has_data_linop = False

        if data is None:
            extend((op_map[t], len(shape), *shape, nargs, 0))

        elif t in ("variable", "param", "diag_vec", "diag_mat"):
            extend((op_map[t], len(shape), *shape, nargs, 1, int(data)))

        elif t == "scalar_const":
            extend((op_map[t], len(shape), *shape, nargs, 2, f64_bits(float(data))))

        elif t == "dense_const":
            arr = np.asarray(data, dtype=np.float64)
            flat = arr.ravel(order='F')
            float_chunks.append(flat)
            n = len(flat)
            extend((op_map[t], len(shape), *shape, nargs,
                    3, float_offset, n, arr.ndim, *arr.shape))
            float_offset += n

        elif t in linop_data_ops:
            # Data is a LinOp — serialized inline after this node, before args
            extend((op_map[t], len(shape), *shape, nargs, 6))
            has_data_linop = True

        elif t == "sparse_const":
            csc = sp.csc_array(data)
            vals = np.asarray(csc.data, dtype=np.float64)
            indices = np.asarray(csc.indices, dtype=np.int64)
            indptr = np.asarray(csc.indptr, dtype=np.int64)
            float_chunks.append(vals)
            int_chunks.append(indices)
            int_chunks.append(indptr)
            extend((
                op_map[t], len(shape), *shape, nargs,
                4,
                float_offset, len(vals),
                int_offset, len(indices),
                int_offset + len(indices), len(indptr),
                csc.shape[0], csc.shape[1],
            ))
            float_offset += len(vals)
            int_offset += len(indices) + len(indptr)

        elif t == "index":
            extend((op_map[t], len(shape), *shape, nargs, 5, len(data)))
            for s in data:
                extend((int(s.start), int(s.stop), int(s.step)))

        elif t == "sum_entries":
            extend((op_map[t], len(shape), *shape, nargs))
            axis = data[0]
            keepdims = bool(data[1]) if len(data) > 1 else False
            _append_axis_data(meta, axis, keepdims)

        elif t == "transpose":
            if len(data) > 0:
                extend((op_map[t], len(shape), *shape, nargs))
                _append_axis_data(meta, data[0], False)
            else:
                extend((op_map[t], len(shape), *shape, nargs, 0))

        elif t == "concatenate":
            axis = data[0] if data else None
            if axis is None:
                extend((op_map[t], len(shape), *shape, nargs, 8, 0, 0))
            else:
                extend((op_map[t], len(shape), *shape, nargs, 8, 1, int(axis)))

        else:
            extend((op_map[t], len(shape), *shape, nargs, 0))

        # If data is a LinOp, serialize it BEFORE args
        if has_data_linop:
            _serialize_node(data)

        # Serialize args in order
        for arg in lin_op.args:
            _serialize_node(arg)

    try:
        for lin_op in lin_ops:
            _serialize_node(lin_op)
    except KeyError as exc:
        raise ValueError(
            f"LinOp type {exc.args[0]!r} is not supported by the RUST "
            f"canonicalization backend"
        ) from exc

    node_meta = np.array(meta, dtype=np.int64)

    # Concatenate buffers
    if float_chunks:
        float_data = np.concatenate(float_chunks)
    else:
        float_data = np.empty(0, dtype=np.float64)

    if int_chunks:
        int_data = np.concatenate(int_chunks)
    else:
        int_data = np.empty(0, dtype=np.int64)

    return node_meta, float_data, int_data


class RustCanonBackend(CanonBackend):
    """
    Rust canonicalization backend using PyO3 bindings to cvxpy_rust.

    The LinOp trees are pre-serialized into flat numpy buffers
    (:func:`serialize_linop_trees`), then the coefficient matrix is built in
    Rust with exact-NNZ pre-allocation and rayon parallelism.

    Usage:
        prob.solve(canon_backend="RUST")

    For benchmarks against the SCIPY/CPP/COO backends and implementation
    notes, see rust_benchmarks/ at the repository root and
    https://github.com/cvxpy/cvxpy/pull/3018.
    """

    def build_matrix(
        self, lin_ops: list[LinOp], order: str = 'F'
    ) -> sp.csc_array:
        import cvxpy_rust
        if order != 'F':
            raise ValueError(
                f"order={order!r} is not supported by the RUST canonicalization "
                f"backend; only column-major order ('F') is implemented."
            )
        self.id_to_col[-1] = self.var_length

        nodes, float_data, int_data = serialize_linop_trees(lin_ops)
        data, (rows, cols), shape = cvxpy_rust.build_matrix_serialized(
            nodes, float_data, int_data,
            self.param_size_plus_one,
            self.id_to_col,
            self.param_to_size,
            self.param_to_col,
            self.var_length,
        )

        self.id_to_col.pop(-1)
        return sp.csc_array((data, (rows, cols)), shape=shape)
