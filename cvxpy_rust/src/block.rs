//! Typed Block IR for forward-mode-AD-style canonicalization.
//!
//! Each LinOp subtree's value is a Jacobian matrix. The legacy code flattens
//! every subtree to COO triplets immediately. This module introduces a typed
//! representation so subtrees can stay in their natural form (Identity,
//! Dense, ColPerm, ...) until the very end, enabling fast paths for
//! `Mul(A, x)` and friends.
//!
//! Leaf handlers produce typed Blocks (Identity for Variable, Dense for
//! dense/scalar constants, Coo for SparseConst/Param); `process_index_block`
//! produces Identity or ColPerm; `process_mul`'s fast-path probe is the
//! consumer that exploits the typed structure today. Other handlers still
//! flow through `Block::Coo` via `NodeValue::from_coo`.
//!
//! The variant set is intentionally minimal — just what's emitted and
//! consumed today. Add new variants when there's a producer *and* a
//! consumer, not before.

use std::sync::Arc;

use crate::tensor::SparseTensor;

/// Strided F-order (column-major) dense matrix view backed by an `Arc<[f64]>`.
///
/// Allows zero-copy Reshape / Transpose / Index by manipulating strides.
/// Element at logical `(i, j)` is at `data[row_offset + i * row_stride + j * col_stride]`.
#[derive(Debug, Clone)]
pub struct DenseF {
    pub rows: usize,
    pub cols: usize,
    pub data: Arc<[f64]>,
    pub row_stride: usize,
    pub col_stride: usize,
    pub row_offset: usize,
}

impl DenseF {
    /// Wrap a plain F-order (column-major) buffer of length `rows * cols`.
    pub fn from_col_major(rows: usize, cols: usize, data: Arc<[f64]>) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        DenseF {
            rows,
            cols,
            data,
            row_stride: 1,
            col_stride: rows,
            row_offset: 0,
        }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        debug_assert!(i < self.rows && j < self.cols);
        self.data[self.row_offset + i * self.row_stride + j * self.col_stride]
    }
}

/// Typed representation of one subtree's Jacobian, before COO flattening.
#[derive(Debug, Clone)]
pub enum Block {
    /// `I_n`. Realised at flatten time as `n` entries `(1.0, i, var_col_offset+i)`.
    Identity { n: usize },
    /// Permutation: row `i` has a single `1.0` at column `perm[i]`.
    /// Emitted by `Index` over an `Identity` input when the slice is
    /// non-contiguous.
    ColPerm { perm: Arc<[i64]> },
    /// Strided F-order dense matrix.
    Dense(Arc<DenseF>),
    /// Mixed-parametric COO triplets — the escape hatch. Wraps a
    /// `SparseTensor` whose `param_offsets` already encode parameter slots,
    /// so `BlockEntry::param_slot` is unused for `Coo` blocks.
    Coo(SparseTensor),
}

impl Block {
    /// Number of nonzero entries this block will emit when flattened.
    pub fn estimated_nnz(&self) -> usize {
        match self {
            Block::Identity { n } => *n,
            Block::ColPerm { perm } => perm.len(),
            Block::Dense(d) => d.rows * d.cols,
            Block::Coo(t) => t.nnz(),
        }
    }
}

/// One contribution to a `NodeValue`.
///
/// For typed blocks (Identity / Dense / SparseCsc / ...) the placement is
/// `(param_slot, var_col_offset)`. For `Block::Coo`, both fields are ignored
/// — the COO entries already carry their own row/col/param indices.
#[derive(Debug, Clone)]
pub struct BlockEntry {
    pub param_slot: i64,
    pub var_col_offset: i64,
    pub block: Block,
}

/// The Jacobian of one LinOp subtree, expressed as a sum of typed contributions.
///
/// Conceptual shape is `(out_rows, var_cols)` per parameter slot. Non-parametric
/// subtrees have a single entry with `param_slot == ctx.const_param()`.
#[derive(Debug, Clone)]
pub struct NodeValue {
    pub out_rows: usize,
    pub var_cols: usize,
    pub blocks: Vec<BlockEntry>,
}

impl NodeValue {
    /// Empty value (zero matrix, no contributions).
    pub fn empty(out_rows: usize, var_cols: usize) -> Self {
        NodeValue {
            out_rows,
            var_cols,
            blocks: Vec::new(),
        }
    }

    /// Wrap an existing `SparseTensor` as a single COO contribution.
    ///
    /// Used as the boundary between legacy COO-returning handlers and any
    /// future Block-aware caller. `param_slot` and `var_col_offset` on the
    /// resulting `BlockEntry` are placeholders (zero); callers must not rely
    /// on them for `Coo` blocks.
    pub fn from_coo(tensor: SparseTensor) -> Self {
        let (out_rows, var_cols) = tensor.shape;
        NodeValue {
            out_rows,
            var_cols,
            blocks: vec![BlockEntry {
                param_slot: 0,
                var_col_offset: 0,
                block: Block::Coo(tensor),
            }],
        }
    }

    /// Number of nonzero entries the flattened tensor will contain.
    pub fn estimated_nnz(&self) -> usize {
        self.blocks.iter().map(|e| e.block.estimated_nnz()).sum()
    }

    /// Flatten all contributions to a single `SparseTensor`.
    ///
    /// The output has the same `(out_rows, var_cols)` shape as the conceptual
    /// matrix. Entries from different blocks are concatenated in block order.
    /// Within a typed block the entries are emitted with monotonically
    /// increasing `flat_row = col * out_rows + row`, but across blocks they
    /// are not sorted — `BuildMatrixResult::from_tensor` handles the final
    /// parallel sort, preserving the April 3 sort optimisation.
    pub fn into_coo(self) -> SparseTensor {
        let shape = (self.out_rows, self.var_cols);
        let total = self.estimated_nnz();
        let mut out = SparseTensor::with_capacity(shape, total);

        for entry in self.blocks {
            let BlockEntry {
                param_slot,
                var_col_offset,
                block,
            } = entry;
            match block {
                Block::Identity { n } => {
                    for i in 0..n {
                        out.push(1.0, i as i64, var_col_offset + i as i64, param_slot);
                    }
                }

                Block::ColPerm { perm } => {
                    for (row, &col) in perm.iter().enumerate() {
                        out.push(1.0, row as i64, var_col_offset + col, param_slot);
                    }
                }

                Block::Dense(dense) => {
                    // F-order walk: cols outer, rows inner — gives monotone flat_row.
                    for j in 0..dense.cols {
                        let col = var_col_offset + j as i64;
                        for i in 0..dense.rows {
                            let v = dense.get(i, j);
                            if v != 0.0 {
                                out.push(v, i as i64, col, param_slot);
                            }
                        }
                    }
                }

                Block::Coo(t) => {
                    out.extend(t);
                }
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn const_param(param_size_plus_one: i64) -> i64 {
        param_size_plus_one - 1
    }

    #[test]
    fn identity_block_round_trips() {
        let nv = NodeValue {
            out_rows: 3,
            var_cols: 7,
            blocks: vec![BlockEntry {
                param_slot: const_param(2),
                var_col_offset: 4,
                block: Block::Identity { n: 3 },
            }],
        };
        let t = nv.into_coo();
        assert_eq!(t.shape, (3, 7));
        assert_eq!(t.data, vec![1.0, 1.0, 1.0]);
        assert_eq!(t.rows, vec![0, 1, 2]);
        assert_eq!(t.cols, vec![4, 5, 6]);
        assert_eq!(t.param_offsets, vec![1, 1, 1]);
    }

    #[test]
    fn col_perm_emits_one_per_row() {
        let perm: Arc<[i64]> = Arc::from(vec![2_i64, 0, 1]);
        let nv = NodeValue {
            out_rows: 3,
            var_cols: 4,
            blocks: vec![BlockEntry {
                param_slot: 0,
                var_col_offset: 1,
                block: Block::ColPerm { perm },
            }],
        };
        let t = nv.into_coo();
        assert_eq!(t.data, vec![1.0, 1.0, 1.0]);
        assert_eq!(t.rows, vec![0, 1, 2]);
        // var_col_offset=1, perm=[2,0,1] -> cols=[3,1,2]
        assert_eq!(t.cols, vec![3, 1, 2]);
    }

    #[test]
    fn dense_f_order_walk_is_column_major() {
        // 2x3 column-major: rows 2, cols 3
        // Logical:
        //   1 3 5
        //   2 4 6
        let data: Arc<[f64]> = Arc::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let dense = Arc::new(DenseF::from_col_major(2, 3, data));
        let nv = NodeValue {
            out_rows: 2,
            var_cols: 5,
            blocks: vec![BlockEntry {
                param_slot: 0,
                var_col_offset: 2,
                block: Block::Dense(dense),
            }],
        };
        let t = nv.into_coo();
        // Expect column-major emission: (col=2,row=0,1.0), (col=2,row=1,2.0),
        //                               (col=3,row=0,3.0), (col=3,row=1,4.0),
        //                               (col=4,row=0,5.0), (col=4,row=1,6.0).
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(t.cols, vec![2, 2, 3, 3, 4, 4]);
        assert_eq!(t.rows, vec![0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn dense_skips_zero_entries() {
        let data: Arc<[f64]> = Arc::from(vec![0.0, 2.0, 0.0, 0.0]);
        let dense = Arc::new(DenseF::from_col_major(2, 2, data));
        let nv = NodeValue {
            out_rows: 2,
            var_cols: 2,
            blocks: vec![BlockEntry {
                param_slot: 0,
                var_col_offset: 0,
                block: Block::Dense(dense),
            }],
        };
        let t = nv.into_coo();
        assert_eq!(t.nnz(), 1);
        assert_eq!(t.data, vec![2.0]);
        assert_eq!(t.rows, vec![1]);
        assert_eq!(t.cols, vec![0]);
    }

    #[test]
    fn from_coo_round_trips() {
        let mut t = SparseTensor::empty((4, 6));
        t.push(7.0, 1, 2, 0);
        t.push(-3.0, 0, 5, 2);
        let nv = NodeValue::from_coo(t.clone());
        let back = nv.into_coo();
        assert_eq!(back.shape, t.shape);
        assert_eq!(back.data, t.data);
        assert_eq!(back.rows, t.rows);
        assert_eq!(back.cols, t.cols);
        assert_eq!(back.param_offsets, t.param_offsets);
    }

    #[test]
    fn multi_block_concatenates() {
        // Two Identity blocks at different var_col_offsets — emitted as
        // separate runs in the into_coo output.
        let nv = NodeValue {
            out_rows: 2,
            var_cols: 4,
            blocks: vec![
                BlockEntry {
                    param_slot: 0,
                    var_col_offset: 0,
                    block: Block::Identity { n: 2 },
                },
                BlockEntry {
                    param_slot: 0,
                    var_col_offset: 2,
                    block: Block::Identity { n: 2 },
                },
            ],
        };
        let t = nv.into_coo();
        assert_eq!(t.nnz(), 4);
        assert_eq!(t.data, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(t.rows, vec![0, 1, 0, 1]);
        assert_eq!(t.cols, vec![0, 1, 2, 3]);
    }
}
