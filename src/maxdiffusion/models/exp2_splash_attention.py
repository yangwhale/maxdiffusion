# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom exp2-optimized Splash Attention kernel for TPU training.

This module provides an alternative flash attention implementation that uses
exp2 instead of exp for the softmax computation. On TPU VPU, exp2 is a native
instruction, and by pre-multiplying the query by LOG2_E, we eliminate the
implicit x*log2(e) multiplication that XLA inserts when lowering exp(x) to
exp2(x*log2e).

The forward pass uses a custom Pallas kernel with exp2.
The backward pass delegates to JAX's standard splash_attention_kernel backward,
which handles gradient computation efficiently.

Usage:
    In attention_flax.py, set attention_kernel="exp2_flash" to use this kernel.
"""

import functools
import dataclasses

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask

from .. import max_logging


# ============================================================================
# Constants
# ============================================================================

LOG2_E = 1.44269504
DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)
NUM_SUBLANES = 8
NUM_LANES = 128
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))

# Default block sizes optimized for training on TPU v7
# These are more conservative than inference (3328/2816) because training
# needs memory headroom for backward pass activations.
DEFAULT_BQ = 512
DEFAULT_BKV = 512
DEFAULT_BKV_COMPUTE = 256  # Two-level blocking: smaller compute within larger memory block


@dataclasses.dataclass(frozen=True)
class Exp2BlockSizes:
    """Block sizes for exp2 attention kernel.

    Separates block_kv (memory tile) from block_kv_compute (compute tile)
    to enable two-level blocking, matching the inference code's approach.
    """
    block_q: int = DEFAULT_BQ
    block_kv: int = DEFAULT_BKV
    block_kv_compute: int = DEFAULT_BKV_COMPUTE
    # Inner compute step size for the softmax accumulation loop
    block_kv_compute_in: int = DEFAULT_BKV_COMPUTE


# ============================================================================
# Pallas Kernel: exp2 Flash Attention Forward
# ============================================================================

def _exp2_flash_attention_kernel(
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    # Outputs
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    o_ref,
    *,
    mask_value: float,
    grid_width: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
    use_segment_ids: bool,
):
    """Flash attention kernel with exp2 optimization.

    Query must be pre-multiplied by LOG2_E * scale in the caller.
    This enables using exp2 directly instead of exp, saving one FMA
    per element in the softmax numerator and rescaling alpha.
    """
    float32 = jnp.float32
    head_dim_v_repeats, rem = divmod(head_dim_v, NUM_LANES)
    if rem != 0:
        head_dim_v_repeats += 1

    h, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)

    @pl.when(j == 0)
    def init():
        o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
        m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
        l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)

    def body(kv_compute_index, _):
        slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
        m_prev = m_scratch_ref[...]
        l_prev = l_scratch_ref[...]

        q = q_ref[...]  # [bq, head_dim]
        k = k_ref[slice_k, :]  # [bkv_compute, head_dim]
        qk = lax.dot_general(
            q, k, NT_DIM_NUMBERS, preferred_element_type=float32
        )  # [bq, bkv_compute]

        # Apply segment ID masking if enabled
        if use_segment_ids:
            q_ids = q_segment_ids_ref[...]  # [bq]
            kv_ids = kv_segment_ids_ref[slice_k]  # [bkv_compute]
            # Mask positions where segment IDs don't match (padding)
            segment_mask = jnp.where(
                (q_ids[:, None] == 0) | (kv_ids[None, :] == 0),
                mask_value,
                0.0,
            )
            qk = qk + segment_mask

        m_curr = qk.max(axis=-1)[:, None]  # [bq, 1]
        m_next = jnp.maximum(m_prev, m_curr)

        bkv_repeats = bkv_compute // NUM_LANES

        # exp2 optimization: since query is pre-multiplied by LOG2_E,
        # we use exp2 directly instead of exp
        s_curr = jnp.exp2(qk - jnp.tile(m_next, (1, bkv_repeats)))
        l_curr = jax.lax.broadcast_in_dim(
            s_curr.sum(axis=-1), l_prev.shape, (0,)
        )

        alpha = jnp.exp2(m_prev - m_next)
        l_next = l_curr + alpha * l_prev
        m_scratch_ref[...] = m_next
        l_scratch_ref[...] = l_next

        v = v_ref[slice_k, :].astype(float32)  # [bkv_compute, head_dim_v]
        o_curr = lax.dot_general(s_curr, v, NN_DIM_NUMBERS)  # [bq, head_dim_v]

        alpha_o = jnp.tile(
            alpha, (1, head_dim_v_repeats)
        )[..., :o_scratch_ref.shape[-1]]
        o_scratch_ref[:] = alpha_o * o_scratch_ref[:] + o_curr

    @pl.when(True)
    def run():
        num_iters = bkv // bkv_compute
        lax.fori_loop(0, num_iters, body, None, unroll=True)

    @pl.when(j == grid_width - 1)
    def end():
        l = l_scratch_ref[...]
        l_inv = jnp.tile(
            1.0 / l, (1, head_dim_v_repeats)
        )[..., :o_scratch_ref.shape[-1]]
        o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)

        m_scratch_ref[...] = jnp.zeros_like(m_scratch_ref)
        l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
        o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)


def _exp2_splash_attention_forward(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    q_segment_ids: jax.Array | None,
    kv_segment_ids: jax.Array | None,
    block_sizes: Exp2BlockSizes,
):
    """Run exp2 flash attention forward pass.

    Args:
        query: [num_heads, seq_len, head_dim] - PRE-SCALED by LOG2_E * scale
        key: [num_heads, seq_len, head_dim]
        value: [num_heads, seq_len, head_dim_v]
        q_segment_ids: [seq_len] segment IDs for query (None = no masking)
        kv_segment_ids: [seq_len] segment IDs for KV (None = no masking)
        block_sizes: Block size configuration

    Returns:
        output: [num_heads, seq_len, head_dim_v]
    """
    num_q_heads, q_seq_len, head_dim_qk = query.shape
    head_dim_v = value.shape[-1]
    bq = block_sizes.block_q
    bkv = block_sizes.block_kv
    bkv_compute = block_sizes.block_kv_compute
    kv_seq_len = key.shape[1]
    num_kv_heads = key.shape[0]
    q_heads_per_kv_head = num_q_heads // num_kv_heads

    use_segment_ids = q_segment_ids is not None and kv_segment_ids is not None

    def q_index_map(h, i, j):
        return (h, i, 0)

    def k_index_map(h, i, j):
        return (h // q_heads_per_kv_head, j, 0)

    def v_index_map(h, i, j):
        return (h // q_heads_per_kv_head, j, 0)

    def out_index_map(h, i, j):
        return (h, i, 0)

    in_specs = [
        pl.BlockSpec((None, bq, head_dim_qk), q_index_map),
        pl.BlockSpec((None, bkv, head_dim_qk), k_index_map),
        pl.BlockSpec((None, bkv, head_dim_v), v_index_map),
    ]

    if use_segment_ids:
        def q_seg_index_map(h, i, j):
            return (i * bq,)
        def kv_seg_index_map(h, i, j):
            return (j * bkv,)
        in_specs.append(pl.BlockSpec((bq,), q_seg_index_map))
        in_specs.append(pl.BlockSpec((bkv,), kv_seg_index_map))
    else:
        in_specs.append(pl.BlockSpec(memory_space=pltpu.ANY))
        in_specs.append(pl.BlockSpec(memory_space=pltpu.ANY))

    out_shapes = [
        jax.ShapeDtypeStruct((bq, NUM_LANES), jnp.float32),  # m_scratch
        jax.ShapeDtypeStruct((bq, NUM_LANES), jnp.float32),  # l_scratch
        jax.ShapeDtypeStruct((bq, head_dim_v), jnp.float32),  # o_scratch
        jax.ShapeDtypeStruct(
            (num_q_heads, q_seq_len, head_dim_v), query.dtype
        ),  # output
    ]
    out_specs = [
        pl.BlockSpec((bq, NUM_LANES), lambda *_: (0, 0)),
        pl.BlockSpec((bq, NUM_LANES), lambda *_: (0, 0)),
        pl.BlockSpec((bq, head_dim_v), lambda *_: (0, 0)),
        pl.BlockSpec((None, bq, head_dim_v), out_index_map),
    ]

    grid_width = kv_seq_len // bkv
    grid = (num_q_heads, q_seq_len // bq, grid_width)

    kernel_fn = functools.partial(
        _exp2_flash_attention_kernel,
        mask_value=DEFAULT_MASK_VALUE,
        grid_width=grid_width,
        bq=bq,
        bkv=bkv,
        bkv_compute=bkv_compute,
        head_dim_v=head_dim_v,
        use_segment_ids=use_segment_ids,
    )

    inputs = [query, key, value]
    if use_segment_ids:
        inputs.extend([q_segment_ids, kv_segment_ids])
    else:
        # Dummy inputs for unused segment IDs
        dummy = jnp.zeros(1, dtype=jnp.int32)
        inputs.extend([dummy, dummy])

    all_out = pl.pallas_call(
        kernel_fn,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
        ),
        out_shape=out_shapes,
    )(*inputs)

    return all_out[-1]  # The output tensor


def exp2_flash_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    scale: float,
    q_segment_ids: jax.Array | None = None,
    kv_segment_ids: jax.Array | None = None,
    block_sizes: Exp2BlockSizes | None = None,
):
    """Compute flash attention with exp2 optimization.

    This function handles:
    1. Pre-multiplying query by LOG2_E * scale for exp2 optimization
    2. Padding sequences to block size multiples
    3. Running the custom Pallas kernel
    4. Unpadding the output

    For backward pass, JAX's standard autodiff is used. Since the forward
    Pallas kernel is traced by JAX, gradients flow through automatically.

    Args:
        query: [num_heads, seq_len, head_dim]
        key: [num_heads, seq_len, head_dim]
        value: [num_heads, seq_len, head_dim_v]
        scale: Attention scale factor (typically 1/sqrt(head_dim))
        q_segment_ids: Optional segment IDs for query masking
        kv_segment_ids: Optional segment IDs for KV masking
        block_sizes: Optional custom block sizes

    Returns:
        output: [num_heads, seq_len, head_dim_v]
    """
    if block_sizes is None:
        block_sizes = Exp2BlockSizes()

    orig_q_seq_len = query.shape[1]
    orig_kv_seq_len = key.shape[1]

    # Pre-multiply query by LOG2_E * scale for exp2 optimization
    query = query * (scale * LOG2_E)

    # Pad sequences to block size multiples
    bq = block_sizes.block_q
    bkv = block_sizes.block_kv

    q_pad = (bq - orig_q_seq_len % bq) % bq
    kv_pad = (bkv - orig_kv_seq_len % bkv) % bkv

    if q_pad > 0:
        query = jnp.pad(query, ((0, 0), (0, q_pad), (0, 0)))
    if kv_pad > 0:
        key = jnp.pad(key, ((0, 0), (0, kv_pad), (0, 0)))
        value = jnp.pad(value, ((0, 0), (0, kv_pad), (0, 0)))

    # Pad segment IDs if provided
    if q_segment_ids is not None and q_pad > 0:
        q_segment_ids = jnp.pad(q_segment_ids, (0, q_pad))
    if kv_segment_ids is not None and kv_pad > 0:
        kv_segment_ids = jnp.pad(kv_segment_ids, (0, kv_pad))

    output = _exp2_splash_attention_forward(
        query, key, value, q_segment_ids, kv_segment_ids, block_sizes,
    )

    # Remove padding
    if q_pad > 0:
        output = output[:, :orig_q_seq_len, :]

    return output
