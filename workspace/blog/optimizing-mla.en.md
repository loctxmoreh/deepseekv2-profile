# DeepSeek-V2 high performance inference optimization notes: MLA optimization

## Preface

Recently, the DeepSeek-V2 model released by Magic Cube has received widespread
attention from academia and industry. As a large MoE model with 236B
parameters, DeepSeek-V2 uses a unique DeepSeekMoE architecture design to
activate only 21B parameters per token, and replaces the traditional MHA and
MQA attention mechanisms with the newly proposed MLA mechanism, achieving a
significant reduction in the size of the KV Cache during the reasoning process.
Therefore, DeepSeek-V2 can achieve model performance comparable to GPT-4 at a
lower reasoning cost.

The MLA mechanism is a core innovation in DeepSeek-V2. As researchers in the
field of computer systems, we naturally dare not comment on the algorithm
design of MLA from the perspective of AI/ML, but from the perspective of
System, MLA is undoubtedly a very excellent design. In recent years, one of the
main reasons for the high cost of large model inference is the low utilization
of GPU computing power. With the emergence of specialized circuits such as
Tensor Core, the computing power of modern high-performance GPUs is much higher
than its memory bandwidth. For every byte of data read by the GPU, it often has
to participate in hundreds of calculations to ensure that the GPU's computing
unit will not be idle, thereby achieving better computing resource utilization
(i.e. MFU). However, due to the limitations of various factors, the task load
of large model inference is usually difficult to provide such a high computing
intensity, that is, the parameters read by the GPU are discarded and the next
parameter is read if they are not involved in enough calculations, which makes
the memory bandwidth the performance bottleneck of the entire inference
process. One of the major obstacles is the space occupation of KV Cache: GPU
memory space is often very limited, and a larger KV Cache will result in fewer
requests being processed simultaneously, that is, a smaller batch size; a
number of works represented by vLLM start from this perspective to optimize the
memory utilization of KV Cache, thereby improving the efficiency of the
reasoning process. On the other hand, for traditional MHA or GQA operators, in
the process of calculating attention, all data in KV Cache is only involved in
one or several calculations after reading, resulting in extremely low MFU of
the operator, and since each request has its own KV Cache, this problem cannot
be solved by increasing the batch size. The MLA operator, from the perspective
of its computational characteristics, solves both problems at the same time: on
the one hand, the size of KV Cache is greatly reduced through low-rank
compression, and on the other hand, the multi-head attention mechanism after
MLA decompression can provide higher computing intensity, which helps to make
full use of the computing power resources of the GPU. Obviously, the MLA
operator is an attention mechanism tailored to the characteristics of modern
GPU hardware. By rebalancing storage and computing, it can give full play to
the advantages of modern GPUs.

The open source code of DeepSeek-V2 does not optimize the MLA operator too
much. We tried to reproduce some optimization points that the MLA operator may
involve in the inference phase (specifically, the decoding phase of the
inference phase), and evaluated and analyzed them.

The address of all the codes involved in this article is:
https://github.com/madsys-dev/deepseekv2-profile

## Calculation process of MLA module

Given an input vector $h_t \in \mathbb{R}^{B \times L \times 5120}$, where $B$
is the batch size and $L$ is the sequence length, the calculation process of
MLA is as follows.

### Q vector

In DeepSeek-V2, the Q vector is also compressed in a low-rank manner. First,
the input vector is projected into a 1536-dimensional low-dimensional space:
$$ c_t^Q = W^{DQ} h_t \in \mathbb{R}^{B \times L \times 1536} $$
Then, project it onto the multi-head vector space of $\mathbb{R}^{H \times
128}$ (where $H=128$ is the number of heads), and get the first part of the Q
vector:
$$ q_t^C = W^{UQ} c_t^Q \in \mathbb{R}^{B \times L \times H \times 128} $$
Then project it onto $\mathbb{R}^{H \times 64}$ and use RoPE to embed the
position information to get the second part of the Q vector:
$$ q_t^R = \mathrm{RoPE}(W^{KR} h_t) \in \mathbb{R}^{B \times L \times H \times 64} $$
Concatenate the two parts into the final Q vector:
$$ q_t = [q_t^C, q_t^R] \in \mathbb{R}^{B \times L \times H \times 192} $$

### KV vector

When calculating the KV vector, you first need to project the input vector into
a 512-dimensional joint compressed representation:
$$ c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{B \times L \times 512} $$

Similar to the calculation process of Q vector, the first part of K vector is
to decompress $c_t^{KV}$ into the multi-headed vector space of $\mathbb{R}^{H
\times 128}$ by projection:
$$ k_t^C = W^{UK} c_t^{KV} \in \mathbb{R}^{B \times L \times H \times 128} $$
The second part of K is to project the input vector into a 64-dimensional
vector space and apply RoPE to embed the position information:
$$ k_t^R = \mathrm{RoPE}(W^{KR} h_t) \in \mathbb{R}^{B \times L \times 64} $$
Unlike Q, the complete K is obtained by broadcasting the second part of K to
each head and concatenating it with the first part:
$$ k_t = \begin{bmatrix}
    k_{t,1}^C & k_t^R \\
    k_{t,2}^C & k_t^R \\
    \vdots & \vdots \\
    \end{bmatrix} \in \mathbb{R}^{B \times L \times H \times 192} $$
That is, the RoPE part of each head is exactly the same.

The calculation of the V vector is relatively simple. Simply decompress
$c_t^{KV}$ to $\mathbb{R}^{H \times 128}$:
$$ v_t = W^{UV} c_t^{KV} \in \mathbb{R}^{B \times L \times H \times 128} $$

### Attention calculation

The calculation process of Attention is no different from the traditional MHA.
First, calculate the attention score:
$$ a = \mathrm{softmax}\left(\frac{q_t^\top k_t + \mathrm{Mask}}{\sqrt{192}}\right) =
\mathrm{softmax}\left(\frac{{q_t^C}^\top k_t^C + {q_t^R}^\top k_t^R + \mathrm{Mask}}{\sqrt{128 + 64}} \right)
\in \mathbb{R}^{B \times L \times H \times L} $$
Calculate the weighted sum of V and flatten all heads to get the Attention output:
$$ o = a \cdot v_t \in \mathbb{R}^{B \times L \times H \times 128} \cong \mathbb{R}^{B \times L \times 16384} $$
After another matrix projection, we can get the final output of MLA:
$$ u = W^O o \in \mathbb{R}^{B \times L \times 5120} $$

## Open Source MLA Analysis

``` Python
def forward(...):
    bsz, q_len, _ = hidden_states.size()
    
    # Calculate Q: Reduce the dimension first and then increase the dimension. The advantage is that compared with directly using a matrix of size [5120, 24576]
    # [5120, 1536] * [1536, 24576] Such low-rank decomposition greatly reduces the storage space and computational complexity.
    q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    # Split rope and non-rope parts
    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )
    
    # Calculate KV
    # An optimized MLA KVCache implementation only needs to cache this compressed_kv, but it is actually expanded later
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    # Here compressed_kv corresponds to c_t^{KV} in the formula
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    # Expand MLA to standard MHA form
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        .transpose(1, 2)
    )
    # Because kv_b_proj packs W^{UK} and W^{UV}, separate them
    k_nope, value_states = torch.split(
        kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )
    ...
    # Add rope to the part that needs rope
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
    
    # Update and concatenate historical KVCache. You can see that the expanded MHA KVCache is stored here.
    # where q_head_dim is equal to qk_nope_head_dim + qk_rope_head_dim
    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
    key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos} # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # The following is the standard MHA code, no further explanation
    ...
```

## Implementation optimization of MLA module

KV Caching

In the decoding process of the original transformer model, each iteration needs
to calculate the KV vectors corresponding to all tokens, which often brings a
large overhead. In fact, the values ​​of these KV vectors are the
same in each iteration; therefore, we can adopt the strategy of "trading space
for time" to cache the values ​​of KV vectors in the previous
iteration, so that in the subsequent iterations, there is no need to repeatedly
calculate the KV vectors, thereby greatly reducing the amount of calculation in
the model inference process.

However, in traditional Attention operators represented by MHA, this strategy
of exchanging space for time often goes too far. Since the KV cache occupies a
large space and the data in the KV cache is only involved in calculations once
in each iteration, after using the KV cache, although the amount of calculation
is reduced, the amount of video memory occupied and the demand for video memory
bandwidth have increased sharply, becoming a new bottleneck restricting the
efficiency of large model reasoning. The design of MLA greatly reduces the
occupancy of the KV cache by sharing the compressed KV representation with
multiple heads. On the other hand, since the Compressed KV is involved in the
calculation in each head, the 128 heads of DeepSeek-V2 can provide sufficient
computing intensity, so the MFU of the Attention part has also been greatly
improved.

In the open source version, MLA operator caches the complete KV Cache, losing
the above benefits of MLA. We try to cache the compressed KV Cache instead and
compare it with caching the complete KV Cache. Of course, here we also cache
the k_pe after RoPE into the KV Cache.

``` Python
# CacheCompressed
def forward(self, hidden_states_q: torch.Tensor, q_position_ids: torch.LongTensor, compressed_kv: torch.Tensor):
    ...
    kv_seq_len = compressed_kv.size(1)
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    k_pe = k_pe.view(bsz, kv_seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = self.kv_b_proj(compressed_kv) \
        .view(bsz, kv_seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim) \
        .transpose(1, 2)
    
    k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    ...
```

The KV Cache occupancy and computational effort of the two implementations are
shown in the following table:

| Implementation version | Cache size per token per layer | Computation amount per token per layer |
| :---: | :---: | :---: |
| CacheDecompressed (CD) | 81.92 kB | 0.08 MFLOP |
| CacheCompressed (CC) | 1.152 kB | 33.64 MFLOPs |

As we can see, although the CacheDecompressed strategy can save almost all
floating-point calculations, its video memory usage reaches 81.92kB per token.
This makes it easy for the bottleneck of CacheDecompressed to be stuck on video
memory capacity and video memory bandwidth. However, the video memory usage of
CacheCompressed is reduced by about 98.6%. Therefore, we can expect that the
CacheCompressed strategy can make more balanced use of the hardware
capabilities of the GPU and provide a larger batch size, thereby reducing the
inference cost.

We tested the performance of the above implementations on A100-PCIe-40G
(Compute80 architecture) and GeForce RTX 4080 (Compute89 architecture). For a
single request, the performance of various implementations is shown in the
following figure:

![](data/caching-B1.png)

The performance of CacheDecompressed is significantly better than that of
CacheCompressed. This shows that the CacheCompressed strategy needs to be
further optimized to reduce the amount of calculation per token in order to
achieve better performance.

When Batch Size=32, the performance of each implementation is shown in the
following figure:

![](data/caching-B32.png)

The test results are basically the same as those for a single query.

Projection Absorption

The above analysis and experimental results show that compared with caching the
complete KV Cache, caching the compressed KV Cache will lead to a significant
performance degradation. Another important issue is that the current
CacheCompressed implementation does not actually alleviate the problem of KV
Cache being too large, because when calculating MLA, it is still necessary to
store the decompressed complete KV Cache, which is likely to cause OOM crashes.

Fortunately, the DeepSeek-V2 paper proposes that the decompressed matrix of KV
can be absorbed into Q-projection and Out-projection, so that the final
Attention result can be directly calculated without decompressing the KV Cache.
For the absorption of K, in the calculation formula of Attention Score, the
non-RoPE part can be expanded as follows:
$$
{q_t^C}^\top k_t^C = (W^{UQ} c_t^Q)^{\top} W^{UK} c_t^{KV} = {c_t^Q}^{\top}{W^{UQ}}^{\top} W^{UK} c_t^{KV} = ({c_t^Q}^{\top}{W^{UQ}}^{\top} W^{UK}) c_t^{KV}
$$
That is, through the associative law of matrix multiplication, we can calculate
$({c_t^Q}^{\top}{W^{UQ}}^{\top} W^{UK})$ instead, avoiding the need to
decompress the complete K matrix. In addition, in the decompression process of
the original version, since the key of each token needs to be multiplied with
$W^{UK}$ to obtain it, the amount of calculation is large; after matrix
absorption, $W^{UK}$ only needs to multiply the vector $q_t^C$, which also
greatly reduces the amount of floating-point calculations.

For the absorption of V, the situation is slightly more complicated. For the
sake of clarity, we use the Einstein summation convention to describe the
process:
``` Python
v_t = einsum('hdc,blc->blhd', W_UV, c_t_KV) # (1)
o = einsum('bqhl,blhd->bqhd', a, v_t) # (2)
u = einsum('hdD,bhqd->bhD', W_o, o) # (3)

# Combine the above three formulas to get the overall calculation process
u = einsum('hdc,blc,bqhl,hdD->bhD', W_UV, c_t_KV, a, W_o)

# Use associative law to change the order of calculation
o_ = einsum('bhql,blc->bhqc', a, c_t_KV) # (4)
o = einsum('bhqc,hdc->bhqd', o_, W_UV) # (5)
u = einsum('hdD,bhqd->bhD', W_o, o) # (6)
```

The specific code implementation is as follows:
``` Python
#Absorbed_CacheCompressed
def forward(hidden_states_q: torch.Tensor, q_position_ids: torch.LongTensor, compressed_kv: torch.Tensor):
    ...
    kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
    q_absorb = kv_b_proj[:, :self.qk_nope_head_dim,:]
    out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :]
    
    cos, sin = self.rotary_emb(q_pe)
    q_pe = apply_rotary_pos_emb(q_pe, cos, sin, q_position_ids)
    
    qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, qk_head_dim)
    # The calculation order of q_nope is changed here
    query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
    query_states[:, :, :, self.kv_lora_rank :] = q_pe
    
    ...

    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(q_nope.dtype)
    # The calculation order of attn_output is changed here
    attn_output = torch.einsum('bhql,blc->bhqc', attn_weights, compressed_kv)
    attn_output = torch.einsum('bhqc,hdc->bhqd', attn_output, out_absorb)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
            f"{attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
    attn_output = self.o_proj(attn_output)
```

#### Move Elision
However, this does not fully demonstrate the power of MLA. In the original
code, query_states and key_states are obtained by concatenating the RoPE and
non-RoPE parts:
``` Python
def forward(...):
    ...
    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

    key_states = k_pe.new_empty(bsz, self.num_heads, kv_seq_len, self.q_head_dim)
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
    ...
```
After we adopt the above optimization, the splicing process here will generate
a lot of useless data copying and broadcasting, and will also occupy a lot of
video memory space and cause OOM. To this end, we adopt the MoveElision
optimization strategy.
That is, the process of concatenating the RoPE part and the non-RoPE part is
omitted here, and the amount of attention score of the amount part is directly
calculated and added (considering $q_t^\top k_t = {q_t^C}^\top k_t^C +
{q_t^R}^\top k_t^R$):
``` Python
#Absorbed_CacheCompressed_MoveElision
def forward(...):
    ...
    # qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
    # query_states = k_pe.new_empty(bsz, self.num_heads, q_len, qk_head_dim)
    # query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
    # query_states[:, :, :, self.kv_lora_rank :] = q_pe

    # key_states = k_pe.new_empty(bsz, self.num_heads, kv_seq_len, qk_head_dim)
    # key_states[:, :, :, : self.kv_lora_rank] = compressed_kv.unsqueeze(1)
    # key_states[:, :, :, self.kv_lora_rank :] = k_pe

    # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

    attn_weights = torch.matmul(q_pe, k_pe.transpose(2, 3)) + torch.einsum('bhqc,blc->bhql', q_nope, compressed_kv)
    attn_weights *= self.softmax_scale
    ...
```

In this way, we get the following four versions of optimized implementation:

| Implementation version | Cache size per token per layer | Computation amount per token per layer |
| :---: | :---: | :---: |
| CacheDecompressed (CD) | 81.92 kB | 0.08 MFLOP |
| CacheCompressed (CC) | 1.152 kB | 33.64 MFLOPs |
| Absorbed_CacheCompressed (A_CC) | 1.152 kB | 0.28 MFLOP |
| Absorbed_CacheCompressed_MoveElision (A_CC_ME) | 1.152 kB | 0.28 MFLOP |

The test results on A100-PCIe-40G and GeForce RTX 4080 are as follows, which
are exactly consistent with the theoretical analysis.

![](data/absorption-B1-annotated.png)

![](data/absorption-B32-annotated.png)

It is worth noting that when the MoveElision strategy is adopted, the batch
size and sequence length that can be processed are significantly increased due
to the reduction in video memory usage, which fully reflects the advantages of
MLA's compressed representation.

#### Materializing Projection Matrices?

The DeepSeek-V2 paper says:
> ..., we can absorb $W^{UK}$ into $W^{UQ}$, and $W^{UV}$ into $W^O$.

However, it seems unnecessary to change the order, preprocess the model
parameters, multiply $W^{UK}$ by $W^{UQ}$, and multiply $W^{UV}$ by $W^O$. This
is because the result of multiplying $W^{UK}$ by $W^{UQ}$ can be regarded as
$H$ low-rank (no more than 128) matrices of size $1536 \times 512$, and the
result of multiplying $W^{UV}$ by $W^O$ can be regarded as $H$ low-rank
matrices of size $5120 \times 512$. Compared with using these particularly
large low-rank matrices for projection, it is obviously not as cost-effective
as multiplying them in sequence according to the low-rank decomposition form.
Therefore, we believe that this step of optimization is not very necessary.

We implemented this optimized version (AM_CC_ME) and tested it, and the test
results confirmed our point of view.

![](data/am-B1-annotated.png)

![](data/am-B32-annotated.png)

The performance of this optimization is significantly worse than the original
version, especially when the sequence length is small and the calculation time
of these projections dominates.

## Subsequent Optimization

The current code implementation is based on matrix multiplication, so the
attention score matrix needs to be fully calculated during the calculation
process. If further optimization is required, you can consider a similar
approach to FlashAttention, that is, reading the entire KV-pair at one time for
calculation. Since the K and V of MLA share the same compressed representation
(in fact, the above optimized MLA implementation is very similar to MQA that
satisfies $K=V$), this can further reduce video memory reading and improve
calculation intensity.


