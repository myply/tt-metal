
import ttnn
# print(ttnn.__file__)
from ttnn.transformer import scaled_dot_product_attention_decode
from ttnn.transformer import scaled_dot_product_attention_decodet
from ttnn.transformer import multi_head_latent_attention_decode0
import torch
import math
import sys
sys.path.append('/workspace/tt-metal/ttnn/tt_lib/_internal/')
from comparison_funcs import comp_pcc
# sys.path.append('/workspace/tt-metal/build/lib.linux-x86_64-3.10/tracy/')
# import tracy
def is_watcher_enabled():
    return os.environ.get("TT_METAL_WATCHER") is not None


def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def nearest_pow_2(x):
    if x < 1:
        raise ValueError("x must be >= 1")
    import math

    power = math.ceil(math.log2(x))
    return 1 << power
    # if (2**math.log2(x) == x):
    #     return x
    # return 2**(int(x).bit_length())


def num_to_corerange(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(num_x - 1, num_y - 1),
    )


def get_chunk_size(max_start_pos, s):
    if max_start_pos <= 32:
        chunk_size = 32
    elif max_start_pos <= 64:
        chunk_size = 32
    elif max_start_pos <= 128:
        chunk_size = 32
    elif max_start_pos <= 1024:
        chunk_size = 128
    else:
        chunk_size = 512
    # find maximum power of 2 divisor of s
    for i in range(1, s):
        if s % (2 ** (i + 1)) != 0:
            break
    chunk_size = min(chunk_size, 2**i)
    return chunk_size


def fa_rand(*shape):
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.bfloat16()
    key = key.bfloat16()
    value = value.bfloat16()
    # key = key.repeat_interleave(h_q // h_kv, dim=0)
    # value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.bfloat16)
    return attn_weight @ value, lse
def test_mla1(b=1,
    nh=4,
    nkv=1,
    s=4096,
    d=128,
    dv=256,
    dtype=ttnn.bfloat16 ,
    grid_size=(8, 8),
    q_dtype=ttnn.bfloat16,
    cur_pos_tensor=False,
    sharded_in=False,
    sharded_out=False,
    start_indices=None,
    causal=True,
    start_core=ttnn.CoreCoord(0, 0),
    sub_core_grids=None,
    override_q_chunk_size=None,
    override_k_chunk_size=None):
    device_id=0
    # MODIFIED(QXF)
    device = ttnn.open_device(device_id=device_id, dispatch_core_config=ttnn.device.DispatchCoreConfig(ttnn.device.DispatchCoreType.ETH))
    #device = ttnn.open_device(device_id=device_id)
    print(f"b : {b} nh : {nh} nkv : {nkv} s : {s} d : {d} dv : {dv} ")
    compute_grid_size = device.compute_with_storage_grid_size()
    #print(f"compute_grid_size : {compute_grid_size}")
    if sub_core_grids is None:
        if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
            print(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    else:
        unharvested_grid_size = (7, 10)
        if unharvested_grid_size[0] > compute_grid_size.x or unharvested_grid_size[1] > compute_grid_size.y:
            print(f"0Need {unharvested_grid_size} grid size to run this test but core grid is {compute_grid_size}")
        if grid_size[0] * grid_size[1] > sub_core_grids.num_cores():
            print(
                f"1Need {grid_size[0]*grid_size[1]} grid size to run this test but core grid is {sub_core_grids.num_cores()}"
            )
    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    num_parallel_cores = grid_size[0] * grid_size[1] // b
    if num_parallel_cores == 1:
        min_pcc = 0.90
    else:
        min_pcc = 0.99
        if q_dtype == ttnn.bfloat8_b:
            min_pcc = 0.98
        min_pcc = 0.91 if dtype == ttnn.bfloat4_b else min_pcc
    # min_pcc = 0.98
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

    if sub_core_grids is None:
        shard_grid = ttnn.CoreRangeSet({num_to_corerange(b)})
        compute_sub_core_grids = None
    else:
        shard_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(start_core, b, sub_core_grids, row_wise=True)
        compute_sub_core_grids = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            start_core, grid_size[0] * grid_size[1], sub_core_grids, row_wise=True
        )
    shard_spec = ttnn.ShardSpec(shard_grid, (padded_num_heads, d), ttnn.ShardOrientation.ROW_MAJOR)

    height_sharded_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    K = fa_rand(b, nkv, s, d)
    #V=fa_rand(b, nkv, s, dv)
    V = K[:,:,:,:dv]

    tt_K = ttnn.as_tensor(K, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)
    tt_V = ttnn.as_tensor(V, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)
    # MODIFIED(QXF)
    # start_indices = [s // 2 for _ in range(b)] if start_indices is None else start_indices
    #start_indices = [s for _ in range(b)] if start_indices is None else start_indices
    start_indices = [s-1 for _ in range(b)] if start_indices is None else start_indices
    max_start_idx = max(start_indices)
    scale = d**-0.5

    q_chunk_size = padded_num_heads if override_q_chunk_size is None else override_q_chunk_size
    k_chunk_size = get_chunk_size(max_start_idx + 1, s) if override_k_chunk_size is None else override_k_chunk_size
    print(f"k_chunk_size : {k_chunk_size}")
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        sub_core_grids=compute_sub_core_grids,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    # MODIFIED(QXF)
    padded_layer_len = nearest_n(max_start_idx + 1, n=k_chunk_size) if causal else s
    print(f"padded_layer_len : {padded_layer_len}")
    #padded_layer_len = s


    if causal:
        attn_mask = torch.zeros((b, nh, 1, padded_layer_len))
        for i in range(b):
            start_idx = start_indices[i]
            attn_mask[i, :, :, start_idx + 1 :] = torch.finfo(torch.float32).min
    else:
        attn_mask = torch.bernoulli(
            torch.full(
                (b, nh, 1, padded_layer_len),
                0.25,
            )
        )
        attn_mask = attn_mask * torch.finfo(torch.float32).min

    Q = fa_rand(1, b, nh, d)

    tt_Q = ttnn.as_tensor(
        Q[:, :, :nh],
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=height_sharded_memcfg if sharded_in else dram_memcfg,
    )
    if causal:
        if cur_pos_tensor:
            start_indices_tt = ttnn.Tensor(torch.tensor(start_indices), ttnn.int32).to(device)
            tt_back = ttnn.transformer.multi_head_latent_attention_decode1(
                tt_Q,
                tt_K,
                tt_V,
                cur_pos_tensor=start_indices_tt,
                scale=scale,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=height_sharded_memcfg if sharded_out else dram_memcfg,
            )
        else:
            print(start_indices)
            #tt_back = ttnn.transformer.scaled_dot_product_attention_decode(
            tt_back = ttnn.transformer.multi_head_latent_attention_decode1(
                tt_Q,
                tt_K,
                tt_V,
                cur_pos=start_indices,
                scale=scale,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=height_sharded_memcfg if sharded_out else dram_memcfg,
            )

    tt_back = ttnn.to_torch(tt_back)
    print(f"tt_back.shape : {tt_back.shape}")
    tt_back = tt_back[:, :, :nh, :]

    Q_slice = Q[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, d
    K_slice = K[:, :, :padded_layer_len, :]  # b, nkv, S, d
    K_slice = torch.cat(
        [K_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, S, d
    V_slice = V[:, :, :padded_layer_len, :]  # b, nkv, S, dv
    V_slice = torch.cat(
        [V_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, S, dv
    attn_mask_slice = attn_mask[:, :nh, :, :]  # b, nh, 1, S
    # print(f"Q_slice : {Q_slice.shape}")
    # print(f"K_slice : {K_slice.shape}")
    # print(f"V_slice : {V_slice.shape}")
    # print(f"attn_mask_slice : {attn_mask_slice.shape}")
    expect = torch.nn.functional.scaled_dot_product_attention(
        Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
    )  # b, nh, 1, d
    

    # ## test the for hand-write sdpa
    # out = torch.empty(b, nh, 1, dv, dtype=torch.bfloat16)
    # # lse = torch.empty(b, nh, s, dtype=torch.bfloat16)
    # for i in range(b):
    #     # begin = i * max_seqlen_pad
    #     # end = begin + cache_seqlens[i]
    #     O, LSE = scaled_dot_product_attention(
    #         Q_slice[i],
    #         K_slice[i][:, :start_indices[i], :],
    #         V_slice[i][:, :start_indices[i], :],
    #         # K_slice[i],
    #         # V_slice[i],
    #         h_q=nh,
    #         h_kv=nkv,
    #         is_causal=causal,
    #     )
    #     #out[i] = O.transpose(0, 1)
    #     out[i] = O
    #     # lse[i] = LSE
    # # print(f"expect : {expect.shape}")
    # # print(f"out : {out.shape}")
    # print(f"max  val : {out.abs().max()}")
    # print(f"mean val : {out.abs().mean()}")
    # print(f"max  err : {(expect-out).abs().max()}")
    # print(f"mean err : {(expect-out).abs().mean()}")
    # out_pass, out_pcc = comp_pcc(expect, out, min_pcc)
    # print(f"pass : {out_pass}! out_pcc : {out_pcc}")

    expect = expect.squeeze(2).unsqueeze(0)
    # print(f"expect.shape : {expect.shape}")
    non_skip_indices = torch.tensor(start_indices) != -1
    out_pass, out_pcc = comp_pcc(expect[:, non_skip_indices], tt_back[:, non_skip_indices], min_pcc)
    # print(f"max  val : {tt_back.abs().max()}")
    # print(f"mean val : {tt_back.abs().mean()}")
    # print(f"max  err : {(expect[:, non_skip_indices]-tt_back[:, non_skip_indices]).abs().max()}")
    # print(f"mean err : {(expect[:, non_skip_indices]-tt_back[:, non_skip_indices]).abs().mean()}")
    ttnn.DumpDeviceProfiler(device)
    ttnn.close_device(device)
    print(f"pass : {out_pass}! out_pcc : {out_pcc}")
    assert(out_pass)

def main():
    torch.manual_seed(1234)
    
    ## passed test 
    # for b in [1,4,16]:
    #     for s in [1024,8192]:
    #         for nh in [1,4,16]: 
    #             for nkv in [1]:
    #                 for d in [32,64,128]:
    #                     dv=d
    #                     test_mla1(b, nh, nkv, s, d, dv)
    # for b in [1,4,16]:
    #     for s in [1024,8192]:
    #         for nh in [1,4,16]: 
    #             for nkv in [1]:
    #                 for chunk_size in [32,64]:
    #                     for d in [256,512]:
    #                         dv=d
    #                         test_mla1(b, nh, nkv, s, d, dv,override_k_chunk_size=chunk_size)
    # for b in [1,4,16]:
    #     for s in [1024,8192]:
    #         for nh in [1,4,16]: 
    #             for nkv in [1]:
    #                 for d in [32,64,128]:
    #                     dv=d//2
    #                     test_mla1(b, nh, nkv, s, d, dv)
    # for b in [1,4,16]:
    #     for s in [1024,8192]:
    #         for nh in [1,4,16]: 
    #             for nkv in [1]:
    #                 for chunk_size in [32,64]:
    #                     for d in [256,512]:
    #                         dv=d//2
    #                         test_mla1(b, nh, nkv, s, d, dv,override_k_chunk_size=chunk_size)
    # for b in [1,4,16]:
    #     for s in [1024,8192]:
    #         for nh in [1,4,16]: 
    #             for nkv in [1]:
    #                 for d in [64,128]:
    #                     dv=d-32
    #                     test_mla1(b, nh, nkv, s, d, dv)
    # for b in [1,4,16]:
    #     for s in [1024,8192]:
    #         for nh in [1,4,16]: 
    #             for nkv in [1]:
    #                 for d in [256]:
    #                     for chunk_size in [32,64]:
    #                         dv=d-32
    #                         test_mla1(b, nh, nkv, s, d, dv,override_k_chunk_size=chunk_size)
    # benchmark
    # for b in [32]:
    #     for s in [16384]:
    #         for nh in [8]: 
    #             for nkv in [1]:
    #                 for d in [64]:
    #                     dv=d
    #                     test_mla1(b, nh, nkv, s, d, dv)
    # deepseek
    for b in [32]:
        for s in [16384]:
            for nh in [128]: 
                for nkv in [1]:
                    for d in [576]:
                        for chunk_size in [64]:
                            dv=d-64
                            test_mla1(b, nh, nkv, s, d, dv,override_k_chunk_size=chunk_size)


    
if __name__ == "__main__":
    main()