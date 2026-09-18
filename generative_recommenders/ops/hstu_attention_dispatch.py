from typing import Optional, Tuple
import importlib, importlib.util, torch

def _is_hopper() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 9  # H100=sm_90+

def _has_cpp_ext() -> bool:
    return importlib.util.find_spec("generative_recommenders.ops.cpp.hstu_attention") is not None

def hstu_attention_from_cache(
    *,
    impl: str,  # "auto" | "cpp" | "triton" | "ref"
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cached_q: Optional[torch.Tensor],
    cached_k: Optional[torch.Tensor],
    delta_x_offsets,
    x_offsets,
    all_timestamps,
    invalid_attn_mask,
    rel_attn_bias,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    choose_cpp = (impl == "cpp") or (impl == "auto" and _is_hopper() and _has_cpp_ext())
    if choose_cpp:
        try:
            cpp = importlib.import_module("generative_recommenders.ops.cpp.hstu_attention")
            print("[HSTU] Using attention backend: cpp")
            return cpp.from_cache(
                num_heads=num_heads,
                attention_dim=attention_dim,
                linear_dim=linear_dim,
                q=q, k=k, v=v,
                cached_q=cached_q, cached_k=cached_k,
                delta_x_offsets=delta_x_offsets,
                x_offsets=x_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                rel_attn_bias=rel_attn_bias,
            )
        except Exception as e:
            print(f"[HSTU] cpp backend unavailable, falling back ({type(e).__name__}): {e}")

    print("[HSTU] Using attention backend: triton/ref (fallback)")
    raise RuntimeError("fallback")
