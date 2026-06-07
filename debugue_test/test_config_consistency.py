import torch
try:
    import tomllib
except ImportError:
    import tomli as tomllib
from naylisgdn.model import NaylisGDN

def test_config_consistency():
    config_path = "config/1B.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    # Flatten config
    cfg = {}
    for section in config.values():
        if isinstance(section, dict):
            cfg.update(section)

    soft_cap = cfg.get("soft_cap")
    if soft_cap == 0:
        soft_cap = None

    model = NaylisGDN(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
        use_rope=cfg["use_rope"],
        rope_base=cfg.get("rope_base", 10000),
        use_yarn=cfg["use_yarn"],
        yarn_scale=cfg["yarn_scale"],
        yarn_original_max_len=cfg["yarn_original_max_len"],
        use_swiglu=cfg["use_swiglu"],
        n_kv_heads=cfg["n_kv_heads"],
        use_qk_norm=cfg["use_qk_norm"],
        soft_cap=soft_cap,
        use_flash_attn=cfg["use_flash_attn"],
        use_fp8=cfg.get("use_fp8", False),
        hybrid_ratio=cfg.get("hybrid_ratio", 3),
        gdn_head_dim=cfg.get("gdn_head_dim", None),
        gdn_v_heads=cfg.get("gdn_v_heads", None),
        gdn_qk_heads=cfg.get("gdn_heads_qk", None),
        attn_head_dim=cfg.get("attn_head_dim", None),
        rope_dim=cfg.get("rope_dim", None),
    )

    print(f"Model embed_dim: {model.embed_dim} (Expected: {cfg['embed_dim']})")
    print(f"Model gdn_head_dim: {model.gdn_head_dim} (Expected: {cfg['gdn_head_dim']})")
    print(f"Model attn_head_dim: {model.attn_head_dim} (Expected: {cfg['attn_head_dim']})")

    # Check GDN block heads
    gdn_block = model.blocks[0]
    print(f"GDN heads: {gdn_block.num_heads} (Expected: {cfg['gdn_heads_qk']})")
    print(f"GDN v_heads: {gdn_block.num_v_heads} (Expected: {cfg['gdn_v_heads']})")

    if gdn_block.num_heads == cfg['gdn_heads_qk'] and gdn_block.num_v_heads == cfg['gdn_v_heads']:
        print("SUCCESS: Config parameters correctly propagated.")
    else:
        print("FAILURE: Parameter mismatch detected!")

if __name__ == "__main__":
    test_config_consistency()
