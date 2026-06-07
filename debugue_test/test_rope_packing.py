import torch
from naylisgdn.rope import RotaryPositionalEmbedding

def test_rope_packing():
    dim = 64
    max_seq_len = 1024
    rope = RotaryPositionalEmbedding(dim=dim, max_seq_len=max_seq_len)

    # Simulate a packed sequence with two documents: doc1 (len 4) and doc2 (len 4)
    # Total seq_len = 8
    q = torch.ones(1, 1, 8, dim)
    k = torch.ones(1, 1, 8, dim)

    # Apply RoPE
    q_out, k_out = rope(q, k)

    # In a correctly handled packing, the first token of doc1 (index 0)
    # and the first token of doc2 (index 4) should have the same RoPE transformation
    # if we want them to start at position 0.

    pos0_emb = q_out[0, 0, 0]
    pos4_emb = q_out[0, 0, 4]

    print(f"Embedding at pos 0: {pos0_emb[:4]}...")
    print(f"Embedding at pos 4: {pos4_emb[:4]}...")

    if torch.allclose(pos0_emb, pos4_emb):
        print("SUCCESS: RoPE positions are reset (unexpected given current code)")
    else:
        print("FAILURE: RoPE positions are NOT reset for packed sequences.")
        print("This confirms a silent ML risk: document boundaries are ignored by RoPE.")

if __name__ == "__main__":
    test_rope_packing()
