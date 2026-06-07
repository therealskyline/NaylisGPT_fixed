import torch
import torch.nn.functional as F

def test_attention_leakage():
    # Simulate a packed sequence: [doc1_tok1, doc1_tok2, doc2_tok1, doc2_tok2]
    # Length = 4
    # We want to ensure doc2 tokens don't attend to doc1 tokens.

    batch_size = 1
    seq_len = 4
    num_heads = 1
    head_dim = 8

    q = torch.ones(batch_size, num_heads, seq_len, head_dim)
    k = torch.ones(batch_size, num_heads, seq_len, head_dim)
    v = torch.ones(batch_size, num_heads, seq_len, head_dim)

    # Define document boundaries (cu_seqlens): [0, 2, 4]
    # doc1: 0 to 2
    # doc2: 2 to 4

    # Case 1: SDPA with is_causal=True (current implementation fallback)
    # SDPA with is_causal=True uses a simple causal mask.
    # It doesn't know about document boundaries.

    output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Check if doc2 tokens (index 2, 3) attended to doc1 tokens (index 0, 1)
    # If they did, the attention weights would be distributed across all preceding tokens.
    # We can check this by making keys different.

    k[0, 0, :2] = 0.0 # doc1 keys are 0
    k[0, 0, 2:] = 1.0 # doc2 keys are 1
    v[0, 0, :2] = 10.0 # doc1 values are 10
    v[0, 0, 2:] = 1.0  # doc2 values are 1

    output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    print(f"Output for doc2 token (pos 2): {output[0, 0, 2, 0].item()}")

    # If it attended to doc1, the value will be > 1.0 (average of 10 and 1)
    if output[0, 0, 2, 0].item() > 1.0:
        print("FAILURE: Attention leakage detected! Doc 2 attends to Doc 1 via causal mask.")
    else:
        print("SUCCESS: No leakage (unlikely with just is_causal=True)")

if __name__ == "__main__":
    test_attention_leakage()
