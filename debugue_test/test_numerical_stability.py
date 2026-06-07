import torch
import torch.nn.functional as F
from naylisgdn.gdn_block import _gdn2_torch

def test_numerical_stability():
    B, T, H, Dk, Dv = 1, 128, 4, 32, 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Données de test
    q = torch.randn(B, T, H, Dk, device=device)
    k = torch.randn(B, T, H, Dk, device=device)
    v = torch.randn(B, T, H, Dv, device=device)
    g = -torch.rand(B, T, H, Dk, device=device) * 0.1 # decay léger
    b = torch.sigmoid(torch.randn(B, T, H, Dk, device=device))
    w = torch.sigmoid(torch.randn(B, T, H, Dv, device=device))

    # Référence en FP64 (très précis)
    out_ref, _ = _gdn2_torch(q.double(), k.double(), v.double(), g.double(), b.double(), w.double())

    # Test en BF16 (votre config actuelle)
    out_bf16, _ = _gdn2_torch(q.bfloat16(), k.bfloat16(), v.bfloat16(), g.bfloat16(), b.bfloat16(), w.bfloat16())

    # Test en FP32
    out_fp32, _ = _gdn2_torch(q.float(), k.float(), v.float(), g.float(), b.float(), w.float())

    diff_bf16 = (out_ref.float() - out_bf16.float()).abs().mean().item()
    diff_fp32 = (out_ref.float() - out_fp32.float()).abs().mean().item()

    print(f"Erreur moyenne BF16 vs FP64 : {diff_bf16:.8f}")
    print(f"Erreur moyenne FP32 vs FP64 : {diff_fp32:.8f}")
    print(f"Rapport d'erreur : {diff_bf16 / diff_fp32:.1f}x")

    if diff_bf16 > diff_fp32 * 10:
        print("\nRISQUE IDENTIFIÉ : La dérive numérique en BF16 est significative pour la récurrence GDN.")

if __name__ == "__main__":
    test_numerical_stability()
