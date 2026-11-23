"""SVD-factorise the full-rank item-item matrices of EASE and EDLAE (FR) for MSD.

autoencoders.py stores the full items x items matrix B in enc_<model>.pt. This
script writes an encoder of shape (n_items, rank) and a decoder of shape
(rank, n_items), split as U sqrt(S) and sqrt(S) V^T, to enc_<model>_svd.pt and
dec_<model>_svd.pt next to it. model_ddp_ae.py / model_ddp_vae.py load the
decoder transposed.

    python experiments/MSD/factorize.py --checkpoint-dir checkpoints/msd --rank 4096
"""
import argparse
import os
import time

import torch


def factorize_ease(B, rank=1024, tol=1e-6):
    # Cast to float32 for stable decomposition
    original_dtype = B.dtype
    B = B.to(torch.float32)

    asym = ((B - B.T).norm() / B.norm()).item()
    print(f"Asymmetry ratio: {asym:.2e}")

    if asym < tol:
        print("-> symmetric, using eigh")
        t0 = time.time()
        eigvals, eigvecs = torch.linalg.eigh(B)
        torch.cuda.synchronize()
        print(f"  eigh took {time.time() - t0:.1f}s")
        idx = torch.argsort(eigvals.abs(), descending=True)[:rank]
        L = eigvals[idx]
        Q = eigvecs[:, idx]
        sqrt_abs = L.abs().sqrt()
        encoder = Q * (sqrt_abs * L.sign())
        decoder = sqrt_abs.unsqueeze(1) * Q.T
        spectrum = eigvals.abs()
    else:
        print("-> not symmetric, using svd")
        t0 = time.time()
        U, S, Vh = torch.linalg.svd(B, full_matrices=False)
        torch.cuda.synchronize()
        print(f"  svd took {time.time() - t0:.1f}s")
        sqrt_S = S[:rank].sqrt()
        encoder = U[:, :rank] * sqrt_S
        decoder = sqrt_S.unsqueeze(1) * Vh[:rank, :]
        spectrum = S

    return encoder.to(original_dtype), decoder.to(original_dtype), spectrum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", default="checkpoints/msd")
    parser.add_argument("--models", nargs="+", default=["edlae_full", "ease"])
    parser.add_argument("--rank", type=int, default=4096)
    args = parser.parse_args()

    for model in args.models:
        t_start = time.time()
        BB = torch.load(os.path.join(args.checkpoint_dir, f"enc_{model}.pt")).cuda().to(torch.float32)
        print(f"{model}: B {tuple(BB.shape)}")
        encoder, decoder, S = factorize_ease(BB, args.rank)
        print(encoder.shape, decoder.shape)
        if encoder.shape == decoder.shape:
            decoder = decoder.T
        print(f"\nTotal time: {(time.time() - t_start) / 60:.2f} min")
        torch.save(encoder, os.path.join(args.checkpoint_dir, f"enc_{model}_svd.pt"))
        torch.save(decoder, os.path.join(args.checkpoint_dir, f"dec_{model}_svd.pt"))

        B_hat = encoder @ decoder
        err = ((BB - B_hat).norm() / BB.norm()).item()
        print(f"Reconstruction error: {err:.4f}")

        # Energy retained at multiple ranks
        total = (S**2).sum()
        for k in [256, 512, 1024, 2048, 4096]:
            energy = ((S[:k]**2).sum() / total).item()
            print(f"rank {k:5d}: {energy*100:.2f}% energy")
