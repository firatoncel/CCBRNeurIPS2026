"""SVD-factorise the full-rank item-item matrices of EASE and EDLAE (FR).

autoencoders.py stores the full items x items matrix B in enc_<model>.pt. This
script writes an encoder/decoder pair of shape (n_items, rank) next to it,
enc_<model>_svd.pt (U S) and dec_<model>_svd.pt (V), which the second stage
(main_ddp_movie_ae.py) loads as a linear backbone.

    python experiments/ML-20M/factorize.py --checkpoint-dir checkpoints/ml20m --rank 1024
"""
import argparse
import os

import torch


def factorize(BB, rank):
    U, S, V = torch.svd_lowrank(BB, q=rank, niter=5)
    UU = U @ torch.diag(S)   # (N, rank)
    VV = V                   # (N, rank)
    err = (BB - UU @ VV.T).norm() / BB.norm()
    print(f"  factorized to rank {rank}. Relative error: {err.item():.6f}")
    return UU, VV


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint-dir', default='checkpoints/ml20m')
    parser.add_argument('--models', nargs='+', default=['edlae_full', 'ease'])
    parser.add_argument('--rank', type=int, default=1024)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    for model in args.models:
        BB = torch.load(os.path.join(args.checkpoint_dir, f'enc_{model}.pt'),
                        map_location=args.device)
        print(f"{model}: B {tuple(BB.shape)}")
        encoder, decoder = factorize(BB, args.rank)
        torch.save(encoder.cpu(), os.path.join(args.checkpoint_dir, f'enc_{model}_svd.pt'))
        torch.save(decoder.cpu(), os.path.join(args.checkpoint_dir, f'dec_{model}_svd.pt'))
