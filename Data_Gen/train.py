"""
Phase 3: WGAN-GP Training Loop
Trains the conditional Wasserstein GAN on Base.csv data.

Training strategy:
- Critic:Generator ratio = 5:1
- Gradient penalty lambda = 10
- Adam optimizer (lr=1e-4, betas=(0.5, 0.9))
- 300 epochs with quality monitoring
- Saves best generator checkpoint
"""
import os
import sys
import time
import json

import torch
import torch.optim as optim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from model import Generator, Critic, gradient_penalty
from data_prep import load_prepared_data, prepare_data

# Hyperparameters
NOISE_DIM = 128
FEATURE_DIM = 51
NUM_DIM = 25
CAT_DIMS = [5, 7, 7, 2, 5]
LABEL_DIM = 1

EPOCHS = 300
CRITIC_ITERS = 5       # train critic 5x per generator step
LAMBDA_GP = 10
LR = 1e-4
BETAS = (0.5, 0.9)
BATCH_SIZE = 512

CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, 'checkpoints')
BEST_MODEL_PATH = os.path.join(SCRIPT_DIR, 'generator.pt')
TRAINING_LOG = os.path.join(SCRIPT_DIR, 'training_log.json')


def train_wgan():
    """Full WGAN-GP training loop."""
    print("=" * 60)
    print("  PHASE 3: WGAN-GP Training")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    # --- Load data ---
    print("\n[1/3] Loading training data...")
    if os.path.exists(os.path.join(SCRIPT_DIR, 'training_data.pt')):
        dataloader, transforms = load_prepared_data(batch_size=BATCH_SIZE)
        print(f"  Loaded from cache")
    else:
        dataloader, transforms = prepare_data(batch_size=BATCH_SIZE)

    feature_dim = transforms['total_feature_dim']
    print(f"  Feature dim: {feature_dim}")
    print(f"  Batches/epoch: {len(dataloader)}")

    # --- Initialize models ---
    print("\n[2/3] Initializing models...")
    gen = Generator(
        noise_dim=NOISE_DIM,
        label_dim=LABEL_DIM,
        num_dim=NUM_DIM,
        cat_dims=CAT_DIMS,
        feature_dim=feature_dim
    ).to(device)

    critic = Critic(
        feature_dim=feature_dim,
        label_dim=LABEL_DIM
    ).to(device)

    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=BETAS)
    opt_critic = optim.Adam(critic.parameters(), lr=LR, betas=BETAS)

    print(f"  Generator params:  {sum(p.numel() for p in gen.parameters()):,}")
    print(f"  Critic params:     {sum(p.numel() for p in critic.parameters()):,}")

    # --- Training loop ---
    print(f"\n[3/3] Training for {EPOCHS} epochs...")
    print(f"  Critic iters: {CRITIC_ITERS} | Lambda GP: {LAMBDA_GP}")
    print(f"  LR: {LR} | Betas: {BETAS}")
    print(f"  {'─' * 56}")
    print(f"  {'Epoch':>6} | {'D_loss':>8} | {'G_loss':>8} | {'W_dist':>8} | {'GP':>8} | Time")
    print(f"  {'─' * 56}")

    training_log = []
    best_w_dist = float('inf')
    start_time = time.time()

    # Get one real batch for quality comparison
    real_sample, real_labels = next(iter(dataloader))
    real_means = real_sample.mean(dim=0).numpy()

    for epoch in range(1, EPOCHS + 1):
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_gp = 0
        epoch_w_dist = 0
        n_critic_steps = 0
        n_gen_steps = 0

        for batch_idx, (real_data, labels) in enumerate(dataloader):
            real_data = real_data.to(device)
            labels = labels.to(device)
            batch_size = real_data.size(0)

            # ========== Train Critic ==========
            for _ in range(CRITIC_ITERS):
                critic.zero_grad()

                # Critic on real
                critic_real = critic(real_data, labels).mean()

                # Critic on fake
                noise = torch.randn(batch_size, NOISE_DIM, device=device)
                fake_data = gen(noise, labels).detach()
                critic_fake = critic(fake_data, labels).mean()

                # Gradient penalty
                gp = gradient_penalty(critic, real_data, fake_data, labels, device, LAMBDA_GP)

                # Critic loss: maximize (real - fake), i.e. minimize -(real - fake) + gp
                d_loss = -critic_real + critic_fake + gp

                d_loss.backward()
                opt_critic.step()

                w_dist = (critic_real - critic_fake).item()
                epoch_d_loss += d_loss.item()
                epoch_gp += gp.item()
                epoch_w_dist += w_dist
                n_critic_steps += 1

            # ========== Train Generator ==========
            gen.zero_grad()

            noise = torch.randn(batch_size, NOISE_DIM, device=device)
            fake_data = gen(noise, labels)
            critic_fake = critic(fake_data, labels).mean()

            # Generator loss: maximize critic_fake, i.e. minimize -critic_fake
            g_loss = -critic_fake
            g_loss.backward()
            opt_gen.step()

            epoch_g_loss += g_loss.item()
            n_gen_steps += 1

        # Epoch averages
        avg_d = epoch_d_loss / max(n_critic_steps, 1)
        avg_g = epoch_g_loss / max(n_gen_steps, 1)
        avg_gp = epoch_gp / max(n_critic_steps, 1)
        avg_w = epoch_w_dist / max(n_critic_steps, 1)
        elapsed = time.time() - start_time

        log_entry = {
            'epoch': epoch,
            'd_loss': round(avg_d, 4),
            'g_loss': round(avg_g, 4),
            'w_dist': round(avg_w, 4),
            'gp': round(avg_gp, 4),
            'time': round(elapsed, 1),
        }
        training_log.append(log_entry)

        # Print progress
        if epoch <= 10 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"  {epoch:>6} | {avg_d:>+8.4f} | {avg_g:>+8.4f} | {avg_w:>8.4f} | {avg_gp:>8.4f} | {elapsed:.0f}s")

        # Quality check every 50 epochs
        if epoch % 50 == 0:
            gen.eval()
            with torch.no_grad():
                noise = torch.randn(1000, NOISE_DIM, device=device)
                test_labels = torch.zeros(1000, 1, device=device)
                fake_sample = gen(noise, test_labels)
                fake_means = fake_sample.cpu().mean(dim=0).numpy()

                # Compare first 5 numerical feature means
                print(f"         Quality check — feature mean comparison (first 5 numerical):")
                for i in range(5):
                    diff = abs(real_means[i] - fake_means[i])
                    status = "OK" if diff < 0.1 else "DRIFT"
                    print(f"           col{i}: real={real_means[i]:.3f} fake={fake_means[i]:.3f} diff={diff:.3f} {status}")
            gen.train()

        # Save best model (lowest absolute Wasserstein distance = most stable)
        if epoch > 50 and abs(avg_w) < best_w_dist:
            best_w_dist = abs(avg_w)
            torch.save({
                'epoch': epoch,
                'generator_state': gen.state_dict(),
                'critic_state': critic.state_dict(),
                'w_dist': avg_w,
            }, BEST_MODEL_PATH)

    # Save final model too
    torch.save({
        'epoch': EPOCHS,
        'generator_state': gen.state_dict(),
        'critic_state': critic.state_dict(),
        'w_dist': avg_w,
    }, BEST_MODEL_PATH)

    # Save training log
    with open(TRAINING_LOG, 'w') as f:
        json.dump(training_log, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Final W-distance: {avg_w:.4f}")
    print(f"  Best W-distance: {best_w_dist:.4f}")
    print(f"  Saved: {os.path.basename(BEST_MODEL_PATH)}")
    print(f"  Log: {os.path.basename(TRAINING_LOG)}")

    return gen


if __name__ == '__main__':
    train_wgan()
