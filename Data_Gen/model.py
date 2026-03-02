"""
Phase 2: WGAN-GP Model Architecture
Conditional Wasserstein GAN with Gradient Penalty for tabular fraud data.

Generator:  noise(128) + label(1) → 4 layers → 51-dim feature vector
Critic:     features(51) + label(1) → 4 layers → scalar (Wasserstein score)
"""
import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Conditional Generator.
    Takes random noise + fraud label → produces realistic 51-dim feature vector.
    
    Output split:
      - First num_dim values → Sigmoid (numerical features in [0,1])
      - Remaining cat_dim values → grouped Softmax per categorical column
    """

    def __init__(self, noise_dim=128, label_dim=1, num_dim=25, cat_dims=None, feature_dim=51):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_dim = num_dim
        self.cat_dims = cat_dims or [5, 7, 7, 2, 5]  # payment, employment, housing, source, device
        self.feature_dim = feature_dim

        input_dim = noise_dim + label_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, feature_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, noise, labels):
        """
        Args:
            noise: (batch, noise_dim)
            labels: (batch, 1) — fraud label
        Returns:
            features: (batch, feature_dim) — numerical sigmoid + categorical softmax
        """
        x = torch.cat([noise, labels], dim=1)
        x = self.net(x)

        # Split output into numerical and categorical parts
        num_part = torch.sigmoid(x[:, :self.num_dim])

        # Apply softmax per categorical group
        cat_parts = []
        offset = self.num_dim
        for dim in self.cat_dims:
            cat_logits = x[:, offset:offset + dim]
            cat_parts.append(torch.softmax(cat_logits, dim=1))
            offset += dim

        return torch.cat([num_part] + cat_parts, dim=1)


class Critic(nn.Module):
    """
    Wasserstein Critic (no sigmoid — outputs raw score).
    Takes features + fraud label → scalar score.
    Higher score = more "real-looking".
    """

    def __init__(self, feature_dim=51, label_dim=1):
        super().__init__()

        input_dim = feature_dim + label_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),  # raw Wasserstein score
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features, labels):
        """
        Args:
            features: (batch, feature_dim)
            labels: (batch, 1)
        Returns:
            score: (batch, 1)
        """
        x = torch.cat([features, labels], dim=1)
        return self.net(x)


def gradient_penalty(critic, real_data, fake_data, labels, device='cpu', lambda_gp=10):
    """
    Compute gradient penalty for WGAN-GP.
    Interpolates between real and fake samples, then penalizes
    critic gradients that deviate from norm 1.

    Args:
        critic: Critic model
        real_data: (batch, feature_dim)
        fake_data: (batch, feature_dim)
        labels: (batch, 1)
        device: torch device
        lambda_gp: penalty coefficient

    Returns:
        gradient penalty loss (scalar)
    """
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_data)

    interpolated = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

    critic_interpolated = critic(interpolated, labels)

    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()

    return penalty


if __name__ == '__main__':
    # Quick architecture verification
    print("=" * 60)
    print("  PHASE 2: WGAN-GP Architecture Verification")
    print("=" * 60)

    NOISE_DIM = 128
    FEATURE_DIM = 51
    BATCH = 16

    gen = Generator(noise_dim=NOISE_DIM, feature_dim=FEATURE_DIM)
    crit = Critic(feature_dim=FEATURE_DIM)

    # Test generator
    noise = torch.randn(BATCH, NOISE_DIM)
    labels = torch.randint(0, 2, (BATCH, 1)).float()
    fake = gen(noise, labels)
    print(f"\n  Generator:")
    print(f"    Input:  noise={noise.shape}, labels={labels.shape}")
    print(f"    Output: {fake.shape}")
    print(f"    Range:  [{fake.min():.4f}, {fake.max():.4f}]")
    print(f"    Params: {sum(p.numel() for p in gen.parameters()):,}")

    # Check numerical part is in [0,1]
    num_part = fake[:, :25]
    assert num_part.min() >= 0 and num_part.max() <= 1, "Numerical features should be in [0,1]"
    print(f"    Numerical [0,1]: OK")

    # Check categorical softmax sums to 1
    cat_dims = [5, 7, 7, 2, 5]
    offset = 25
    for i, dim in enumerate(cat_dims):
        cat_slice = fake[:, offset:offset + dim]
        sums = cat_slice.sum(dim=1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5), f"Cat {i} softmax broken"
        offset += dim
    print(f"    Categorical softmax: OK")

    # Test critic
    score_real = crit(fake.detach(), labels)
    print(f"\n  Critic:")
    print(f"    Input:  features={fake.shape}, labels={labels.shape}")
    print(f"    Output: {score_real.shape}")
    print(f"    Score:  {score_real.mean():.4f}")
    print(f"    Params: {sum(p.numel() for p in crit.parameters()):,}")

    # Test gradient penalty
    real_data = torch.randn(BATCH, FEATURE_DIM).clamp(0, 1)
    gp = gradient_penalty(crit, real_data, fake.detach(), labels)
    print(f"\n  Gradient Penalty: {gp.item():.4f}")

    print(f"\n{'=' * 60}")
    print(f"  All checks passed! Architecture ready for training.")
    print(f"{'=' * 60}")
