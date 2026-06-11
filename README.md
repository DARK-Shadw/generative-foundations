# Generative Modeling Foundations — Autoencoders, VAEs, GANs

The ground floor of my generative modeling work: autoencoders → VAEs → GANs → WGAN-GP, all implemented from scratch in PyTorch. (The next floors are in [diffusion-from-scratch](https://github.com/DARK-Shadw/diffusion-from-scratch) and [playable-diffusion](https://github.com/DARK-Shadw/playable-diffusion).)

## [`vae_gan_wgan_gp.ipynb`](./vae_gan_wgan_gp.ipynb)

- **VAE on MNIST** — reparameterization trick (`z = μ + σ·ε`), BCE reconstruction + KL divergence, and the loss-reduction subtleties that cause posterior collapse when you get them wrong
- **Convolutional VAE on anime faces** (64×64 RGB, 10K images) — strided conv encoder / transposed-conv decoder with BatchNorm, 128-dim latent; generates novel faces from `N(0,1)` and interpolates smoothly between them
- **Vanilla GAN** on the same dataset — and a front-row seat to mode collapse and discriminator/generator loss imbalance
- **WGAN-GP** — critic with Wasserstein loss and gradient penalty, plus the stability tricks that actually made adversarial training converge: label smoothing, training the critic on alternate batches, learning-rate balancing, input noise

## [`latent-space-explorer/`](./latent-space-explorer)

A small Flask app that makes latent spaces tangible: walk through a 2D autoencoder latent space with arrow keys and watch the decoder generate in real time. Built to demonstrate *why* plain autoencoders fail as generative models (the latent space is full of gaps) and why VAEs fix it. Dockerized for Hugging Face Spaces.
