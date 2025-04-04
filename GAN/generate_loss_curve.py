import matplotlib.pyplot as plt
import numpy as np

# Loss values from the GAN training
epochs = list(range(1, 51))
discriminator_loss = [
    0.0391,
    0.0247,
    0.2784,
    0.3516,
    0.7943,
    0.8930,
    0.9027,
    0.4572,
    0.6267,
    0.9494,
    1.3591,
    0.4286,
    0.2814,
    0.7245,
    0.8021,
    0.6389,
    0.3148,
    0.2203,
    0.0330,
    0.1397,
    0.0679,
    0.3811,
    0.1854,
    0.3040,
    0.2594,
    0.2726,
    0.3582,
    0.1518,
    0.9023,
    0.2845,
    0.4088,
    0.3299,
    0.5395,
    0.4631,
    0.4029,
    0.4081,
    0.6955,
    0.5181,
    0.5107,
    0.7191,
    0.8644,
    0.3761,
    0.5430,
    0.4965,
    0.5782,
    0.5757,
    0.3068,
    0.4417,
    0.3763,
    0.7164,
]

generator_loss = [
    6.5052,
    10.2173,
    7.0374,
    9.4681,
    3.9339,
    2.4434,
    1.7047,
    2.3680,
    2.1660,
    1.4111,
    0.9439,
    2.1348,
    3.0554,
    3.0511,
    2.3209,
    5.6900,
    4.9226,
    6.1628,
    6.8429,
    4.8779,
    10.8177,
    7.2194,
    4.9723,
    6.3438,
    5.3354,
    5.6151,
    3.1430,
    5.0258,
    4.4585,
    4.1478,
    6.3521,
    3.1381,
    3.9585,
    4.3529,
    4.0678,
    4.1831,
    4.0042,
    3.5277,
    4.6456,
    4.1482,
    4.2930,
    3.4970,
    2.9410,
    3.1774,
    4.6992,
    3.0882,
    3.3397,
    3.5201,
    2.7461,
    4.1095,
]

# Create the figure and axis
plt.figure(figsize=(12, 6))

# Plot the loss curves
plt.plot(epochs, discriminator_loss, "b-", label="Discriminator Loss")
plt.plot(epochs, generator_loss, "r-", label="Generator Loss")

# Add a horizontal line at y=0.693 (log(0.5)) to represent theoretical equilibrium
plt.axhline(
    y=0.693, color="g", linestyle="--", label="Theoretical Equilibrium (log(0.5))"
)

# Add labels and title
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN Training: Discriminator and Generator Loss")
plt.grid(True, alpha=0.3)
plt.legend()

# Add annotations to highlight key training phases
plt.annotate(
    "Early Training\n(D dominates)",
    xy=(3, 7),
    xytext=(5, 8),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
)

plt.annotate(
    "Mid Training\n(G improves)",
    xy=(10, 1.5),
    xytext=(12, 3),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
)

plt.annotate(
    "Fluctuating Dynamics\n(Adversarial Balance)",
    xy=(30, 5),
    xytext=(35, 7),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
)

# Add a text box explaining the significance of the loss curves
textstr = "\n".join(
    (
        "GAN Training Dynamics:",
        "• High G loss, low D loss: Generator produces poor fakes",
        "• Decreasing G loss: Generator improving",
        "• Increasing D loss: Discriminator struggling",
        "• Fluctuations: Typical of adversarial training",
    )
)

props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
plt.text(
    0.02,
    0.98,
    textstr,
    transform=plt.gca().transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=props,
)

# Save the figure
plt.tight_layout()
plt.savefig("gan_loss_curve.png", dpi=300)
plt.close()

print("Loss curve visualization saved as 'gan_loss_curve.png'")
