# GAN from Scratch — PyTorch MNIST Example 🧠✨

This repository demonstrates how to implement and train a Generative Adversarial Network (GAN) from scratch using **PyTorch**, with a focus on generating handwritten digits from the **MNIST** dataset. **[Refer to this blog post for more information - CloudAIApp.dev](https://cloudaiapp.dev/understanding-gans-how-machines-learn-to-create/)**

---

## 📂 Project Structure

```
├── GAN.ipynb                # Main notebook for training and visualizing the GAN
├── activation_functions.py  # Visualization of common activation functions used in GANs
├── generate_loss_curve.py   # Script to generate loss curve plots from saved logs
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/azure-bk-solutions/python-projects.git
cd python-projects/GAN
```

### 2. Set up the environment
Make sure you have Python 3.7+ and install dependencies:
```bash
pip install torch torchvision matplotlib numpy
```

---

## 🧪 Training the GAN

Open the Jupyter notebook:
```bash
jupyter notebook GAN.ipynb
```

- It trains a basic GAN on MNIST
- Generates sample digits after every epoch
- Saves visual outputs in the `gan_samples/` folder

---

## 📈 Visualizing Training

To generate loss curves from training logs:
```bash
python generate_loss_curve.py
```

---

## 🧠 Activation Functions

You can run `activation_functions.py` to visualize how different activations (ReLU, LeakyReLU, Tanh, etc.) behave.

```bash
python activation_functions.py
```

---

## 📸 Sample Output

- [Activation Functions](images/activation_functions.png)
- [Discriminator and Generator losses are tracked and visualized](images/gan_loss_curve.png)
- [Example animation of training evolution](images/gan_training.gif)

---

## 📎 References

- [Ian Goodfellow’s Original GAN Paper](https://arxiv.org/abs/1406.2661)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

---

## 🧠 Author

Crafted by [Balaji Kithiganahalli](https://github.com/azure-bk-solutions)

