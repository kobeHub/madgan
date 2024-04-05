import matplotlib.pyplot as plt


def plot_losses(g_loss, d_loss, img_path=None):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_loss, label="G")
    plt.plot(d_loss, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    if img_path:
        plt.savefig(img_path)
    plt.show()
