import re
import glob
import matplotlib.pyplot as plt
from PIL import Image


class VisTrain():
    def __init__(self, hparam, log_dir):
        self.hparam = hparam
        self.vis_dir = log_dir + "/vis/"

    def vis_val_monolith(self, x, x_hat, epoch, mode):
        x, x_hat = x[0].detach().squeeze().cpu().numpy(), x_hat[0].detach().squeeze().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].plot(x, label="x")
        axes[0].set_title(f"gt - {self.hparam['MODEL']} {mode} epoch {epoch}")

        axes[1].plot(x_hat, label="x_hat")
        axes[1].set_title(f"reconstruction - {self.hparam['MODEL']} {mode} epoch {epoch}")

        plt.tight_layout()
        plt.savefig(f"{self.vis_dir}{mode}_epoch{epoch}.png")
        plt.close(fig)

    def vis_val_modular(self, x, x_hat, epoch):
        return

    def create_train_gif(self):
        # sorting all images in directory for creating the gif
        def to_int(str):
            return int(str) if str.isdigit() else str

        def natural_keys(str):
            return [to_int(c) for c in re.split(r"(\d+)", str)]

        # Create the frames
        frames = []
        imgs = glob.glob(self.vis_dir + "*.png")
        imgs.sort(key=natural_keys)
        for i in imgs:
            new_frame = Image.open(i)
            frames.append(new_frame)

        # Save into a GIF file that loops forever
        frames[0].save(f"{self.vis_dir}training.gif", format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)
        return print("\nTraining GIF was created ...\n\n")