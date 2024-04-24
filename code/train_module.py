import os
import gc
import math
import numpy as np

import torch.cuda
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import utils as u
import models as m
from data_module import *
from eval_module import *


class pipeline():
    def __init__(self, hparam):
        self.hparam = hparam

        self.log_dir = self.create_log_dir()

        self.vstr = u.VisTrain(hparam=hparam, log_dir=self.log_dir)
        self.logger = SummaryWriter(log_dir=self.log_dir)
        self.autostop = u.AutoStop(tolerance_e=50, min_delta=0.0001, max_delta=0.1)

    def train_net(self, datamodule):
        # get data
        dl_train, dl_val, dl_test = self.get_normal_dataloaders(datamodule)

        # for server
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.cuda.device(self.hparam["DEVICE"]) if torch.cuda.device_count() > 1 else torch.cuda.device(0):
            print(f"run on {self.device} {torch.cuda.get_device_name()}...\n")
            x_channel, y_channel = self.get_dims(dl_train)

            # init model + to device
            self.model = self.get_model(in_ch=x_channel, out_ch=y_channel, window_size=self.hparam["WINDOW_SIZE"])
            self.model.to(self.device)

            # init optimizer
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparam["LEARNING_RATE"], weight_decay=self.hparam["WEIGHT_DECAY"])
            # optimizer = torch.optim.SGD(self.model.parameters(), lr=hparam["LEARNING_RATE"], weight_decay=hparam["WEIGHT_DECAY"], momentum=0.1, dampening=0.01)
            # optimizer = torch.optim.ASGD(self.model.parameters(), lr=hparam["LEARNING_RATE"], weight_decay=hparam["WEIGHT_DECAY"], lambd=0.0001, alpha=0.75)

            # print training overview
            print(self.hparam, "\n\n")
            print(f"**********************************************************************\n"
                  f"Model structure: {self.model}\n")
            print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
            print("***********************************************************************\n")
            self.i = 0
            with tqdm(range(self.hparam["MAX_EPOCHS"]), unit="epoch") as tepoch:
                for self.e in tepoch:
                    # train step and logging
                    l_train = self.train_step(dl_train, optimizer)
                    self.logger.add_scalar("loss_train", l_train, self.e)
                    if math.isnan(l_train):
                        break

                    # val step and logging
                    if self.e % 50 == 0 and self.e != 0:
                        l_val = self.val_step(dl_val, modus="val")
                        self.logger.add_scalar("loss_val_y", l_val, self.e)

                        tepoch.set_postfix(loss_train=l_train, loss_val=l_val)
                        if self.autostop.auto_stop(l_train, l_val):
                            break

            # test step
            l_test = self.val_step(dl_test, modus="test")
            self.logger.flush()
            self.logger.close()

            torch.save(self.model.state_dict(), self.log_dir + "/model.pth")
            u.save_results(hparam=self.hparam, metrics={"l_train":l_train, "l_val":l_val, "l_test":l_test}, log_dir=self.log_dir, modus="train_results")
            self.vstr.create_train_gif()

            # freeing memory on GPU
            del self.model
            gc.collect()
            torch.cuda.empty_cache()

            return l_val

    def eval_net(self, datamodule_normal, datamodule_anomalous):
        _, _, dl_test_normal = self.get_normal_dataloaders(datamodule_normal)
        dl_dict = self.get_anomaly_dataloaders(datamodule_anomalous)

        # for server
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.cuda.device(self.hparam["DEVICE"]) if torch.cuda.device_count() > 1 else torch.cuda.device(0):
            print(f"run on {self.device} {torch.cuda.get_device_name()}...\n")

            x_channel, y_channel = self.get_dims(dl_test_normal)

            # init model + to device
            self.model = self.get_model(in_ch=x_channel, out_ch=y_channel, window_size=self.hparam["WINDOW_SIZE"])
            self.model.load_state_dict(torch.load(self.log_dir + "model.pth"))
            self.model.to(self.device)

            losses_ind_n, losses_mean_n = self.eval_step(dl_test_normal, anom="normal")

            eval = {"normal_ind": losses_ind_n, "normal_mean": losses_mean_n}

            for dl in dl_dict:
                print(f"anomaly eval {dl}")
                losses_ind, losses_mean = self.eval_step(dl_dict[dl], anom=dl)

                eval[str(dl) + "_ind"] = losses_ind
                eval[str(dl) + "_mean"] = losses_mean
            u.save_results(hparam=self.hparam, metrics=eval, log_dir=self.log_dir, modus="eval_results")

            #print(eval)

            # freeing memory on GPU
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
            return print(f"Evaluation run for model {self.hparam['MODEL']} on seed {self.hparam['SEED']} finished. All results are saved in {self.log_dir}results.json")

    def train_step(self, dl_train, optimizer):
        self.model.train()
        for x, y in dl_train:
            x, y = x.to(self.device), y.to(self.device)

            y_hat = self.model(x, y)
            loss = l2_loss(y, y_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.logger.add_scalar("loss_train_step", loss, self.i)
            self.i += 1
            #self.vstr.vis_val_monolith(x=x, x_hat=x_hat, epoch=self.e, mode="train")
        return loss.item()

    def val_step(self, dl_val, modus):
        self.model.eval()
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(self.device), y.to(self.device)

                y_hat = self.model(x, y)
                loss = rmse_loss(y, y_hat)

                self.logger.add_scalar("loss_" + modus + "_step", loss)

            # saving pred plots in vis dir
            try:
                self.vstr.vis_val_monolith(x=y, x_hat=y_hat, epoch=self.e, mode=modus)
            except:
                print("Vis failed ... \n")
        return loss.item()


    def eval_step(self, dl, anom):
        losses_ind, losses_mean, counter = [], [], 0
        self.model.eval()
        with (torch.no_grad()):
            for x, y in dl:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x,y)

                delta_sq = (y.cpu()-y_hat.cpu())**2
                mse = torch.mean(delta_sq, dim=1)
                rmse_ind  = torch.sqrt(mse)
                rmse_mean = torch.mean(rmse_ind, dim=1)

                rmse_ind  = rmse_ind.tolist()
                rmse_mean = rmse_mean.tolist()

                for i, j in zip(rmse_ind, rmse_mean):
                    losses_ind.append(i)
                    losses_mean.append(j)

                y, y_hat = y.detach().squeeze().cpu().numpy(), y_hat.detach().squeeze().cpu().numpy()

                with open(str(self.log_dir + "/vis/" + anom + str(counter) + "_gt.npy"), "wb") as f:
                    np.save(f, y)
                with open(str(self.log_dir + "/vis/" + anom + str(counter) + "_pred.npy"), "wb") as f:
                    np.save(f, y_hat)

                # self.vstr.vis_val_monolith(x=x, x_hat=x_hat, epoch=self.e, mode="eval")
        return losses_ind, losses_mean

    def get_normal_dataloaders(self, datamodule):
        dl_train = datamodule.train_loader
        dl_val   = datamodule.val_loader
        dl_test  = datamodule.test_loader
        return dl_train, dl_val, dl_test

    def get_anomaly_dataloaders(self, datamodule):
        dl_weight   = datamodule.test_loader_weight
        dl_drop     = datamodule.test_loader_drop
        dl_stop     = datamodule.test_loader_stop
        dl_out      = datamodule.test_loader_out
        dl_push     = datamodule.test_loader_push
        dl_speed    = datamodule.test_loader_speed
        dl_wrench   = datamodule.test_loader_wrench
        return {"weight": dl_weight, "drop": dl_drop, "stop": dl_stop, "out": dl_out, "push": dl_push, "speed": dl_speed, "wrench": dl_wrench}

    def get_model(self, in_ch, out_ch, window_size):
        # Monolothic Types
        if "MonolithAE" in self.hparam["MODEL"]:
            model = m.MonolithAE(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        elif "MonolithVAE" in self.hparam["MODEL"]:
            model = m.MonolithVAE(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        elif "MonolithRNN" in self.hparam["MODEL"]:
            model = m.MonolithRNN(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        elif "MonolithLSTM" in self.hparam["MODEL"]:
            model = m.MonolithLSTM(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        elif "MonolithGRU" in self.hparam["MODEL"]:
            model = m.MonolithGRU(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        # Modular Types -  Type 0
        elif "Modular0AE" in self.hparam["MODEL"]:
            model = m.Modular0AE(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        elif "Modular0VAE" in self.hparam["MODEL"]:
            model = m.Modular0VAE(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        elif "Modular0RNN" in self.hparam["MODEL"]:
            model = m.Modular0RNN(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        elif "Modular0LSTM" in self.hparam["MODEL"]:
            model = m.Modular0LSTM(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        elif "Modular0GRU" in self.hparam["MODEL"]:
            model = m.Modular0GRU(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        # Modular Types - Type 1
        elif "Modular1AE" in self.hparam["MODEL"]:
            model = m.Modular1AE(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        elif "Modular1VAE" in self.hparam["MODEL"]:
            model = m.Modular1VAE(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        elif "Modular1RNN" in self.hparam["MODEL"]:
            model = m.Modular1RNN(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        elif "Modular1LSTM" in self.hparam["MODEL"]:
            model = m.Modular1LSTM(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        elif "Modular1GRU" in self.hparam["MODEL"]:
            model = m.Modular1GRU(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        # Modular Types - Type 2
        elif "Modular2AE" in self.hparam["MODEL"]:
            model = m.Modular2AE(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        elif "Modular2VAE" in self.hparam["MODEL"]:
            model = m.Modular2VAE(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        elif "Modular2RNN" in self.hparam["MODEL"]:
            model = m.Modular2RNN(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        elif "Modular2LSTM" in self.hparam["MODEL"]:
            model = m.Modular2LSTM(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        elif "Modular2GRU" in self.hparam["MODEL"]:
            model = m.Modular2GRU(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        # Modular Types - Type 2
        elif "Modular3AE" in self.hparam["MODEL"]:
            model = m.Modular3AE(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        elif "Modular3VAE" in self.hparam["MODEL"]:
            model = m.Modular3VAE(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        elif "Modular3RNN" in self.hparam["MODEL"]:
            model = m.Modular3RNN(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        elif "Modular3LSTM" in self.hparam["MODEL"]:
            model = m.Modular3LSTM(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        elif "Modular3GRU" in self.hparam["MODEL"]:
            model = m.Modular3GRU(hparam=self.hparam, in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        return model

    def get_dims(self, dl):
        """
        Returns number of channels for x and y
        """
        x, y = next(iter(dl))
        sample_len = self.hparam["WINDOW_SIZE"]
        x_channels = x.shape[2]
        y_channels = y.shape[2]
        return x_channels, y_channels

    def create_log_dir(self):
        log_dir = self.hparam["LOG_DIR"] + "logs/train/" + self.hparam["MODEL"] + "/" + str(self.hparam["EXPERIMENT_ID"]) + "/"
        if not os.path.exists(log_dir + "vis/"):
            os.makedirs(log_dir + "vis/")
        return log_dir

# Quickrun
if __name__ == "__main__":
    exp_setup = {
        "EXPERIMENT_ID": "test",
        "DATA_DIR": "../data/dataset/",
        "LOG_DIR": "../exp/logs/",
        "DATA_MODE": "elec",
        "DATASET": "change_weight",
        "SEED": 1,
        "NUM_WORKERS": 16,
        "WINDOW_SIZE": 1000,
        "STRIDE": 100,
        "BATCH_SIZE": 16,

        "N_LAYERS": 4,
        "AE_F_ENCODING": 1.5,
        "DROPOUT_P": 0.01,

        "HIDDEN_SIZES": [128],
        "LATENT_SIZE": 256,

        "MODEL": "Modular3RNN",
        "DEVICE": 0,
        "MAX_EPOCHS": 10,
        "LEARNING_RATE": 0.00001,
        "WEIGHT_DECAY": 0.000
    }

    print(f"Loading data for ... {exp_setup}")
    # data module & dataloaders
    data_normal = DataModuleNormal(hparam=exp_setup)
    data_anomalous = DataModuleAnomalous(hparam=exp_setup)

    pipeline(exp_setup).train_net(data_normal)
    pipeline(exp_setup).eval_net(datamodule_normal=data_normal, datamodule_anomalous=data_anomalous)