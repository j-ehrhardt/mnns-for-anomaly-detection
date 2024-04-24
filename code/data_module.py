import torch
import torch.nn
import glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split


class TimeSeriesDataset(Dataset):
    """
    Dataset class to create datasets of different length.
    """

    def __init__(self, hparam, dataset):
        self.hparam = hparam
        self.csv_files = glob.glob(self.hparam["DATA_DIR"] + dataset + "/*.csv")
        self.data_in, self.data_out = self.load_data()

    def load_data(self):
        data_in, data_out = [], []
        for file in self.csv_files:
            df = pd.read_csv(file, sep=" ", index_col="timestamp")

            cols = [col for col in df.columns if ("actual_current_" in col) or ("actual_qd_" in col)] # fixed inputs are: actual joint current and actual joint acceleration
            df_in = df[[cols[-1]] + cols[:-1]]
            df_in = df_in.sort_index(axis=1)

            cols = [col for col in df.columns if ("actual_q_" in col)]  # outputs are: actual joint position
            df_out = df[[cols[-1]] + cols[:-1]]
            df_out = df_out.sort_index(axis=1)

            windows = self.windowing(df_in)
            for w in windows: data_in.append(w[::1].values)
            windows = self.windowing(df_out)
            for w in windows: data_out.append(w[::1].values)
        return data_in, data_out

    def windowing(self, ds):
        ds_len = len(ds)
        windows = []
        for i in range(0, ds_len - self.hparam["WINDOW_SIZE"] + 1, self.hparam["STRIDE"]):
            j = i + self.hparam["WINDOW_SIZE"]
            window = ds.iloc[i:j, :]
            windows.append(window)
        return windows

    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, item):
        x = torch.tensor(self.data_in[item], dtype=torch.float32)  # .unsqueeze(0)
        y = torch.tensor(self.data_out[item], dtype=torch.float32)  # .unsqueeze(0)
        return x, y


class DataModule():
    def __init__(self, hparam: dict):
        """
        datamodule to convert .csv to datasets
        including slicer, train- test- val- sampler, and dataloader.

        :param hparam: [dict] containing experimental setup
        """
        self.hparam = hparam

    def sampler(self, ds):
        """
        Sampler for normal dataset, anomalous are not split into train, test, val.
        """
        train_size = int(0.6 * len(ds))
        val_size = int(0.2 * len(ds))
        test_size = len(ds) - train_size - val_size

        ds_train, ds_val, ds_test = random_split(ds, [train_size, val_size, test_size])
        return ds_train, ds_val, ds_test

    def init_dataloader(self, ds, batch_size=1, shuffle=False):
        return DataLoader(ds, batch_size=batch_size, num_workers=self.hparam["NUM_WORKERS"], shuffle=shuffle,
                          drop_last=True, pin_memory=True)  # collate_fn=self.custom_collate)


class DataModuleNormal(DataModule):
    """
    Class that returns train/val/test split separated datasets for training
    """

    def __init__(self, hparam):
        super(DataModule, self).__init__()
        self.hparam = hparam

        train, val, test = self.sampler(TimeSeriesDataset(self.hparam, "normal"))  # TODO change back to "normal_" for prototyping

        self.train_loader = self.init_dataloader(train, batch_size=self.hparam["BATCH_SIZE"])
        self.val_loader = self.init_dataloader(val, batch_size=self.hparam["BATCH_SIZE"])
        self.test_loader = self.init_dataloader(test, batch_size=self.hparam["BATCH_SIZE"])



class DataModuleAnomalous(DataModule):
    """
    Class that return anomaly datasets of 7 anomalies
    """

    def __init__(self, hparam):
        super(DataModule, self).__init__()
        self.hparam = hparam
        anomalies = ["change_weight", "drop_object", "force_stop", "outside_box", "push", "scale_speed", "tcp_wrench"]

        weight = self.load_anomalous_ds(anomalies[0])
        drop = self.load_anomalous_ds(anomalies[1])
        stop = self.load_anomalous_ds(anomalies[2])
        out = self.load_anomalous_ds(anomalies[3])
        push = self.load_anomalous_ds(anomalies[4])
        speed = self.load_anomalous_ds(anomalies[5])
        wrench = self.load_anomalous_ds(anomalies[6])

        self.test_loader_weight = self.init_dataloader(weight, shuffle=False)
        self.test_loader_drop = self.init_dataloader(drop, shuffle=False)
        self.test_loader_stop = self.init_dataloader(stop, shuffle=False)
        self.test_loader_out = self.init_dataloader(out, shuffle=False)
        self.test_loader_push = self.init_dataloader(push, shuffle=False)
        self.test_loader_speed = self.init_dataloader(speed, shuffle=False)
        self.test_loader_wrench = self.init_dataloader(wrench, shuffle=False)

    def load_anomalous_ds(self, anomaly):
        ds = TimeSeriesDataset(self.hparam, anomaly + "/test/")
        return ds


# quick test and quick run
if __name__ == "__main__":
    exp_setup = {
        "EXPERIMENT_ID": "test",
        "DATA_DIR": "../data/dataset/",
        "LOG_DIR": "../exp/logs/",
        "DATASET": "normal",
        "SEED": 1,
        "NUM_WORKERS": 16,
        "WINDOW_SIZE": 500,
        "STRIDE": 100,
        "BATCH_SIZE": 16
    }

    print(exp_setup)

    n_data = DataModuleNormal(hparam=exp_setup)
    train = n_data.train_loader
    val = n_data.val_loader
    test = n_data.test_loader

    # print first sample in train_dataloader
    train_x, train_y = next(iter(train))
    val_x, val_y = next(iter(val))
    test_x, test_y = next(iter(test))

    print("format: torch.Tensor(batch, channels, seq_len: ")
    print("train_x shape: ", train_x.shape)
    print("train_y shape: ", train_y.shape)
    print("val_x shape: ", val_x.shape)
    print("val_y shape: ", val_y.shape)
    print("test_x shape: ", test_x.shape)
    print("test_y shape: ", test_y.shape)

    a_data = DataModuleAnomalous(hparam=exp_setup)
    a1_test = a_data.test_loader_weight
    a2_test = a_data.test_loader_drop
    a3_test = a_data.test_loader_stop
    a4_test = a_data.test_loader_out
    a5_test_ = a_data.test_loader_push
    a6_test = a_data.test_loader_speed
    a7_test = a_data.test_loader_wrench

    a1_test_x, a1_test_y = next(iter(a1_test))
    a7_test_x, a7_test_y = next(iter(a7_test))

    print("format: torch.Tensor(batch, channels, seq_len: ")
    print("a1_test_x shape: ", a1_test_x.shape)
    print("a1_test_y shape: ", a1_test_y.shape)
    print("a7_test_x shape: ", a7_test_x.shape)
    print("a7_test_y shape: ", a7_test_y.shape)
