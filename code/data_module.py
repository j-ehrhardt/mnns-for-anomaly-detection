import torch
import torch.nn
import glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

class TimeSeriesDataset(Dataset):
    """
    Dataset class to create datasets of different length.
    """
    def __init__(self, hparam, dataset, variable_type):
        self.hparam = hparam
        self.variable_type = variable_type
        self.csv_files = glob.glob(self.hparam["DATA_DIR"] + dataset + "/*.csv")
        self.data = self.load_data()

    def load_data(self):
        data = []
        for file in self.csv_files:
            df = pd.read_csv(file, sep=" ", index_col="timestamp")
            if self.variable_type == "elec":
                cols = [col for col in df.columns if ("actual_robot_current" in col) or ("actual_current" in col)] #("actual" in col and "current" in col) or ("actual" in col and "voltage" in col)]]
                df = df[[cols[-1]] + cols[:-1]]
            elif self.variable_type == "pos":
                cols = [col for col in df.columns if ("target_moment" in col)] # or ("actual" in col and "TCP_pose" in col)
                cols.reverse()
                df = df[cols]

            windows = self.windowing(df)
            for w in windows:
                data.append(w[::1].values)
        return data

    def windowing(self, ds):
        ds_len = len(ds)
        windows = []
        for i in range(0, ds_len - self.hparam["WINDOW_SIZE"] + 1, self.hparam["STRIDE"]):
            j = i + self.hparam["WINDOW_SIZE"]
            window = ds.iloc[i:j, :]
            windows.append(window)
        return windows

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, item):
        return torch.tensor(self.data[item], dtype=torch.float32)#.unsqueeze(0)


class DataModule():
    def __init__(self, hparam:dict):
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
        val_size   = int(0.2 * len(ds))
        test_size = len(ds) - train_size - val_size

        ds_train, ds_val, ds_test = random_split(ds, [train_size, val_size, test_size])
        return ds_train, ds_val, ds_test


    def init_dataloader(self, ds, batch_size=1, shuffle=False):  # TODO set shuffle True for prototyping
        return DataLoader(ds, batch_size=batch_size, num_workers=self.hparam["NUM_WORKERS"], shuffle=shuffle, drop_last=True, pin_memory=True) #collate_fn=self.custom_collate)


class DataModuleNormal(DataModule):
    """
    Class that returns train/val/test split separated datasets for training
    """
    def __init__(self, hparam):
        super(DataModule, self).__init__()
        self.hparam = hparam


        electric_n_train, electric_n_val, electric_n_test = self.sampler(TimeSeriesDataset(self.hparam, "normal", "elec")) # TODO change back to "normal_" for prototyping
        force_n_train, force_n_val, force_n_test = self.sampler(TimeSeriesDataset(self.hparam, "normal", "pos")) # TODO change back to "normal_" for prototyping

        self.train_loader_electric = self.init_dataloader(electric_n_train, batch_size=self.hparam["BATCH_SIZE"])
        self.train_loader_force = self.init_dataloader(force_n_train, batch_size=self.hparam["BATCH_SIZE"])
        self.val_loader_electric = self.init_dataloader(electric_n_val, batch_size=self.hparam["BATCH_SIZE"])
        self.val_loader_force = self.init_dataloader(force_n_val, batch_size=self.hparam["BATCH_SIZE"])
        self.test_loader_electric = self.init_dataloader(electric_n_test, batch_size=self.hparam["BATCH_SIZE"])
        self.test_loader_force = self.init_dataloader(force_n_test, batch_size=self.hparam["BATCH_SIZE"])

class DataModuleAnomalous(DataModule):
    """
    Class that return anomaly datasets of 7 anomalies
    """
    def __init__(self, hparam):
        super(DataModule, self).__init__()
        self.hparam = hparam
        anomalies = ["change_weight", "drop_object", "force_stop", "outside_box", "push", "scale_speed", "tcp_wrench"]

        weight = self.load_anomalous_ds(anomalies[0])
        drop   = self.load_anomalous_ds(anomalies[1])
        stop   = self.load_anomalous_ds(anomalies[2])
        out    = self.load_anomalous_ds(anomalies[3])
        push   = self.load_anomalous_ds(anomalies[4])
        speed  = self.load_anomalous_ds(anomalies[5])
        wrench = self.load_anomalous_ds(anomalies[6])

        self.test_loader_weight_electric = self.init_dataloader(weight[0], shuffle=False)
        self.test_loader_weight_force    = self.init_dataloader(weight[1], shuffle=False)

        self.test_loader_drop_electric = self.init_dataloader(drop[0], shuffle=False)
        self.test_loader_drop_force    = self.init_dataloader(drop[1], shuffle=False)

        self.test_loader_stop_electric = self.init_dataloader(stop[0], shuffle=False)
        self.test_loader_stop_force    = self.init_dataloader(stop[1], shuffle=False)

        self.test_loader_out_electric = self.init_dataloader(out[0], shuffle=False)
        self.test_loader_out_force    = self.init_dataloader(out[1], shuffle=False)

        self.test_loader_push_electric = self.init_dataloader(push[0], shuffle=False)
        self.test_loader_push_force    = self.init_dataloader(push[1], shuffle=False)

        self.test_loader_speed_electric = self.init_dataloader(speed[0], shuffle=False)
        self.test_loader_speed_force    = self.init_dataloader(speed[1], shuffle=False)

        self.test_loader_wrench_electric = self.init_dataloader(wrench[0], shuffle=False)
        self.test_loader_wrench_force    = self.init_dataloader(wrench[1], shuffle=False)

    def load_anomalous_ds(self, anomaly):
        ds_elec = TimeSeriesDataset(self.hparam, anomaly + "/test/", "elec")
        ds_pos = TimeSeriesDataset(self.hparam, anomaly + "/test/", "pos")
        return [ds_elec, ds_pos]



# quick test and quick run
if __name__ == "__main__":
    exp_setup = {
        "EXPERIMENT_ID": "test",
        "DATA_DIR": "../data/dataset/",
        "LOG_DIR": "../exp/logs/",
        "DATASET": "normal",
        "SEED": 1,
        "NUM_WORKERS": 16,
        "WINDOW_SIZE":500,
        "STRIDE": 100,
        "BATCH_SIZE": 16
    }

    print(exp_setup)

    n_data = DataModuleNormal(hparam=exp_setup)
    x_train_I, x_train_f = n_data.train_loader_electric, n_data.train_loader_force
    x_val_I, x_val_f = n_data.val_loader_electric, n_data.val_loader_force
    x_test_I, x_test_f = n_data.test_loader_electric, n_data.test_loader_force

    # print first sample in train_dataloader
    I1_train, f1_train = next(iter(x_train_I)), next(iter(x_train_f))
    I1_val, f1_val = next(iter(x_val_I)), next(iter(x_val_f))
    I1_test, f1_test = next(iter(x_test_I)), next(iter(x_test_f))

    print(type(next(iter(x_train_I))), type(next(iter(x_train_f))))

    print("format: torch.Tensor(batch, channels, seq_len: ")
    print("I1_train shape: ", I1_train.shape)
    print("f1_train shape: ", f1_train.shape)
    print("I1_val shape: ", I1_val.shape)
    print("f1_val shape: ", f1_val.shape)
    print("I1_test shape: ", I1_test.shape)
    print("f1_test shape: ", f1_test.shape)


    a_data = DataModuleAnomalous(hparam=exp_setup)
    a1_test_I, a1_test_f = a_data.test_loader_weight_electric, a_data.test_loader_weight_force
    a2_test_I, a2_test_f = a_data.test_loader_drop_electric, a_data.test_loader_drop_force
    a3_test_I, a3_test_f = a_data.test_loader_stop_electric, a_data.test_loader_stop_force
    a4_test_I, a4_test_f = a_data.test_loader_out_electric, a_data.test_loader_out_force
    a5_test_I, a5_test_f = a_data.test_loader_push_electric, a_data.test_loader_push_force
    a6_test_I, a6_test_f = a_data.test_loader_speed_electric, a_data.test_loader_push_force
    a7_test_I, a7_test_f = a_data.test_loader_wrench_electric, a_data.test_loader_wrench_force

    a1_test_I, a1_test_f = next(iter(a1_test_I)), next(iter(a1_test_f))
    a7_test_I, a7_test_f = next(iter(a7_test_I)), next(iter(a7_test_f))

    print("format: torch.Tensor(batch, channels, seq_len: ")
    print("a1_test_I shape: ", a1_test_I.shape)
    print("a1_test_f shape: ", a1_test_f.shape)
    print("a7_test_I shape: ", a7_test_I.shape)
    print("a7_test_f shape: ", a7_test_f.shape)

