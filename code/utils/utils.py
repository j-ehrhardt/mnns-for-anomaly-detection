import math
import os
import json
import numpy as np

class AutoStop:
    def __init__(self, tolerance_e=40, min_delta=0.0001, max_delta=0.001):
        self.tolerance = tolerance_e
        self.max_delta = max_delta
        self.min_delta = min_delta
        self.counter_max = 0
        self.counter_min = 0
        self.min_loss = np.inf

    def auto_stop(self, l_x_val, l_y_val):
        if l_x_val < self.min_loss and l_y_val < self.min_loss:
            if (l_x_val - self.min_loss) < self.min_delta and (l_y_val - self.min_loss) < self.min_delta:
                self.counter_min += 1
                if self.counter_min > self.tolerance*2:
                    print("\n\nAutostop: Convergence criterion\n\n")
                    return True
            self.min_loss = max([l_x_val, l_y_val])
            self.counter_min, self.counter_max = 0, 0
        elif l_x_val > (self.min_loss + self.max_delta) and l_y_val > (self.min_loss + self.max_delta):
            self.counter_max += 1
            if self.counter_max >= self.tolerance:
                print("\n\nAutostop: Divergence criterion\n\n")
                return True
        elif l_x_val == np.inf or l_y_val == np.inf:
            return True
        elif math.isnan(l_x_val):
            return True
        return False


def launch_tensorboard(log_dir = "logs/ds3/"):
    os.system('tensorboard --logdir="' + log_dir + '"')
    return


def save_results(hparam: dict, metrics: dict, log_dir: str, modus: str):
    """
    saving results and experimental setup as .json file in log-dir
    metrics must be passed as dict: [train_y_MSE, train_X_MSE, test_y_MSE, test_X_MSE]
    """
    results = {"setup": hparam, str(modus):metrics}

    print(results)

    with open(log_dir + "/results.json", "w") as handle:
        json.dump(results, handle, indent=4)
        handle.close()
    return print(f"Logged results of experiment ", hparam["EXPERIMENT_ID"])


def load_hparam(path="../exp/exp_setup/experiments.json"):
    with open(path, "r") as json_file:
        hparam = json.load(json_file)
    return hparam

def save_hparam(hparam, path="../exp/exp_setup/best_runs.json"):
    with open(path, "w") as json_file:
        json.dump(hparam, json_file, indent=4)
    return

def transfer_best_hparams(exp_dict, res_dict, index):
    """
    translates the best hparams from the hparam studies into "best_experiments.json"
    """
    for key_res in res_dict:
        if key_res.upper() in exp_dict.keys():
            exp_dict[key_res.upper()] = res_dict[key_res]

    best_dict = load_hparam(path="../exp/exp_setup/best_runs.json")
    best_dict[index] = exp_dict

    save_hparam(best_dict, path="../exp/exp_setup/best_runs.json")
    return print(f"Best hyperparameters for experiment logged in 'best_runs.json'...\n")