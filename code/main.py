import sys
import json
import random
import optuna
import warnings
from utils import *
from train_module import *
warnings.filterwarnings("ignore", category=UserWarning)


class hparam_search():
    def __init__(self, hparam):
        self.hparam = hparam
        self.data_module = DataModuleNormal(hparam=hparam)

        # init pipeline seeds
        random.seed(hparam["SEED"])
        np.random.seed(hparam["SEED"])
        torch.manual_seed(hparam["SEED"])
        torch.cuda.manual_seed(hparam["SEED"])

        self.study = optuna.create_study(direction="minimize", storage="sqlite:///" + hparam["LOG_DIR"] + "hparam_studies" + ".db", study_name=hparam["MODEL"], load_if_exists=True, pruner=True)
        self.study.optimize(self.experimental_run, n_trials=32)

    def experimental_run(self, trial):
        # setup
        self.hparam["EXPERIMENT_ID"] = self.hparam["MODEL"] + "_exp_" + str(trial.number)

        # parameters for forward models
        if "VAE" in self.hparam["MODEL"]:
            self.hparam["N_LAYERS"] = trial.suggest_int("n_layers", 2, 4)
            self.hparam["AE_F_ENCODING"] = trial.suggest_float("ae_f_encoding", 0.5, 4.0)
            self.hparam["DROPOUT_P"] = trial.suggest_float("dropout_p", 0.0, 0.5)
            # training parameters
            self.hparam["MAX_EPOCHS"] = trial.suggest_int("max_epochs", 10, 2000)
            self.hparam["LEARNING_RATE"] = trial.suggest_float("learning_rate", 0.00001, 0.01)
            self.hparam["WEIGHT_DECAY"] = trial.suggest_float("weight_decay", 0.0, 0.0001)

        if "AE" in self.hparam["MODEL"] and "VAE" not in self.hparam["MODEL"]:
            self.hparam["N_LAYERS"] = trial.suggest_int("n_layers", 2, 4)
            self.hparam["AE_F_ENCODING"] = trial.suggest_float("ae_f_encoding", 0.5, 4.0)
            self.hparam["DROPOUT_P"] = trial.suggest_float("dropout_p", 0.0, 0.5)
            # training parameters
            self.hparam["MAX_EPOCHS"] = trial.suggest_int("max_epochs", 10, 2000)
            self.hparam["LEARNING_RATE"] = trial.suggest_float("learning_rate", 0.0001, 0.01)
            self.hparam["WEIGHT_DECAY"] = trial.suggest_float("weight_decay", 0.0, 0.001)

        # parameters for recurrent models
        if "AE" not in self.hparam["MODEL"]:
            n_hidden_sizes = trial.suggest_int("hidden_layer_factor", 1 ,4)
            hidden_sizes = trial.suggest_categorical("hidden_sizes_l", [16, 32, 64, 128, 256]) # ["16", "32", "64", "128", "256", "512"])
            self.hparam["HIDDEN_SIZES"] = [int(hidden_sizes)] * n_hidden_sizes
            self.hparam["LATENT_SIZE"] = trial.suggest_int("latent_size", 16, 1024)
            # training parameters
            self.hparam["MAX_EPOCHS"] = trial.suggest_int("max_epochs", 10, 2000)
            self.hparam["LEARNING_RATE"] = trial.suggest_float("learning_rate", 0.0001, 0.01)
            self.hparam["WEIGHT_DECAY"] = trial.suggest_float("weight_decay", 0.0, 0.0001)

        # setting up pipeline
        try:
            criterion = pipeline(self.hparam).train_net(self.data_module)
        except:
            criterion = "failed trial :("
        return criterion


class replication_study():
    """
    Rerun the best-performing experiments on 5 different seeds to proof replicability
    """
    def __init__(self, hparam):
        seeds = [42, 21, 13, 4, 29, 32, 11, 3]

        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            hparam["SEED"] = seed
            data_module = DataModuleNormal(hparam=hparam)
            self.run_training(data_module, hparam)

    def run_training(self, data_module, hparam):
        try:
            hparam["EXPERIMENT_ID"] = hparam["MODEL"] + "_seed_" + str(hparam["SEED"])
            hparam["LOG_DIR"] = "../exp/logs/repl_studies/"

            criterion = pipeline(hparam).train_net(data_module)
            return print(f"Experiment {hparam['EXPERIMENT_ID']} on seed {hparam['SEED']} finished successfully with a criterion of {criterion}")

        except:
            return print(f"Experiment {hparam['EXPERIMENT_ID']} on seed {hparam['SEED']} failed :(")


class evaluation_study():
    """
    All evaluations and print them into the evaluation script.
    """
    def __init__(self, hparam):
        seeds = [42, 21, 13, 4, 29, 32, 11, 3]
        hparam["LOG_DIR"] = "../exp/logs/repl_studies/"

        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            hparam["SEED"] = seed
            hparam["EXPERIMENT_ID"] = hparam["MODEL"] + "_seed_" + str(hparam["SEED"])

            data_module_normal = DataModuleNormal(hparam=hparam)
            data_module_anomalous = DataModuleAnomalous(hparam=hparam)

            self.eval_experiment(hparam, data_module_normal, data_module_anomalous)

    def eval_experiment(self, hparam, data_module_normal, data_module_anomalous):
        pipeline(hparam=hparam).eval_net(datamodule_normal=data_module_normal, datamodule_anomalous=data_module_anomalous)
        return



def load_experiments(path="../exp/exp_setup/experiments.json"):
    with open(path) as json_file:
        exps = json.load(json_file)
    return exps



# Quickrun
if __name__ == "__main__":

    device = sys.argv[1]                            # device number (int)
    print(f"Experiments {sys.argv[2:]} will be conducted on device {sys.argv[1]}")
    
    # for click and replicate:
    exps = load_experiments()
    for model_index in sys.argv[2:]:
        exp = exps[str(model_index)]
        exp["DEVICE"] = int(device)
        hparam_study = hparam_search(exp)                       # run hyperparameter study
        best = hparam_study.study.best_params                   # return  best hyperparameter
        transfer_best_hparams(exp, best, model_index)           # log best hparams in best_runs.json

    exps = load_experiments(path="../exp/exp_setup/best_runs.json")
    for model_index in sys.argv[2:]:
        exp =  exps[str(model_index)]
        exp["DEVICE"] = int(device)
        replication_study(exp)
        evaluation_study(exp)
    """ """
    """
    # for hyperparameter search only
    exps = load_experiments()
    for model_index in sys.argv[2:]:    # experiment numbers (int)
        exp =  exps[str(model_index)]
        exp["DEVICE"] = int(device)
        hparam_search(exp)
    """

    """ 
    # for replication studies + evaluation study only
    exps = load_experiments(path="../exp/exp_setup/best_runs.json")
    for model_index in sys.argv[2:]:    # experiment numbers (int)
        exp =  exps[str(model_index)]
        exp["DEVICE"] = int(device)
        replication_study(exp)
        evaluation_study(exp)
    """

    """    
    # for evaluation only 
    exps = load_experiments(path="../exp/exp_setup/best_runs.json")

    for model_index in sys.argv[2:]:
        exp = exps[str(model_index)]
        exp["DEVICE"] = int(device)
        evaluation_study(hparam=exp)
    """