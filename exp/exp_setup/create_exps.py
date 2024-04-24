import json

def create_exps():
    model_list = ["MonolithAE", "MonolithVAE", "Modular0AE", "Modular0VAE", "Modular1AE", "Modular1VAE", "Modular2AE", "Modular2VAE", "Modular3AE", "Modular3VAE"]
    # model_list = ["MonolithAE", "MonolithVAE", "MonolithRNN", "MonolithLSTM", "MonolithGRU", "Modular0AE", "Modular0VAE", "Modular0RNN", "Modular0LSTM",
    #              "Modular0GRU", "Modular1AE", "Modular1VAE", "Modular1RNN", "Modular1LSTM", "Modular1GRU", "Modular2AE", "Modular2VAE", "Modular2RNN",
    #              "Modular2LSTM", "Modular2GRU", "Modular3AE", "Modular3VAE", "Modular3RNN", "Modular3LSTM", "Modular3GRU"]

    hparams, i = {}, 0

    for model in model_list:
        hparam_base= {
            "EXPERIMENT_ID": i,
            "DATA_DIR": "../data/dataset/",
            "LOG_DIR": "../exp/logs/",
            "DATA_MODE": "modus",
            "DATASET": "normal",
            "SEED": 1,
            "NUM_WORKERS": 0,
            "DEVICE": 0,
            "MODEL":model,
            "WINDOW_SIZE":1000,
            "STRIDE":100,
            "BATCH_SIZE":64
        }

        hparams[str(i)] = hparam_base
        i += 1

    with open("experiments.json", "w") as file:
        json.dump(hparams, file, indent=4)

    return print("Experiments were created")

# Quickrun
if __name__ == "__main__":
    create_exps()