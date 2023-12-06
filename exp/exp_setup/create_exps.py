import json

def create_exps():
    model_list = ["MonolithAE", "MonolithVAE", "Modular0AE", "Modular0VAE", "Modular1AE", "Modular1VAE", "Modular2AE", "Modular2VAE"]

    modus_list = ["elec", "pos"]


    hparams, i = {}, 0
    for modus in modus_list:
        for model in model_list:
            hparam_base= {
                "EXPERIMENT_ID": i,
                "DATA_DIR": "../data/dataset/",
                "LOG_DIR": "../exp/logs/",
                "DATA_MODE": modus,
                "DATASET": "change_weight",
                "SEED": 1,
                "NUM_WORKERS": 16,
                "DEVICE": 0,
                "MODEL":model + "_" + modus[0],
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