import os
import json
import math
import statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support

class ImportResults():
    def __init__(self, path, model_list, ds_list, anomaly_list):
        glob_res = self.results_to_dict(path, model_list, ds_list)
        self.ind_res, self.mean_res = self.ind_res_mean_res(glob_res)
        self.glob_res = self.global_res(glob_res)

    def load_json(self, path):
        with open(path + "/results.json", "r") as file:
            results = json.load(file)
        return results

    def contains_nan(self, data):
        if isinstance(data, list):
            return any(self.contains_nan(item) for item in data)
        else:
            return isinstance(data, (float, int)) and math.isnan(data)

    def results_to_dict(self, path, model_list, ds_list):
        """
        loading all results.json files into a global dictionary
        """
        results = {}

        for ds in ds_list:
            for model in model_list:
                try:
                    res_dir_path = path + model + ds + "/"
                    dir_list = [d for d in os.listdir(res_dir_path) if os.path.isdir(os.path.join(res_dir_path, d))]

                    for dir_ in dir_list:
                        try:
                            res = self.load_json(res_dir_path + "/" + dir_)
                            results[dir_] = res
                        except:
                            print(f"no results for {dir_}")
                except:
                    print(f"no results for {model}")

        return results

    def ind_res_mean_res(self, results_dict):
        """
        separating the logged results from the global dict into results_ind containing the mse for every file
        and each subsystem and results_mean containing the mse for every file averaged over all subsystems
        """
        results_ind, results_mean = {}, {}

        for key in results_dict:
            results_ind[key], results_mean[key] = {}, {}
            for case in results_dict[key]["eval_results"]:
                if "_ind" in case and not self.contains_nan(results_dict[key]["eval_results"][case]):
                    results_ind[key][case] = results_dict[key]["eval_results"][case]
                elif "_mean" in case and not self.contains_nan(results_dict[key]["eval_results"][case]):
                    results_mean[key][case] = results_dict[key]["eval_results"][case]
                else:
                    continue
        return results_ind, results_mean

    def global_res(self, results_dict):
        global_dir = {}
        for key in results_dict:
            global_dir[key] = results_dict[key]["eval_results"]
        return global_dir


class ImportVis():
    def __init__(self, path, model_list, ds_list, anomaly_case):
        res_dict = self.load_files_into_dict(path, ds_list, model_list, anomaly_case)

        for col in range(0,6):
            self.plot_anom_fig(res_dict, col, ds_list)

    def load_files_into_dict(self, path, ds_list, model_list, anom):
        results = {}

        for ds in ds_list:
            for model in model_list:
                try:
                    res_dir_path = path + model + ds + "/"
                    dir_list = [d for d in os.listdir(res_dir_path) if os.path.isdir(os.path.join(res_dir_path, d))]

                    with open(res_dir_path + dir_list[0] + "/vis/" + anom + "0" + "_gt.npy", "rb") as f:
                        gt = np.load(f)
                    with open(res_dir_path + dir_list[0] + "/vis/" + anom + "0" + "_pred.npy", "rb") as f:
                        pred = np.load(f)

                    results[model + ds] = {"gt": gt, "pred": pred}

                except:
                    print(f"no results for {model}")
        return results

    def plot_anom_fig(self, res_dict, col, ds_list):
        gt = res_dict["MonolithAE" + ds_list[0]]["gt"]
        plt.plot(gt[0, :, col], label="gt")

        for key in res_dict:
            pred = res_dict[key]["pred"]
            plt.plot(pred[0, :, col], label=key, alpha=0.5)

        plt.xlabel("timesteps")
        plt.ylabel("current/moments")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Legend Title')
        plt.title("Reconstructions and Groundtruth of Joint " + str(col))
        plt.tight_layout()
        plt.show()
        return print("Figure was plotted.")


class Metrics():
    def __init__(self, gt, preds):
        true_positives = sum(1 for a, b in zip(gt, preds) if a == b == 1)
        false_positives = sum(1 for a, b in zip(gt, preds) if a == 1 and b == 0)
        true_negatives = sum(1 for a, b in zip(gt, preds) if a == b == 0)
        false_negatives = sum(1 for a, b in zip(gt, preds) if a == 0 and b == 1)

        self.sensitivity = self.calc_sensitivity(true_positives, false_negatives)
        self.specificity = self.calc_specificity(true_negatives, false_positives)
        self.precision   = self.calc_precision(true_positives, false_positives)
        self.recall      = self.sensitivity
        self.accuracy    = self.calc_accuracy(true_positives, true_negatives, false_positives, false_negatives)
        self.f1_score    = self.calc_f1_score(true_positives, false_positives, false_negatives)

        """
        print("tp:", true_positives, "fp:", false_positives, "tn:", true_negatives, "fn:", false_negatives)

        print("sensitivity:", self.sensitivity, "\nspecificity:", self.specificity, "\nprecision:", self.precision,
              "\nrecall:", self.recall, "\naccuracy:", self.accuracy, "\nf1:", self.f1_score)
        """

    def calc_sensitivity(self, true_positives, false_negatives):
        # True positive rate, sensitivity, recall
        return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0.0

    def calc_specificity(self, true_negatives, false_positives):
        # True negative rate, specificity
        return true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) != 0 else 0.0

    def calc_precision(self, true_positives, false_positives):
        # Precision, positive predictive value
        return true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0.0

    def calc_accuracy(self, true_positives, true_negatives, false_positives, false_negatives):
        # Accuracy
        return (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives) if (true_positives + false_positives + true_negatives + false_negatives) != 0 else 0.0

    def calc_f1_score(self, true_positives, false_positives, false_negatives):
        # F1-score
        return (2 * true_positives) / (2 * true_positives + false_positives + false_negatives) if (2 * true_positives + false_positives + false_negatives) != 0 else 1.0

    def return_metrics(self):
        return {"sensitivity": self.sensitivity, "specificity": self.specificity, "precision": self.precision, "recall": self.recall, "accuracy": self.accuracy, "f1_score": self.f1_score}


class H1():
    """
    For H1, we evaluate the performances of the different, employed models in reconstructing multiple subsystems.
    We use L2 loss for evaluating the performances.
    """
    def __init__(self, results_dict, ds_list, model_list):
        aggr_dict = self.aggregate_results_dict(results_dict, model_list, ds_list)
        stats_dict = self.calc_statistics(aggr_dict, ds_list)
        self.df_joints = self.to_df(stats_dict, modus="per_joint")
        self.df_files  = self.to_df(stats_dict, modus="per_file")

    def aggregate_results_dict(self, results_dict, model_list, ds_list):
        """
        concatenating all runs over different seeds into an aggregated dict.
        """
        aggr_dict = {}

        for ds in ds_list:
            for model in model_list:
                aggr_dict[model + ds] = {}
                for key in results_dict:
                    if model + ds in key:
                        normal_ind = results_dict[key]["normal_ind"]
                        normal_mean = results_dict[key]["normal_mean"]

                        nan_indices_ind = [(i, j) for i, sublist in enumerate(normal_ind) for j, value in enumerate(sublist) if math.isnan(value)]
                        nan_indices_mean = [index for index, value in enumerate(normal_mean) if math.isnan(value)]

                        if nan_indices_ind or nan_indices_mean:
                            continue

                        if "normal_ind" not in aggr_dict[model + ds]:
                            aggr_dict[model + ds]["normal_ind"] = normal_ind
                            aggr_dict[model + ds]["normal_mean"] = normal_mean
                        else:
                            aggr_dict[model + ds]["normal_ind"].extend(normal_ind)
                            aggr_dict[model + ds]["normal_mean"].extend(normal_mean)
        return aggr_dict

    def calc_statistics(self, aggr_dict, ds_list):
        """
        calculating mean, std, and var for reconstruction losses over individual joints and files + over files.
        """
        stat_dict = {}

        for ds in ds_list:
            for model in aggr_dict:
                if ds in model:
                    stat_dict[model] = {}

                    L2_per_joint_per_file = np.array(aggr_dict[model]["normal_ind"])
                    if ds == "_e": L2_per_joint_per_file = np.delete(L2_per_joint_per_file, 0, axis=1)
                    L2_per_file = np.array(aggr_dict[model]["normal_mean"])

                    mean_per_joint = np.mean(L2_per_joint_per_file, axis=0)
                    std_per_joint  = np.std(L2_per_joint_per_file, axis=0)
                    var_per_joint  = np.var(L2_per_joint_per_file, axis=0)

                    mean_per_file = np.mean(L2_per_file)
                    std_per_file  = np.std(L2_per_file)
                    var_per_file  = np.std(L2_per_file)

                    stat_dict[model] = {"per_joint": {"mean": mean_per_joint, "std": std_per_joint, "var": var_per_joint},
                                        "per_file":{"mean": mean_per_file, "std": std_per_file, "var": var_per_file}}
        return stat_dict

    def to_df(self, stats_dict, modus="per_file"):
        """
        Translating statistics dictionary into dataframe for joint-wise and file-wise cases.
        """
        df = pd.DataFrame()

        if modus == "per_joint":
            for model in stats_dict:
                if "_e" in model:
                    for elem in range(len(stats_dict[model]["per_joint"]["mean"])):
                        df.at[model[:-2], "$ds_e j_" + str(elem) + "$ mean"] = stats_dict[model]["per_joint"]["mean"][elem]
                        df.at[model[:-2], "$ds_e j_" + str(elem) + "$ var"]  = stats_dict[model]["per_joint"]["var"][elem]
                elif "_p" in model:
                    for elem in range(len(stats_dict[model]["per_joint"]["mean"])):
                        df.at[model[:-2], "$ds_m j_" + str(elem) + "$ mean"] = stats_dict[model]["per_joint"]["mean"][elem]
                        df.at[model[:-2], "$ds_m j_" + str(elem) + "$ var"]  = stats_dict[model]["per_joint"]["var"][elem]

        if modus == "per_file":
            for model in stats_dict:
                if "_e" in model:
                    df.at[model[:-2], "$ds_e$"] = stats_dict[model]["per_file"]["mean"]
                    df.at[model[:-2], "$ds_e var$"] = stats_dict[model]["per_file"]["var"]
                elif "_p" in model:
                    df.at[model[:-2], "$ds_m$"] = stats_dict[model]["per_file"]["mean"]
                    df.at[model[:-2], "$ds_m var$"] = stats_dict[model]["per_file"]["var"]
        return df


class H2():
    """
    For evaluating H2, we classify anomalies on all test-sets and each individual anomaly case.
    An anomaly is labeled as anomaly, when the threshold is 2*\sigma of the normal mean of the loss-value.
    Evaluation metrics are sensitivity, specificity, and F1.
    """
    def __init__(self, results_dict, model_list, ds_list, anomaly_list):
        aggr_dict = self.aggregate_results(results_dict, model_list, ds_list, anomaly_list)
        metrics_dict = self.calc_metrics(aggr_dict)
        anom_dict = self.label_anomalies(aggr_dict, metrics_dict)
        stats_dict = self.calc_statistics_global(anom_dict)
        self.df_files = self.to_df(stats_dict)

    def aggregate_results(self, results_dict, model_list, ds_list, anomaly_list, seed=42):
        aggr_dict = {}
        for ds in ds_list:
            for model in model_list:
                aggr_dict[model + ds] = {}
                for key in results_dict:
                    if model + ds in key and str(seed) in key:
                        for anomaly in anomaly_list:

                            l2_per_joint = results_dict[key][anomaly + "_ind"]  # for individual channels
                            nan_indices_ind = [(i, j) for i, sublist in enumerate(l2_per_joint) for j, value in enumerate(sublist) if math.isnan(value)]
                            if nan_indices_ind:
                                continue

                            if anomaly not in aggr_dict[model + ds]:
                                aggr_dict[model + ds][anomaly] = l2_per_joint
                            else:
                                aggr_dict[model + ds][anomaly].extend(l2_per_joint)
        return aggr_dict

    def calc_metrics(self, aggr_dict):
        metrics_dict = {}
        for model in aggr_dict:
            results = np.array(aggr_dict[model]["normal"])

            mean = np.mean(results, axis=0)
            std  = np.std(results, axis=0)
            var  = np.var(results, axis=0)

            metrics_dict[model] = {"mean": mean, "std": std, "var": var}
        return metrics_dict

    def label_anomalies(self, aggr_dict, metrics_dict):
        anom_dict = {}

        for model in aggr_dict.keys():
            anom_dict[model] = {}

            normal_mean = metrics_dict[model]["mean"]
            normal_std  = metrics_dict[model]["std"]

            for anomaly in aggr_dict[model]:
                anom_dict[model][anomaly] = {}

                elem_wise_list, anom_list = [], []
                for result in aggr_dict[model][anomaly]:

                    elem_list, anom = [], 0
                    for mean_pred, mean_gt, std_gt in zip(result, normal_mean, normal_std):
                        if (mean_gt - 2*std_gt) < mean_pred > (mean_gt + 2*std_gt):
                            elem_list.append(1)
                            anom = 1
                        else:
                            elem_list.append(0)

                        elem_wise_list.append(elem_list)
                    anom_list.append(anom)

                anom_dict[model][anomaly] = {"per_joint": elem_wise_list, "per_file": anom_list}
        return anom_dict

    def calc_statistics_per_anomaly(self, anom_dict):
        stats_dict = {}

        for model in anom_dict:
            stats_dict[model] = {}
            for anomaly in anom_dict[model]:
                pred_ind = np.array(anom_dict[model][anomaly]["per_joint"])
                pred_all = np.array(anom_dict[model][anomaly]["per_file"])

                if anomaly == "normal":
                    gt_ind = np.zeros(shape=pred_ind.shape)
                    gt_all = np.zeros(shape=pred_all.shape)
                else:
                    gt_ind = np.ones(shape=pred_ind.shape)
                    gt_all = np.ones(shape=pred_all.shape)

                m_all = Metrics(gt_all, pred_all)

                stats_dict[model][anomaly]["per_file"] = {"sensitivity": m_all.sensitivity, "specificity": m_all.specificity, "precision": m_all.precision,
                                                          "recall": m_all.recall, "accurracy": m_all.accuracy, "f1_score": m_all.f1_score}
        return stats_dict

    def calc_statistics_global(self, anom_dict):
        stats_dict = {}

        for model in anom_dict:
            glob_pred_all, glob_gt_all = np.array([]), np.array([])
            stats_dict[model] = {"per_joint":{}, "per_file":{}}
            for anomaly in anom_dict[model]:
                pred_all = np.array(anom_dict[model][anomaly]["per_file"])

                if anomaly == "normal":
                    gt_all = np.zeros(shape=pred_all.shape)
                else:
                    gt_all = np.ones(shape=pred_all.shape)

                glob_pred_all, glob_gt_all = np.concatenate((glob_pred_all, pred_all)), np.concatenate((glob_gt_all, gt_all))

            m_all = Metrics(glob_gt_all, glob_pred_all)
            stats_dict[model]["per_file"] = {"sensitivity": m_all.sensitivity, "specificity": m_all.specificity, "precision": m_all.precision,
                                            "recall": m_all.recall, "accurracy": m_all.accuracy, "f1_score": m_all.f1_score}
        return stats_dict

    def to_df(self, stats_dict, modus="per_file"):
        df = pd.DataFrame()

        if modus == "per_joint":
            for model in stats_dict:
                if "_e" in model:
                    for elem in range(len(stats_dict[model]["per_joint"])):
                        df.at[model[:-2], "sensitivity_j" + str(elem) + "_e"] = stats_dict[model]["per_joint"][elem]["sensitivity"]
                        df.at[model[:-2], "specificity_j" + str(elem) + "_e"] = stats_dict[model]["per_joint"][elem]["specificity"]
                        df.at[model[:-2], "f1_score_j"    + str(elem) + "_e"] = stats_dict[model]["per_joint"][elem]["f1_score"]
                elif "_p" in model:
                    for elem in range(len(stats_dict[model]["per_joint"]["mean"])):
                        df.at[model[:-2], "sensitivity_j" + str(elem) + "_m"] = stats_dict[model]["per_joint"][elem]["sensitivity"]
                        df.at[model[:-2], "specificity_j" + str(elem) + "_m"] = stats_dict[model]["per_joint"][elem]["specificity"]
                        df.at[model[:-2], "f1_score_j"    + str(elem) + "_m"] = stats_dict[model]["per_joint"][elem]["f1_score"]

        if modus == "per_file":
            for model in stats_dict:
                if "_e" in model:
                    df.at[model[:-2], "sensitivity" + "_e"] = stats_dict[model]["per_file"]["sensitivity"]
                    df.at[model[:-2], "specificity" + "_e"] = stats_dict[model]["per_file"]["specificity"]
                    df.at[model[:-2], "f1_score" + "_e"]    = stats_dict[model]["per_file"]["f1_score"]
                elif "_p" in model:
                    df.at[model[:-2], "sensitivity" + "_m"] = stats_dict[model]["per_file"]["sensitivity"]
                    df.at[model[:-2], "specificity" + "_m"] = stats_dict[model]["per_file"]["specificity"]
                    df.at[model[:-2], "f1_score" + "_m"]    = stats_dict[model]["per_file"]["f1_score"]
        return df


class H3():
    def __init__(self, results_dict, model_list, ds_list, anomaly_list):
        anom_dict = self.label_anomalies(results_dict, anomaly_list)
        metrics_dict = self.calc_metrics(anom_dict)
        self.df = self.to_df(ds_list, model_list,metrics_dict )

    def aggregate_results(self, results_dict, anomaly_list, seed=42):
        aggr_dict = {}

        for key in results_dict:
            if str(seed) in key:
                aggr_dict[key] = {}
                for anomaly in anomaly_list:
                    aggr_dict[key][anomaly] = {}
                    results_model = []

                    for elem in results_dict[key][anomaly + "_ind"]:
                        res_arr = np.array(elem)
                        if "_e" in key: res_arr = np.delete(res_arr, 0, axis=0)

                        results_model.append(res_arr)

                    aggr_dict[key][anomaly]["mean"] = np.mean(np.vstack(results_model), axis=0)
                    aggr_dict[key][anomaly]["var"] = np.var(np.vstack(results_model), axis=0)
                    aggr_dict[key][anomaly]["std"] = np.std(np.vstack(results_model), axis=0)
        return aggr_dict

    def label_anomalies(self, results_dict, anomaly_list, seed=42):
        anom_dict = {}
        aggr_dict = self.aggregate_results(results_dict, anomaly_list, seed)

        for key in results_dict:
            if str(seed) in key:
                anom_dict[key[:-8]] = {}
                for anomaly in anomaly_list:
                    case_list = []
                    res_list = results_dict[key][anomaly + "_ind"]

                    for res in res_list:
                        if "_e" in key: res = res[1:]

                        anom_list = []
                        normal_mean = aggr_dict[key]["normal"]["mean"]
                        normal_std = aggr_dict[key]["normal"]["std"]

                        for elem, norm_mean, norm_std in zip(res, normal_mean, normal_std):
                            if (norm_mean - 1 * norm_std) < elem > (norm_mean + 1 * norm_std):
                                anom_list.append(1)
                            else:
                                anom_list.append(0)
                        case_list.append(anom_list)
                    anom_dict[key[:-8]][anomaly] = case_list
        return anom_dict

    def calc_metrics(self, anom_dict):
        metrics_dict = {}
        for key in anom_dict:
            metrics_dict[key] = {}
            for anomaly in anom_dict[key]:
                metrics_dict[key][anomaly] = {}
                pred = np.array(anom_dict[key][anomaly])
                pred = pred.T

                if "normal" in anomaly:
                    gt = np.zeros(pred.shape)
                else:
                    gt = np.zeros(pred.shape)
                    gt[:3, :] = 1

                i = 0
                for elem_gt, elem_pred in zip(gt, pred):
                    m = Metrics(elem_gt, elem_pred)
                    metrics_dict[key][anomaly][i] = m.return_metrics()
                    i += 1

        return metrics_dict

    def to_df(self, ds_list, model_list, metrics_dict):
        df = pd.DataFrame()

        for ds in ds_list:
            for model in model_list:
                for joint in range(0,6):
                    #df.at[model, str(joint) + ds] = metrics_dict[model + ds][joint]["sensitivity"]
                    #df.at[model, str(joint) + ds] = metrics_dict[model + ds][joint]["specificity"]
                    #df.at[model, str(joint) + ds] = metrics_dict[model + ds][joint]["precision"]
                    #df.at[model, str(joint) + ds] = metrics_dict[model + ds][joint]["recall"]
                    #df.at[model, str(joint) + ds] = metrics_dict[model + ds][joint]["f1_score"]
                    df.at[model, "j_" + str(joint) + ds] = metrics_dict[model + ds]["push"][joint]["accuracy"]
        return df




# Quickrun
if __name__ == "__main__":
    MODEL_LIST = ["MonolithAE", "MonolithVAE", "Modular0AE", "Modular0VAE", "Modular1AE", "Modular1VAE", "Modular2AE",
                  "Modular2VAE"]
    CASES = ["normal", "weight", "drop", "stop", "out", "push", "speed", "wrench"]
    DS_LIST = ["_e", "_p"]
    EVAL_PATH = "../logs/repl_studies/logs/train/"  # "../logs_test/logs/"

    RESULTS = ImportResults(path=EVAL_PATH, model_list=MODEL_LIST, ds_list=DS_LIST, anomaly_list=CASES)

    h1_res = H1(results_dict=RESULTS.glob_res, model_list=MODEL_LIST, ds_list=DS_LIST)
    #h1_res.df_files

    h2_res = H2(results_dict=RESULTS.glob_res, model_list=MODEL_LIST, ds_list=DS_LIST, anomaly_list=CASES)
    #h2_res.df_files

    h3_res = H3(results_dict=RESULTS.glob_res, model_list=MODEL_LIST, ds_list=DS_LIST, anomaly_list=CASES)
    h3_res.df