from glob import glob
import numpy as np
import matplotlib.pyplot as plt

ppo_train_data = {"r": [], "l": [], "e": []}
for file in glob("./runs/train_ppo*.txt"):
    ppo_train_data["r"].append([])
    ppo_train_data["l"].append([])
    ppo_train_data["e"].append([])
    with open(file, "r") as f:
        for line in f:
            infos = line.split("|")
            reward = float(infos[-1].split(" ")[-1])
            loss = float(infos[2].split(" ")[-2])
            entropy = float(infos[3].split(":")[-1])
            ppo_train_data["r"][-1].append(reward)
            ppo_train_data["l"][-1].append(loss)
            ppo_train_data["e"][-1].append(entropy)

ppo_eval_data = []
for k, file in enumerate(glob("./runs/eval_ppo*.txt")):
    with open(file, "r") as f:
        for line in f:
            infos = line.split("|")
            env = int(infos[1].strip(" ").split("-")[-1])
            reward = float(infos[-1])
            if (env + 1) > len(ppo_eval_data):
                ppo_eval_data.append([[]])
            if k + 1 > len(ppo_eval_data[env]):
                ppo_eval_data[env].append([])
            ppo_eval_data[env][k].append(reward)

dqn_train = "./train_dqn.txt"
dqn_eval = "./eval_dqn.txt"

dqn_train_data = {
    "stm": {"r": [], "l": [], "e": []},
    "ltm": {"r": [], "l": [], "e": []},
}

for file in glob("./runs/train_dqn*.txt"):
    dqn_train_data["stm"]["r"].append([])
    dqn_train_data["stm"]["l"].append([])
    dqn_train_data["stm"]["e"].append([])
    dqn_train_data["ltm"]["r"].append([])
    dqn_train_data["ltm"]["l"].append([])
    dqn_train_data["ltm"]["e"].append([])
    with open(file, "r") as f:
        for line in f:
            infos = line.split("|")
            mode = infos[0].split(" ")[0]
            env = infos[0].split(" ")[2]
            loss = float(infos[1].split(":")[-1])
            entropy = float(infos[2].split(":")[-1])
            reward = float(infos[3].split(":")[-1])
            env_train_data = dqn_train_data[mode]
            env_train_data["r"][-1].append(reward)
            env_train_data["l"][-1].append(loss)
            env_train_data["e"][-1].append(entropy)


dqn_eval_data = {"stm": {}, "ltm": {}}

for k, file in enumerate(glob("./runs/eval_ppo*.txt")):
    with open(file, "r") as f:
        for line in f:
            infos = line.split("|")
            mode = infos[0].strip(" ")
            if mode in ["stm", "ltm"]:
                env = infos[1].split(" ")[0]
                test_env = infos[1].split(" ")[-2]
                r = float(infos[2])
                if test_env not in dqn_eval_data[mode]:
                    dqn_eval_data[mode][test_env] = []
                if (k + 1) > dqn_eval_data[mode][test_env]:
                    dqn_eval_data[mode][test_env].append([])
                dqn_eval_data[mode][test_env][-1].append(r)


def plot_data(
    x,
    y,
    seg,
    labelx="Number of Environment Steps",
    labely="Mean Acc. Reaward per Episode",
    yrange=[-410, 410],
):
    plt.plot(x, y, "tab:blue")
    plt.vlines(seg, yrange[0], yrange[1], linestyles="--", colors="black")
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.xlim([0, x.max()])
    plt.ylim(yrange)
    plt.show()


idx = np.arange(len(dqn_train_data["stm"]["r"])) * 2
transitions = (1 + np.arange(len(idx) / 200 - 1)) * 400
data = 100 * np.array(dqn_train_data["stm"]["r"])
plot_data(idx, data, transitions)

idx = np.arange(len(dqn_train_data["ltm"]["r"])) * 2
transitions = (1 + np.arange(len(idx) / 25 - 1)) * 50
data = 100 * np.array(dqn_train_data["ltm"]["r"])
plot_data(idx, data, transitions)

for env in dqn_eval_data["ltm"]:
    print(env)
    data = np.array(dqn_eval_data["ltm"][env])
    print(data.shape)
    idx = np.arange(data.shape[0])
    transitions = (1 + np.arange(data.shape[0] / 50 - 1)) * 50
    plot_data(idx, data, transitions)
