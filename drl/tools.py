import os
import os.path as osp
import yaml
import matplotlib.pyplot as plt
from types import SimpleNamespace

def plot(steps, model_save_dir, title="rl", x_label="Episode", y_label="Reward", step_interval=None):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.plot(steps)
    RunTime = len(steps)

    path = model_save_dir + '/RunTime' + str(RunTime) + '.jpg'
    if step_interval and len(steps) % step_interval == 0:
        plt.savefig(path)
        print(f'save fig in {path}')
    plt.pause(0.0000001)

def load_config(path):
    if not osp.exists(path):
        raise FileNotFoundError(f"{path} not exists")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    args = SimpleNamespace(**config)
    return args

if __name__ == "__main__":
    yaml_file = '../configs/ppo.yaml'
    args = load_config(yaml_file)
    # print(args)
    print(args.ppo)
    print(args.ppo["lr"])