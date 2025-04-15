import os
import os.path as osp
import yaml
import matplotlib.pyplot as plt


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

class ConfigNamespace:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNamespace(value))
            elif isinstance(value, list):
                setattr(self, key, [
                    ConfigNamespace(item) if isinstance(item, dict) else item
                    for item in value
                ])
            else:
                setattr(self, key, value)

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNamespace):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, ConfigNamespace) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def __repr__(self):
        return repr(self.__dict__)
# 加载函数
def load_config(path: str) -> ConfigNamespace:
    """
    Load YAML config file and return a ConfigNamespace object.

    Args:
        path (str): YAML file path.

    Returns:
        ConfigNamespace: Nested config with attribute-style and dict-style access.
    """
    if not osp.exists(path):
        raise FileNotFoundError(f"{path} not exists")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return ConfigNamespace(config)

if __name__ == "__main__":
    yaml_file = '../configs/ppo.yaml'
    args = load_config(yaml_file)
    # print(args)
    print(args.algo)
    print(args.algo.lr)
    print(args.algo.to_dict())