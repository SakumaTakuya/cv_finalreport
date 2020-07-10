import os


def get_save_path(*path):
    path = os.path.join(*path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


if __name__ == "__main__":
    def save_to(to):
        return get_save_path("a", "b", to)
    print(save_to("to.png"))