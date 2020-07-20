import os


def get_save_path(*path):
    path = os.path.join(*path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_file_name(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def get_file_ext(filepath):
    return os.path.splitext(os.path.basename(filepath))[1]

if __name__ == "__main__":
    def save_to(to):
        return get_save_path("a", "b", to)
    print(save_to("to.png"))