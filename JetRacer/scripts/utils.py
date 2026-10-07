import os


def preprocess(image):
    # Torch stays out of this import. zip_dataset does not need it.
    import torch
    import torchvision.transforms as transforms
    import PIL.Image

    device = torch.device("cuda")
    if not hasattr(preprocess, "mean"):
        preprocess.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        preprocess.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(preprocess.mean[:, None, None]).div_(preprocess.std[:, None, None])
    return image[None, ...]


def enable_oled():
    """Turn the OLED stats page off, then on. The server is on this Jetson."""
    import requests
    base_url = "http://127.0.0.1:8000/stats"
    for action in ("off", "on"):
        response = requests.get("%s/%s" % (base_url, action), timeout=5)
        print("%s:%s" % (action.upper(), response.status_code))


def image_kind(path):
    """road_following_with_sensor if the name has x, y, left, right. Else road_following."""
    name, _ = os.path.splitext(os.path.basename(path))
    items = name.split("_")
    if items and items[0] == "xy":
        items = items[1:]
    if len(items) < 2 or not items[0].lstrip("-").isdigit() or not items[1].lstrip("-").isdigit():
        return None
    if len(items) >= 4 and items[2].lstrip("-").isdigit() and items[3].lstrip("-").isdigit():
        return "road_following_with_sensor"
    return "road_following"


def zip_dataset(folder, category, zip_path, output_model="pytorch"):
    """Write config.json beside the image folder, then zip both. Replaces existing files."""
    import glob
    import json
    import subprocess

    paths = glob.glob(os.path.join(folder, category, "*.jpg"))
    kinds = {}
    for path in paths:
        kind = image_kind(path)
        if kind:
            kinds[kind] = kinds.get(kind, 0) + 1
    if not kinds:
        raise RuntimeError("no x_y images in %s/%s" % (folder, category))
    if len(kinds) > 1:
        raise RuntimeError("mixed image names: %s" % kinds)

    config = {"version": 1, "output_model": output_model, "dataset": next(iter(kinds))}
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    if os.path.exists(zip_path):
        os.remove(zip_path)
    subprocess.check_call(
        ["zip", "-qr", os.path.abspath(zip_path), category, "config.json"],
        cwd=folder,
    )
    print(zip_path, config)
    return config


if __name__ == "__main__":
    assert image_kind("6_102_147_55_034a0b83-b56a-11f1.jpg") == "road_following_with_sensor"
    assert image_kind("118_107_034a0b83-b56a-11f1.jpg") == "road_following"
    assert image_kind("xy_118_107_289_532_eafc.jpg") == "road_following_with_sensor"
    print("utils image_kind ok")