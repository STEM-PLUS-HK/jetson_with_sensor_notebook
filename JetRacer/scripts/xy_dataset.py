import torch
import os
import glob
import json
import uuid
import PIL.Image
import torch.utils.data
import subprocess
import cv2
import numpy as np

# Filenames store raw mm. The model uses 0-300 mm linearly as 1 -> 0
# (0 mm = 1 closest, 300 mm = 0). Anything outside that band is ignored (0).
TOF_LINEAR_MAX_MM = 300.0


def normalize_tof_mm(mm, max_mm=TOF_LINEAR_MAX_MM):
    mm = float(mm)
    if mm < 0.0 or mm > max_mm:
        return 0.0
    return 1.0 - (mm / max_mm)


def normalize_tof_tensor(sensors, max_mm=TOF_LINEAR_MAX_MM):
    sensors = sensors.float()
    inside = (sensors >= 0) & (sensors <= max_mm)
    return torch.where(inside, 1.0 - sensors / max_mm, torch.zeros_like(sensors))


# Center of the strip is 20 mm. The three lines are adjustable and saved.
BAR_NEAR_MM = 20
_BAR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bar_thresholds.json")
_BAR_GREEN = (0, 180, 0)     # farther than 2 car lengths
_BAR_YELLOW = (0, 255, 255)  # 2 car lengths down to 1 car length
_BAR_ORANGE = (0, 140, 255)  # 1 car length down to the car case
_BAR_RED = (0, 0, 255)       # car case down to 20 mm, and the center line


def default_bar_thresholds():
    return {"two_car": 400, "one_car": 200, "car_case": 50}


def _validate_bar_thresholds(data):
    try:
        two = int(data["two_car"])
        one = int(data["one_car"])
        case = int(data["car_case"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("need 2 car lengths, 1 car length, and car case")
    if not (BAR_NEAR_MM < case < one < two):
        raise ValueError("need 20 < car case < 1 car length < 2 car lengths")
    return {"two_car": two, "one_car": one, "car_case": case}


def load_bar_thresholds(path=None):
    """Read the saved lines. Missing or broken file keeps 400 / 200 / 50."""
    global BAR_THRESH
    path = _BAR_FILE if path is None else path
    BAR_THRESH = default_bar_thresholds()
    try:
        with open(path) as f:
            BAR_THRESH = _validate_bar_thresholds(json.load(f))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return dict(BAR_THRESH)


def save_bar_thresholds(two_car, one_car, car_case, path=None):
    """Store the three lines. The next notebook start loads this file."""
    global BAR_THRESH
    cleaned = _validate_bar_thresholds({
        "two_car": two_car, "one_car": one_car, "car_case": car_case,
    })
    path = _BAR_FILE if path is None else path
    with open(path, "w") as f:
        json.dump(cleaned, f)
        f.write("\n")
    BAR_THRESH = cleaned
    return dict(BAR_THRESH)


BAR_THRESH = default_bar_thresholds()
load_bar_thresholds()


def _bar_pos(mm, span, two, near):
    """How many columns to fill from the outer edge toward the center."""
    margin = max(4, span // 5)  # green cap: farther than 2 car lengths
    mm = float(mm)
    if mm >= two:
        return margin
    if mm <= near:
        return span
    t = (two - mm) / float(two - near)
    return max(margin, min(span, int(round(margin + t * (span - margin)))))


def tof_clearance_bar(left_mm, right_mm, width, height=20):
    """BGR strip. Left sensor grows from the left edge, right sensor from the right.

    The center red line is 20 mm and is never painted over. Each distance band
    keeps its own color: green, then yellow, then orange, then red. The live
    reading only decides how far that stack extends. A black tick marks
    2 car lengths, 1 car length, and the car case.
    """
    width = max(int(width), 32)
    height = max(int(height), 1)
    bar = np.full((height, width, 3), 255, dtype=np.uint8)
    mid = width // 2
    # mid-2 and mid+1 stay white. mid-1 and mid are the 20 mm line.
    left_cols = list(range(0, mid - 2))
    right_cols = list(range(width - 1, mid + 1, -1))
    th = BAR_THRESH
    two, one, case = float(th["two_car"]), float(th["one_car"]), float(th["car_case"])
    if not (BAR_NEAR_MM < case < one < two):
        two, one, case = 400.0, 200.0, 50.0
    near = float(BAR_NEAR_MM)

    def paint(cols, mm):
        if mm is None:
            mm = two + 1.0
        mm = float(mm)
        span = len(cols)
        n = _bar_pos(mm, span, two, near)
        cuts = (
            _bar_pos(two, span, two, near),
            _bar_pos(one, span, two, near),
            _bar_pos(case, span, two, near),
        )
        for i, x in enumerate(cols):
            if i in cuts:
                bar[:, x] = (0, 0, 0)
                continue
            if i >= n:
                continue
            if i < cuts[0]:
                color = _BAR_GREEN
            elif i < cuts[1]:
                color = _BAR_YELLOW
            elif i < cuts[2]:
                color = _BAR_ORANGE
            else:
                color = _BAR_RED
            bar[:, x] = color

    paint(left_cols, left_mm)
    paint(right_cols, right_mm)
    bar[:, mid - 1] = _BAR_RED
    bar[:, mid] = _BAR_RED
    return bar


def find_dataset_roots(categories, cwd='.'):
    """Folders that actually contain <category>/*.jpg.

    Matches 1_data_collection_sensor.ipynb (`road_following_A|B/<category>/`),
    an unzipped `./<category>/`, or a kernel started inside the category folder.
    """
    cwd = os.path.abspath(cwd)
    bases = [cwd]
    # train_model.ipynb lives in scripts/; jpgs live one level up with the lab notebooks
    if os.path.basename(cwd) == 'scripts':
        bases.append(os.path.dirname(cwd))
    roots, seen = [], set()

    def add(root):
        root = os.path.abspath(root)
        if root in seen:
            return
        if any(glob.glob(os.path.join(root, cat, '*.jpg')) for cat in categories):
            seen.add(root)
            roots.append(root)

    for base in bases:
        for name in ('A', 'B'):
            add(os.path.join(base, 'road_following_' + name))
        add(base)
        if os.path.basename(base) in categories:
            add(os.path.dirname(base))
    return roots


class XYDataset(torch.utils.data.Dataset):
    def __init__(self, directory, categories, transform=None, random_hflip=False,
                 return_sensors=False):
        super(XYDataset, self).__init__()
        self.directory = directory
        self.categories = categories
        self.transform = transform
        self.refresh()
        self.random_hflip = random_hflip
        self.return_sensors = return_sensors

    def _roots(self):
        d = self.directory
        return list(d) if isinstance(d, (list, tuple)) else [d]
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image = cv2.imread(ann['image_path'], cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError('failed to read %s' % ann['image_path'])
        image = PIL.Image.fromarray(image)
        width = image.width
        height = image.height
        if self.transform is not None:
            image = self.transform(image)
        
        x = 2.0 * (ann['x'] / width - 0.5) # -1 left, +1 right
        y = 2.0 * (ann['y'] / height - 0.5) # -1 top, +1 bottom
        tof_left = 0.0 if ann['tof_left'] is None else float(ann['tof_left'])
        tof_right = 0.0 if ann['tof_right'] is None else float(ann['tof_right'])
        
        if self.random_hflip and float(np.random.random()) > 0.5:
            image = torch.from_numpy(image.numpy()[..., ::-1].copy())
            x = -x
            tof_left, tof_right = tof_right, tof_left

        sample = (image, ann['category_index'], torch.Tensor([x, y]))
        if not self.return_sensors:
            return sample

        # annotations keep raw mm; convert only when feeding the model
        sensors = normalize_tof_tensor(torch.tensor([tof_left, tof_right]))
        return sample + (sensors,)
    
    def _parse(self, path):
        basename = os.path.basename(path)
        name, _ = os.path.splitext(basename)
        items = name.split('_')
        # official NVIDIA names start with xy_; data-collection names do not
        if items and items[0] == 'xy':
            items = items[1:]
        x = int(items[0])
        y = int(items[1])
        tof_left = tof_right = None
        # apex_sensor: <x>_<y>_<tof_left>_<tof_right>_<uuid>
        if len(items) >= 4 and items[2].lstrip('-').isdigit() and items[3].lstrip('-').isdigit():
            tof_left = int(items[2])
            tof_right = int(items[3])
        return x, y, tof_left, tof_right
        
    def refresh(self):
        self.annotations = []
        for root in self._roots():
            for category in self.categories:
                category_index = self.categories.index(category)
                for image_path in glob.glob(os.path.join(root, category, '*.jpg')):
                    try:
                        x, y, tof_left, tof_right = self._parse(image_path)
                    except (ValueError, IndexError, TypeError):
                        continue
                    self.annotations += [{
                        'image_path': image_path,
                        'category_index': category_index,
                        'category': category,
                        'x': x,
                        'y': y,
                        'tof_left': tof_left,
                        'tof_right': tof_right
                    }]
        
    def save_entry(self, category, image, x, y, tof_left=None, tof_right=None):
        category_dir = os.path.join(self._roots()[0], category)
        if not os.path.exists(category_dir):
            subprocess.call(['mkdir', '-p', category_dir])

        if tof_left is not None and tof_right is not None:
            # apex_sensor: <x>_<y>_<tof_left>_<tof_right>_<uuid>.jpg
            filename = '%d_%d_%d_%d_%s.jpg' % (x, y, int(tof_left), int(tof_right), str(uuid.uuid1()))
        else:
            filename = '%d_%d_%s.jpg' % (x, y, str(uuid.uuid1()))
        
        image_path = os.path.join(category_dir, filename)
        cv2.imwrite(image_path, image)
        self.refresh()
        
    def get_count(self, category):
        i = 0
        for a in self.annotations:
            if a['category'] == category:
                i += 1
        return i


class HeatmapGenerator():
    def __init__(self, shape, std):
        self.shape = shape
        self.std = std
        self.idx0 = torch.linspace(-1.0, 1.0, self.shape[0]).reshape(self.shape[0], 1)
        self.idx1 = torch.linspace(-1.0, 1.0, self.shape[1]).reshape(1, self.shape[1])
        self.std = std
        
    def generate_heatmap(self, xy):
        x = xy[0]
        y = xy[1]
        heatmap = torch.zeros(self.shape)
        heatmap -= (self.idx0 - y)**2 / (self.std**2)
        heatmap -= (self.idx1 - x)**2 / (self.std**2)
        heatmap = torch.exp(heatmap)
        return heatmap


if __name__ == '__main__':
    p = XYDataset.__new__(XYDataset)
    assert p._parse('apex_sensor/6_102_147_55_034a0b83-b56a-11f1.jpg') == (6, 102, 147, 55)
    assert p._parse('apex_sensor/xy_118_107_289_532_eafc3a78.jpg') == (118, 107, 289, 532)
    def _px(image, x):
        return tuple(int(v) for v in image[4, x])

    close = tof_clearance_bar(10, 10, 224, 8)
    assert close.shape == (8, 224, 3)
    assert _px(close, 111) == _BAR_RED and _px(close, 112) == _BAR_RED
    assert _px(close, 110) == (255, 255, 255) and _px(close, 113) == (255, 255, 255)
    assert _px(close, 0) == _BAR_GREEN and _px(close, 40) == _BAR_YELLOW and _px(close, 109) == _BAR_RED
    far = tof_clearance_bar(2000, 2000, 224, 8)
    assert _px(far, 0) == _BAR_GREEN and _px(far, 80) == (255, 255, 255)
    assert _px(far, 111) == _BAR_RED
    zoned = tof_clearance_bar(125, 2000, 224, 8)
    assert _px(zoned, 0) == _BAR_GREEN and _px(zoned, 40) == _BAR_YELLOW and _px(zoned, 75) == _BAR_ORANGE
    assert _px(zoned, 100) == (255, 255, 255) and _px(zoned, -1) == _BAR_GREEN

    def left_orange(image):
        return [i for i in range(image.shape[1] // 2) if _px(image, i) == _BAR_ORANGE]

    BAR_THRESH = {"two_car": 400, "one_car": 200, "car_case": 50}
    at_50 = left_orange(tof_clearance_bar(10, 10, 224, 8))
    BAR_THRESH = {"two_car": 400, "one_car": 200, "car_case": 120}
    at_120 = left_orange(tof_clearance_bar(10, 10, 224, 8))
    assert at_50 and at_120 and max(at_120) < max(at_50), (at_50, at_120)
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        assert save_bar_thresholds(500, 250, 80, path=tmp)["car_case"] == 80
        assert load_bar_thresholds(path=tmp) == {"two_car": 500, "one_car": 250, "car_case": 80}
        try:
            save_bar_thresholds(100, 80, 90, path=tmp)
            raise SystemExit("expected reject")
        except ValueError:
            pass
        assert load_bar_thresholds(path=tmp)["two_car"] == 500
    finally:
        os.remove(tmp)
        load_bar_thresholds()
    print('xy_dataset parse ok')