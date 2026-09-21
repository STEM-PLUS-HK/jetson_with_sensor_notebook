import torch
import os
import glob
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
    print('xy_dataset parse ok')