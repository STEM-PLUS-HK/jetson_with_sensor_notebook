"""Notebook widgets for data collection and training.

The notebook keeps the calls. This file keeps the widget wiring.
"""
import threading

_stop_poll = None
_tof_lock = threading.Lock()


def epoch_tick_step(max_epoch):
    """Pick a clean integer step so x labels stay readable."""
    if max_epoch <= 8:
        return 1
    best = 1
    for step in (2, 5, 10, 20, 25, 50):
        count = max_epoch / step
        if 3 <= count <= 8:
            best = step
    if best > 1:
        return best
    for step in (2, 5, 10, 20, 25, 50):
        if max_epoch / step <= 8:
            return step
    return max(50, (max_epoch + 7) // 8)


class CollectionUI:
    """Live camera, click-to-save, and the ToF strips."""

    def __init__(self, camera, datasets, dataset_names, tof):
        import cv2
        import ipywidgets
        import traitlets
        from jetcam.utils import bgr8_to_jpeg
        from jupyter_clickable_image_widget import ClickableImageWidget
        from scripts.xy_dataset import load_bar_thresholds, tof_clearance_bar

        global _stop_poll
        load_bar_thresholds()
        if _stop_poll is not None:
            _stop_poll.set()
        self._stop = threading.Event()
        _stop_poll = self._stop

        self.camera = camera
        self.datasets = datasets
        self.tof = tof
        self.dataset = datasets[dataset_names[0]]
        self.tof_latest = (0, 0)
        self._cv2 = cv2
        self._jpeg = bgr8_to_jpeg
        self._bar = tof_clearance_bar

        camera.unobserve_all()
        self.camera_widget = ClickableImageWidget(width=camera.width, height=camera.height)
        self.snapshot_widget = ipywidgets.Image(width=camera.width, height=camera.height)
        traitlets.dlink((camera, "value"), (self.camera_widget, "value"), transform=bgr8_to_jpeg)

        self.dataset_widget = ipywidgets.Dropdown(options=dataset_names, description="dataset")
        self.category_widget = ipywidgets.Dropdown(
            options=self.dataset.categories, description="category")
        self.count_widget = ipywidgets.IntText(description="count")
        self.tof_left_widget = ipywidgets.IntText(description="ToF left")
        self.tof_right_widget = ipywidgets.IntText(description="ToF right")
        self.count_widget.value = self.dataset.get_count(self.category_widget.value)

        bar_h = 20
        self._bar_h = bar_h

        def _tof_bar():
            return ipywidgets.Image(
                format="png", width=camera.width, height=bar_h,
                layout=ipywidgets.Layout(
                    width="%dpx" % camera.width, height="%dpx" % bar_h,
                    border="1px solid #bbb", margin="4px 0 0 0"),
            )

        self.tof_bar_widget = _tof_bar()
        self.snapshot_bar_widget = _tof_bar()
        self.live_bar_widget = _tof_bar()

        self.dataset_widget.observe(self._set_dataset, names="value")
        self.category_widget.observe(self._update_counts, names="value")
        self.camera_widget.on_msg(self._save_snapshot)
        threading.Thread(target=self._poll, args=(self._stop,), daemon=True).start()
        blank = self._bar_png(2000, 2000)
        self.tof_bar_widget.value = self.live_bar_widget.value = self.snapshot_bar_widget.value = blank

        self.widget = ipywidgets.VBox([
            ipywidgets.HBox([
                ipywidgets.VBox([self.camera_widget, self.tof_bar_widget]),
                ipywidgets.VBox([self.snapshot_widget, self.snapshot_bar_widget]),
            ], layout=ipywidgets.Layout(align_items="flex-start")),
            self.dataset_widget,
            self.category_widget,
            self.count_widget,
            ipywidgets.HBox([self.tof_left_widget, self.tof_right_widget]),
        ])

    def show(self):
        from IPython.display import display
        display(self.widget)

    def _bar_png(self, left_mm, right_mm):
        _ok, buf = self._cv2.imencode(
            ".png", self._bar(left_mm, right_mm, self.camera.width, self._bar_h))
        return bytes(buf)

    def _show_tof(self, left_mm, right_mm):
        self.tof_left_widget.value = int(left_mm)
        self.tof_right_widget.value = int(right_mm)
        png = self._bar_png(left_mm, right_mm)
        self.tof_bar_widget.value = png
        self.live_bar_widget.value = png

    def read_tof_pair(self):
        with _tof_lock:
            left, right = (int(v) for v in self.tof.read_mm())
            self.tof_latest = (left, right)
            return self.tof_latest

    def _set_dataset(self, change):
        self.dataset = self.datasets[change["new"]]
        self.count_widget.value = self.dataset.get_count(self.category_widget.value)

    def _update_counts(self, change):
        self.count_widget.value = self.dataset.get_count(change["new"])

    def _save_snapshot(self, _, content, msg):
        if content["event"] != "click":
            return
        data = content["eventData"]
        x = data["offsetX"]
        y = data["offsetY"]
        tof_left, tof_right = self.read_tof_pair()
        self._show_tof(tof_left, tof_right)
        self.dataset.save_entry(
            self.category_widget.value, self.camera.value, x, y,
            tof_left=tof_left, tof_right=tof_right)
        snapshot = self.camera.value.copy()
        snapshot = self._cv2.circle(snapshot, (x, y), 8, (0, 255, 0), 3)
        self.snapshot_widget.value = self._jpeg(snapshot)
        self.snapshot_bar_widget.value = self._bar_png(tof_left, tof_right)
        self.count_widget.value = self.dataset.get_count(self.category_widget.value)

    def _poll(self, stop):
        while not stop.is_set():
            try:
                left, right = self.read_tof_pair()
            except Exception:
                stop.wait(0.3)
                continue

            def _push(left=left, right=right):
                self._show_tof(left, right)

            try:
                get_ipython().kernel.io_loop.add_callback(_push)
            except Exception:
                _push()
            stop.wait(0.05)


class TrainUI:
    """Bar, epoch, and loss widgets. The training loop stays in the notebook."""

    def __init__(self):
        import ipywidgets
        from scripts.xy_dataset import load_bar_thresholds, save_bar_thresholds

        self._save_bar = save_bar_thresholds
        self.loss_epoch_values = []
        self.loss_values = []
        self.plot = ipywidgets.Output()

        thresholds = load_bar_thresholds()
        field = {"description_width": "initial"}
        self.one_car_widget = ipywidgets.IntText(
            description="1 car width", value=thresholds["one_car"], style=field,
            layout=ipywidgets.Layout(width="200px"))
        self.case_widget = ipywidgets.IntText(
            description="car case", value=thresholds["car_case"], style=field,
            layout=ipywidgets.Layout(width="160px"))
        self.bar_apply_button = ipywidgets.Button(description="apply")
        self.bar_apply_status = ipywidgets.HTML("")
        self.bar_apply_button.on_click(self._apply_bar)

        self.epochs_widget = ipywidgets.IntText(description="epochs", value=1)
        self.eval_button = ipywidgets.Button(description="evaluate")
        self.train_button = ipywidgets.Button(description="train")
        self.loss_widget = ipywidgets.FloatText(description="loss")
        self.progress_widget = ipywidgets.FloatProgress(min=0.0, max=1.0, description="progress")
        self.widget = ipywidgets.VBox([
            ipywidgets.HBox([
                self.one_car_widget, self.case_widget,
                self.bar_apply_button, self.bar_apply_status,
            ]),
            self.epochs_widget,
            self.progress_widget,
            self.loss_widget,
            ipywidgets.HBox([self.train_button, self.eval_button]),
        ])

    def show(self):
        from IPython.display import display
        display(self.widget)

    def _apply_bar(self, _):
        try:
            self._save_bar(self.one_car_widget.value, self.case_widget.value)
            self.bar_apply_status.value = "saved"
        except ValueError as exc:
            self.bar_apply_status.value = str(exc)

    def update_loss_plot(self):
        import matplotlib.pyplot as plt
        from IPython.display import clear_output

        with self.plot:
            clear_output(wait=True)
            if not self.loss_values:
                return
            max_epoch = max(self.loss_epoch_values)
            step = epoch_tick_step(max_epoch)
            xticks = list(range(step, max_epoch + 1, step))
            if xticks[-1] != max_epoch:
                xticks.append(max_epoch)
            plt.figure(figsize=(5, 3))
            plt.plot(self.loss_epoch_values, self.loss_values, marker="o")
            plt.xticks(xticks)
            plt.xlim(0.5, max_epoch + 0.5)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Live Training Loss")
            plt.grid(True)
            plt.show()

if __name__ == "__main__":
    assert epoch_tick_step(5) == 1
    assert epoch_tick_step(20) == 5
    assert epoch_tick_step(100) == 25
    print("collect_ui ok")
