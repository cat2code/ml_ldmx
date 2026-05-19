try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is missing.
    tqdm = None


class SimpleProgress:
    def __init__(self, iterable, total=None, desc="", unit="item"):
        self.iterable = iterable
        self.total = total
        self.desc = desc
        self.unit = unit

    def __iter__(self):
        for idx, item in enumerate(self.iterable, start=1):
            if self.total and (idx == 1 or idx == self.total or idx % max(1, self.total // 10) == 0):
                print(f"{self.desc}: {self.unit} {idx}/{self.total}")
            yield item

    def set_postfix(self, **_kwargs):
        return None


def make_progress(iterable, total=None, desc="", disable=False, unit="event"):
    if disable:
        return iterable
    if tqdm is None:
        return SimpleProgress(iterable, total=total, desc=desc, unit=unit)
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False, unit=unit)
