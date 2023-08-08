from time import time

from src.ddp.ddp_utils import dprint


class TqdmWrapper:
    """
    tqdm hotfix that uses dprint instead of print for distributed printing.
    Does not overwrite the previous line, but instead prints a new line.
    """

    def __init__(self, iterable, desc=None, total=None, colour=None, postfix_dict=None):
        self.iterable = iterable
        self.desc = desc
        self.total = total if total else len(iterable) - 1
        self.colour = colour
        self.postfix_dict = postfix_dict or {}
        self.prefix = f"{desc}: " if desc else ""
        self.colour_dict = {
            "red": "\033[31m",
            "green": "\033[32m",
            "blue": "\033[34m",
            "yellow": "\033[33m"
        }

    def __iter__(self):
        colour_prefix = self.colour_dict.get(self.colour, "")
        colour_suffix = "\033[0m" if colour_prefix else ""
        iter_per_second = 0
        last_time = None
        for i, item in enumerate(self.iterable, start=1):
            if isinstance(last_time, float):
                iter_per_second = 1 / (time() - last_time)
            postfix_str = ', '.join(f'{k}: {v}' for k, v in self.postfix_dict.items())
            dprint(
                f"{colour_prefix}{self.prefix}{i}/{self.total if self.total else ''} | {iter_per_second:.2f} it/s | {postfix_str}{colour_suffix}",
                flush=True)
            last_time = time()
            yield item

    def set_postfix(self, postfix_dict=None, **kwargs):
        if postfix_dict:
            self.postfix_dict.update(postfix_dict)
        if kwargs:
            self.postfix_dict.update(kwargs)

    def close(self):
        pass


def tqdm(iterable, desc=None, total=None, colour=None, *args, **kwargs):
    return TqdmWrapper(iterable, desc=desc, total=total, colour=colour)
