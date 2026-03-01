import os
import sys


ANSI_COLORS = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}
ANSI_RESET = "\033[0m"


def _supports_color(stream):
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(stream, "isatty") and stream.isatty()


class PrefixedConsoleLogger:
    def __init__(self, prefix, color="white", stream=None):
        self.prefix = prefix
        self.color = color.lower()
        self.stream = stream if stream is not None else sys.stdout
        self.enable_color = _supports_color(self.stream)

    def _prefix_text(self):
        raw_prefix = f"[{self.prefix}]"
        color_code = ANSI_COLORS.get(self.color, "")
        if self.enable_color and color_code:
            return f"{color_code}{raw_prefix}{ANSI_RESET}"
        return raw_prefix

    def __call__(self, *values, sep=" ", end="\n", flush=False):
        message = sep.join(str(v) for v in values)
        if message:
            text = f"{self._prefix_text()} {message}"
        else:
            text = self._prefix_text()
        print(text, end=end, flush=flush, file=self.stream)


def build_prefixed_logger(prefix, color="white", stream=None):
    return PrefixedConsoleLogger(prefix=prefix, color=color, stream=stream)
