import sys, os, time, datetime, random
import numpy as np


def import_with_auto_install(packages, scope=locals()):
    if isinstance(packages, str):
        packages = [packages]
    for package in packages:
        if package.find(":") != -1:
            package_import_name, package_pip_name = package.split(":")
        else:
            package_import_name, package_pip_name = package, package
        try:
            scope[package_import_name] = __import__(package_import_name)
        except ImportError:
            import subprocess

            subprocess.call(f"pip install {package_pip_name}", shell=True)
            scope[package_import_name] = __import__(package_import_name)


def get_option_list(argv):
    optionlist = []
    for arg1 in argv:
        if arg1[:2] == "--":
            argname = arg1.split("=")
            optionlist.append(argname[0].lstrip("-"))
    return optionlist


def parse_param_str(param_str):
    """parse a=b:c=d,e to {'a':b, 'c':'d,e'}"""
    params = param_str.split(":")

    name = None
    d = {}
    for pi, p in enumerate(params):
        try:
            k, v = p.split("=")
            if v.lower() == "true":
                v = 1
            elif v.lower() == "false":
                v = 0
            else:
                try:
                    v = int(v)
                except:
                    try:
                        v = float(v)
                    except:
                        if len(v) > 2 and v[0] == '"' and v[-1] == '"':
                            v = v[1:-1]
            d[k] = v
        except:
            if pi == 0:
                name = p
            else:
                color_print(f"ERROR: failed to parse parameter {p}. Ignored")
    return (name, d)


def parsemodopt(optstr=None):
    """This is used so the user can provide the name of a comparator, processor, etc. with options
    in a convenient form. It will parse "dot:normalize=1:negative=0" and return
    ("dot",{"normalize":1,"negative":0})"""

    if optstr is None or len(optstr) == 0:
        return (None, {})
    if optstr.lower() == "none":
        return None  # special case doesn't return a tuple

    op2 = optstr.split(":")
    if len(op2) == 1 or op2[1] == "":
        return (op2[0], {})  # name with no options

    r2 = {}
    for p in op2[1:]:
        try:
            k, v = p.split("=")
        except:
            print("ERROR: Command line parameter parsing failed on ", optstr)
            print("must have the form name:key=value:key=value")
            return (None, None)

        # 		v=v.replace("bdb%","bdb:")
        if v.lower() == "true":
            v = 1
        elif v.lower() == "false":
            v = 0
        else:
            try:
                v = int(v)
            except:
                try:
                    v = float(v)
                except:
                    if len(v) > 2 and v[0] == '"' and v[-1] == '"':
                        v = v[1:-1]
        r2[k] = v

    return (op2[0], r2)


def parsemodopt2(optstr):
    """parse a=b:c=d,e to {'a':b, 'c':'d,e'}"""
    op2 = optstr.split(":")

    r2 = {}
    for p in op2:
        try:
            k, v = p.split("=")
        except:
            color_print("ERROR: Command line parameter parsing failed on ", optstr)
            color_print("must have the form key=value:key=value")
            return {}
        if v.lower() == "true":
            v = 1
        elif v.lower() == "false":
            v = 0
        else:
            try:
                v = int(v)
            except:
                try:
                    v = float(v)
                except:
                    if len(v) > 2 and v[0] == '"' and v[-1] == '"':
                        v = v[1:-1]
        r2[k] = v
    return r2


def has_shiny():
    try:
        from helicon.lib import shiny

        return True
    except:
        return False


def has_streamlit():
    try:
        import streamlit

        return True
    except:
        return False


def color_print(*args, **kargs):
    color = "red"
    if "color" in kargs:
        color = str(kargs["color"]).lower()
        kargs.pop("color")
    end = "\n"
    if "end" in kargs:
        end = kargs["end"]
        kargs.pop("end")
    from rich.console import Console

    console = Console()
    console.print(*args, style=color, end=end, **kargs)


def available_cpu() -> int:
    import os

    if "SLURM_CPUS_ON_NODE" in os.environ:
        cpu = int(os.environ["SLURM_CPUS_ON_NODE"])
    else:
        import psutil

        cpu = max(1, int(psutil.cpu_count() * (1 - psutil.cpu_percent() / 100)))
    try:
        import numba

        cpu = min(cpu, int(numba.config.NUMBA_NUM_THREADS))
    except:
        pass
    return cpu


def omp_get_max_threads():
    import ctypes

    libomp = ctypes.CDLL("libgomp.so.1")  # for gcc on linux
    n = libomp.omp_get_max_threads()
    return n


def omp_set_num_threads(n):
    import ctypes

    libomp = ctypes.CDLL("libgomp.so.1")  # for gcc on linux
    if n <= 0:
        max_n = omp_get_max_threads()
        libomp.omp_set_num_threads(max_n)
    else:
        libomp.omp_set_num_threads(n)


def which(program, useCurrentDir=0):
    """unix which command equivalent"""
    location = None
    if program.find(os.sep) != -1:
        file = os.path.abspath(program)
        if os.path.exists(file) and os.access(file, os.X_OK):
            location = os.path.abspath(file)
    else:
        path = os.environ["PATH"]
        if useCurrentDir:
            path = ".:%s" % (path)
        dirs = path.split(":")
        for d in dirs:
            file = os.path.join(d, program)
            if os.path.exists(file) and os.access(file, os.X_OK):
                location = os.path.abspath(file)
                break
    return location


def download_url(url, target_file_name="", un_tar=0, remove_tarfile=0):
    import urllib.request, shutil, subprocess, os, time

    r = urllib.request.urlopen(url)
    if not target_file_name:
        target_file_name = url.split("/")[-1]
    with open(target_file_name, "wb") as f:
        shutil.copyfileobj(r, f)
    r.close()

    file_info = r.info()
    if "Last-Modified" in file_info:
        last_modified = file_info["Last-Modified"]
        timestamp = time.mktime(
            time.strptime(last_modified, "%a, %d %b %Y %H:%M:%S %Z")
        )
        os.utime(target_file_name, (timestamp, timestamp))

    final_target_file_name = target_file_name

    if un_tar:
        cmd = []
        if target_file_name.endswith(".tar.gz") or target_file_name.endswith(".tgz"):
            cmd = ["tar", "-xzvf", target_file_name]
            final_target_file_name = target_file_name.replace(".tar.gz", "")
        elif target_file_name.endswith(".tar.bz2") or target_file_name.endswith(".tbz"):
            cmd = ["tar", "-xjvf", target_file_name]
            final_target_file_name = target_file_name.replace(".tar.bz2", "")
        elif target_file_name.endswith(".tar"):
            cmd = ["tar", "-xvf", target_file_name]
            final_target_file_name = target_file_name.replace(".tar", "")
        elif target_file_name.endswith(".zip"):
            cmd = ["unzip", "-xzvf", target_file_name]
            final_target_file_name = target_file_name.replace(".zip", "")
        if cmd:
            subprocess.call(cmd)

        if remove_tarfile:
            os.remove(target_file_name)
    return final_target_file_name


def get_terminal_size():
    import shutil

    size = shutil.get_terminal_size()
    return (size.rows, size.columns)


def is_file_readable(filename):
    import os

    if not os.path.exists(filename):
        return False
    if os.path.isfile(filename):
        return os.access(filename, os.R_OK)
    else:
        return False


def is_file_writable(filename):
    import os

    if os.path.exists(filename):
        if os.path.isfile(filename):
            return os.access(filename, os.W_OK)
        else:
            return False
    pdir = os.path.dirname(filename)
    if not pdir:
        pdir = "."
    return os.access(pdir, os.W_OK)


def file_ready(filenames, wait=0, minSize=0):  # wait given seconds and check again
    if type(filenames) == type([]):
        ready = 1
        for f in filenames:
            if not (os.path.exists(f) and os.path.getsize(f)):
                ready = 0
                break
        return ready
    else:
        ready = os.path.exists(filenames) and os.path.getsize(filenames) > minSize
    if ready:
        return 1
    elif wait > 0:
        deadline = time.time() + wait
        delay = 1
        while time.time() <= deadline:
            time.sleep(delay)
            ready = file_ready(filenames, wait=0, minSize=minSize)
            if ready:
                return 1
            else:
                delay *= 2
                now = time.time()
                if now + delay > deadline:
                    delay = deadline - now
        return file_ready(filenames, wait=0)
    else:
        return 0


def convert_file_path(filenames, to="current", relpath_start=os.curdir):
    import pandas as pd

    if to == "current":
        return filenames
    assert to in "current absolute abs real relative rel shortest".split()
    assert isinstance(filenames, pd.Series)
    names = filenames.unique()
    mapping = {}
    import pathlib

    for name in names:
        p = pathlib.Path(name)
        p_abs = p.resolve()
        if to in ["real", "absolute", "abs"]:
            name2 = p_abs.as_posix()
        else:
            name2_rel = os.path.relpath(p_abs, relpath_start)
            if to in ["relative", "rel"]:
                name2 = name2_rel
            elif to == "shortest":
                if len(p_abs.as_posix()) < len(name2_rel):
                    name2 = p_abs.as_posix()
                else:
                    name2 = name2_rel
        if not (
            pathlib.Path(name2).exists()
            or (pathlib.Path(relpath_start) / pathlib.Path(name2)).exists()
        ):
            name2 = name
        mapping[name] = name2
    ret = filenames.map(mapping)
    return ret


def convert_dataframe_file_path(df, attr, to="current", relpath_start=os.curdir):
    if to == "current":
        return df[attr]
    if df[attr].iloc[0].find("@") != -1:
        tmp = df[attr].str.split("@", expand=True)
        indices, filenames = tmp.loc[:, 0], tmp.loc[:, 1]
        filenames = convert_file_path(filenames, to, relpath_start)
        ret = indices + "@" + filenames
    else:
        ret = convert_file_path(df[attr], to, relpath_start)
    return ret


def bytes2units(bytes, to=None, bsize=1024):
    units = {"k": 1, "m": 2, "g": 3, "t": 4, "p": 5, "e": 6}
    unitStr = {"k": "kB", "m": "MB", "g": "GB", "t": "TB", "p": "PB", "e": "EB"}
    if to is None:
        for u in units:
            x = bytes / (bsize ** units[u])
            if x < bsize:
                break
    else:
        u = to
        x = bytes / (bsize ** units[to])
    return (x, unitStr[u])


def ceil_power_of_10(n):
    if n < 0:
        raise ValueError(f"n={n} while n>0 is required")
    if n <= 1:
        return 10
    from math import ceil, log

    exp = log(n, 10)
    exp = ceil(exp)
    return 10**exp


def unique(inputList):
    ret = []
    for v in inputList:
        if v not in ret:
            ret.append(v)
    return ret


def assign_to_groups(numbers, group_size):
    """sort values and assign the values to groups"""
    # return a dict that maps values to group IDs

    from collections import defaultdict

    sorted_numbers = sorted(numbers)

    # Group duplicate values
    value_groups = defaultdict(list)
    for i, num in enumerate(sorted_numbers):
        value_groups[num].append(i)

    result = {}
    group_id = 1
    current_group = []
    current_group_size = 0

    # Group the numbers
    for num, indices in value_groups.items():
        if current_group_size + len(indices) > group_size:
            # If adding this set of duplicates exceeds the group size,
            # finalize the current group and start a new one
            if current_group:
                for value in current_group:
                    result[value] = group_id
                group_id += 1
            current_group = [num] * len(indices)
            current_group_size = len(indices)
        else:
            # Add the duplicates to the current group
            current_group.extend([num] * len(indices))
            current_group_size += len(indices)

        # If the group is full, finalize it
        if current_group_size == group_size:
            for value in current_group:
                result[value] = group_id
            group_id += 1
            current_group = []
            current_group_size = 0

    # Handle the last group
    if current_group:
        if len(current_group) < group_size // 2 and len(result) > 0:
            # Merge with the previous group if it's less than half the group size
            prev_group_id = max(result.values())
            for value in current_group:
                result[value] = prev_group_id
        else:
            # Add as a new group
            for value in current_group:
                result[value] = group_id

    return result


# flatten multiple level list or tuple
# taken from http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
def flatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    if ltype not in ltypes:
        ltype = list
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i : i + 1] = l[i]
        i += 1
    return ltype(l)


def order_by_unique_counts(labels, ignoreNegative=True):  # decreasing order
    if ignoreNegative:
        labels_pos = labels[labels >= 0]
        unique, counts = np.unique(labels_pos, return_counts=True)
        order = np.argsort(counts)[::-1]
        mapping = {unique[v]: i for i, v in enumerate(order)}
        labels_neg = labels[labels < 0]
        mapping.update({v: v for v in np.unique(labels_neg)})
    else:
        unique, counts = np.unique(labels, return_counts=True)
        order = np.argsort(counts)[::-1]
        mapping = {unique[v]: i for i, v in enumerate(order)}
    ret = [mapping[v] for v in labels]
    return ret


def split_array(arr):
    """Split an unordered array into two groups but minimize the difference of the group sums. Return the group indices"""
    total_sum = sum(arr)
    target_sum = total_sum // 2
    n = len(arr)

    # Create a 2D DP table
    dp = [[False for _ in range(target_sum + 1)] for _ in range(n + 1)]

    # Initialize the first column
    for i in range(n + 1):
        dp[i][0] = True

    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(1, target_sum + 1):
            if arr[i - 1] <= j:
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - arr[i - 1]]
            else:
                dp[i][j] = dp[i - 1][j]

    # Find the largest sum <= target_sum that can be achieved
    achieved_sum = 0
    for j in range(target_sum, -1, -1):
        if dp[n][j]:
            achieved_sum = j
            break

    # Backtrack to find the elements in the first group
    group1 = []
    i, j = n, achieved_sum
    while i > 0 and j > 0:
        if not dp[i - 1][j]:
            group1.append(i - 1)
            j -= arr[i - 1]
        i -= 1

    # The second group consists of all elements not in the first group
    group2 = [i for i in range(n) if i not in group1]

    return group1, group2


def set_angle_range(angle, range=[-180, 180]):
    v0, v1 = range[0], range[-1]
    delta = v1 - v0
    ret = angle * 1
    if isinstance(angle, np.ndarray):
        pos = np.where(angle > v0)
        neg = np.where(angle <= v0)
        ret[pos] = np.fmod(angle[pos] - v0, delta) + v0
        ret[neg] = v1 - np.fmod(v0 - angle[neg], delta)
    else:  # assume it is just a scalar
        if angle > v0:
            ret = np.fmod(angle - v0, delta) + v0
        else:
            ret = v1 - np.fmod(v0 - angle, delta)
    return ret


def set_to_periodic_range(v, min=-180, max=180):
    if min <= v <= max:
        return v
    from math import fmod

    tmp = fmod(v - min, max - min)
    if tmp >= 0:
        tmp += min
    else:
        tmp += max
    return tmp


def encode_numpy(img, hflip=False, vflip=False):
    if img.dtype != np.dtype("uint8"):
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            tmp = (255 * (img - vmin) / (vmax - vmin)).astype(np.uint8)
        else:
            tmp = np.zeros_like(img, dtype=np.uint8)
    else:
        tmp = img
    if hflip:
        tmp = tmp[:, ::-1]
    if vflip:
        tmp = tmp[::-1, :]
    from PIL import Image

    pil_img = Image.fromarray(tmp)
    return encode_PIL_Image(pil_img)


def encode_PIL_Image(img, hflip=False, vflip=False):
    import io, base64

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64, {encoded}"


def get_logger(logfile="", verbose=0):
    import logging

    if not logfile:
        logfile = os.path.splitext(os.path.basename(sys.argv[0]))[0] + ".log"

    logger = logging.getLogger(logfile)
    logger.setLevel(logging.DEBUG)

    # save to the log file
    fh = logging.FileHandler(logfile, mode="at")
    fh.setLevel(logging.INFO)

    # print to screen
    ch = logging.StreamHandler()
    if verbose <= 0:
        ch.setLevel(logging.ERROR)
    elif verbose == 1:
        ch.setLevel(logging.WARNING)
    elif verbose == 2:
        ch.setLevel(logging.INFO)
    elif verbose > 2:
        ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    if os.path.getsize(logfile) > 0:
        logger.info("%s" % ("#" * 128))
    return logger


def unique_attr_name(data, attr_prefix):
    if attr_prefix not in data:
        return attr_prefix
    attr_i = 2
    attr = f"{attr_prefix}{attr_i}"
    while attr in data:
        attr_i += 1
        attr = f"{attr_prefix}{attr_i}"
    return attr


def first_matched_atrr(data, attrs):
    ret = None
    for attr in attrs:
        if attr in data:
            ret = attr
            break
    return ret


def log_command_line():
    try:
        hist = open(".helicon.txt", "r+")
        hist.seek(0, os.SEEK_END)
    except:
        try:
            hist = open(".helicon.txt", "w")
        except:
            return -1
    from datetime import datetime

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"{current_time}: {' '.join(sys.argv)}\n"
    hist.write(msg)
    hist.close()


def get_context_function_name():
    import inspect

    return inspect.stack()[1].function


class Timer:
    def __init__(self, info="Timer", verbose=1):
        self.info = info
        self.verbose = verbose

    def __enter__(self):
        from timeit import default_timer

        self.start = default_timer()
        if self.verbose:
            print(f"{self.info}: started at {datetime.datetime.now()}")
        return self

    def __exit__(self, *args):
        from timeit import default_timer

        self.end = default_timer()
        self.interval = self.end - self.start
        if self.verbose:
            print(
                f"{self.info}: ended at {datetime.datetime.now()}, duration={self.interval} seconds"
            )


class DummyMemory:
    """Dummy joblib.Memory"""

    def __init__(self, location=None, bytes_limit=-1, verbose=0):
        self.location = location
        self.verbose = verbose

    def cache(self, func=None, **kwargs):
        def decorator(f):
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

            return wrapper

        if func is None:
            return decorator
        else:
            return decorator(func)
