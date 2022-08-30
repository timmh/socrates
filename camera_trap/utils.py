import hashlib


def get_uptime():
    with open("/proc/uptime", "r") as f:
        uptime_seconds = float(f.readline().split()[0])

    return uptime_seconds

def checksum(filename, hash_factory=hashlib.md5, chunk_num_blocks=128):
    h = hash_factory()
    with open(filename, "rb") as f: 
        for chunk in iter(lambda: f.read(chunk_num_blocks * h.block_size), b""): 
            h.update(chunk)
    return h

def get_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.readline()) / 1000
    except:
        return None