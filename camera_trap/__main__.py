import sys
import logging
import asyncio
# from systemd.journal import JournalHandler
from .camera_trap import CameraTrap
from dotenv import load_dotenv


def main():
    assert sys.version_info >= (3, 6)
    load_dotenv()
    logging.basicConfig(
        handlers=[logging.FileHandler("log.txt"), logging.StreamHandler()],
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y%m%d%H%M%S",
    )
    camera_trap = CameraTrap()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(camera_trap.loop())
    except KeyboardInterrupt:
        camera_trap.__del__()

if __name__ == "__main__":
    main()