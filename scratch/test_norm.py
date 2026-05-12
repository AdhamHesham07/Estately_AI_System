import os
import sys

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(BASE_DIR, "1.Agent"))

from data_bridge import DataBridge

def test_norm():
    locs = ["Fifth Settlement", "Maadi", "New Cairo", "Cairo"]
    for l in locs:
        norm = DataBridge.normalize_location(l)
        print(f"'{l}' -> '{norm}'")

if __name__ == "__main__":
    test_norm()
