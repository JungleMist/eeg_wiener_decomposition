import argparse
from pathlib import Path
import ECMAD as p
from eeg_bg.config.settings import load_config

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--cache', type=Path, required=True)
    a.add_argument('--workers', type=int, default=4)
    args = a.parse_args()
    cfg = load_config('configs/ECMAD.yaml')
    p.prepare_features(args.cache, p.read_records(args.cache), cfg, list(p.condition_specs(cfg)), args.workers)

if __name__ == '__main__':
    main()
