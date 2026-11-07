"""Prepared expression/audio programs using the accepted developer runtime.

No A2F inference or social expression overlay is started here. Camera targets
remain dedicated to eye/neck tracking; facial targets come from verified assets.
"""
import sys
from vivis.paths import ROOT
from vivis.motion.mapping import read_json
from vivis.runtime.developer import run_example


def main(argv=None,default_work='0001'):
    options=list(sys.argv[1:] if argv is None else argv)
    if not any(value=='--robot' or value.startswith('--robot=') for value in options):
        options+=['--robot',str(ROOT/read_json(ROOT/'configs/interaction/p1.json')['robot'])]
    return run_example('eyes-neck',options,default_work=default_work,default_repeat=1)


if __name__=='__main__':raise SystemExit(main())
