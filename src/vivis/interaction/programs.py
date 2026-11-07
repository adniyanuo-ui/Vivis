"""Selectable programs share the same robot, perception and exclusive controller."""
import argparse


def main(argv=None):
    parser=argparse.ArgumentParser(description='节目选择；先Ctrl+C退出当前节目，再启动另一个。')
    parser.add_argument('program',choices=('expression-follow','singing','baigufuren','wz2','dialogue','standby','interaction'))
    args,remaining=parser.parse_known_args(argv)
    if args.program=='expression-follow':
        from vivis.interaction.follow import main as run
    elif args.program=='dialogue':
        from vivis.interaction.dialogue import main as run
    elif args.program=='standby':
        from vivis.interaction.standby import main as run
    elif args.program=='interaction':
        from vivis.interaction.standby import main as idle
        from vivis.paths import ROOT
        def run(options):return idle(options,default_config=ROOT/'configs/interaction/interaction.json')
    elif args.program in ('baigufuren','wz2'):
        from vivis.interaction.prerecorded import main as prepared
        def run(options):return prepared(options,default_work='wz2' if args.program=='wz2' else '0001')
    else:
        from vivis.interaction.singing import main as run
    return run(remaining)


if __name__=='__main__':raise SystemExit(main())
