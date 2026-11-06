"""Selectable programs share the same robot, perception and exclusive controller."""
import argparse


def main(argv=None):
    parser=argparse.ArgumentParser(description='节目选择；先Ctrl+C退出当前节目，再启动另一个。')
    parser.add_argument('program',choices=('expression-follow','singing'))
    args,remaining=parser.parse_known_args(argv)
    if args.program=='expression-follow':
        from vivis.interaction.follow import main as run
    else:
        from vivis.interaction.singing import main as run
    return run(remaining)


if __name__=='__main__':raise SystemExit(main())
