import argparse
from src.matchmove import MatchMoveApp
from src.selectutil import SelectUtilApp

apps = {
    "main" : MatchMoveApp,
    "util" : SelectUtilApp
}

parser = argparse.ArgumentParser()
parser.add_argument('usage', choices=apps.keys())


def main(args=parser.parse_args()):
    apps[args.usage]().run()


if __name__ == "__main__":
    main()