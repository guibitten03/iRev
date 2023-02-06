from argparse import ArgumentParser

import irev.commands

CLI = ArgumentParser()

SUBPARSERS = CLI.add_subparsers(dest="subcommand")

def subcommand(args=[], parent=SUBPARSERS, **kwargs):

    def decorator(func):
        parser = parent.add_parser(kwargs.get("command_name", func.__name__), description=func.__doc__)

        for arg in args:
            parser.add_argument(*arg[0], **arg[1])

        parser.set_defaults(func=func)

    return decorator

def argument(*name_or_flags, **kwargs):
    return ([ *name_or_flags ], kwargs)

@subcommand(
    args=[
        argument("--algorithms", type=str, nargs='+', help="Algorithms to be executed.", required=True),
        argument("--datasets", type=str, nargs='+', help="Path to datasets.", required=True),
        argument("--metrics", type=str, nargs='+', help="Metrics to be executed.", required=True),
        argument("--extra-modules", type=str, nargs='+', help="Any extra modules that need to be imported.", required=False, default=[]),
        argument("--config-file", type=str, help="Path to config file to read parameters from.", required=True),
    ],
    command_name="execute-experiment"
)
def execute_experiment(args):
    irev.commands.execute_experiment(args)

if __name__ == "__main__":
    args = CLI.parse_args()

    args.func(args) if args.subcommand is not None else CLI.print_help()
