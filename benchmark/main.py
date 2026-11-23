"""SatBench command line.

python main.py run            run or continue the whole benchmark
python main.py job <job_id>   run a single job (used internally by "run")
python main.py report         rebuild the report from finished results
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="SatBench: CNN vs ViT benchmark on remote-sensing scenes")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="run or continue the benchmark")
    job = sub.add_parser("job", help="run one job")
    job.add_argument("job_id")
    sub.add_parser("report", help="rebuild tables, figures and the results PDF")
    args = parser.parse_args()

    if args.command == "run":
        from satbench.runner import orchestrate

        orchestrate()
    else:
        from satbench.jobs import run_job

        sys.exit(run_job(args.job_id if args.command == "job" else "report"))


if __name__ == "__main__":
    main()
