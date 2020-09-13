import argparse

from src.transcribe_utils import delete_all_jobs


def parser() -> argparse.Namespace:
    parse = argparse.ArgumentParser()
    parse.add_argument("--base-job-name", type=str, default="dpv")
    return parse.parse_args()


if __name__ == "__main__":
    args = parser()
    delete_all_jobs(prefix=args.base_job_name)