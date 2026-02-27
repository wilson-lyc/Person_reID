from datetime import datetime


def generate_run_id() -> str:
    return datetime.now().strftime("%m%d_%H%M%S")


if __name__ == "__main__":
    print(generate_run_id())
