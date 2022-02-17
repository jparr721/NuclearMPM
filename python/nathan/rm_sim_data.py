from pathlib import Path

def remove_simulation_files(text_folder="../../build/tmp",
                            other_ext=["*.pkl","*.csv"]):
    text_folder = Path(text_folder)
    print(f"Deleting folder {text_folder}")
    for path in text_folder.iterdir():
        path.unlink()
    print(f"Deleted {text_folder}")

    for file_ext in other_ext:
        file_paths = list(Path(".").glob(file_ext))
        for path in file_paths:
            path.unlink()
            print(f"Deleted {path}")

if __name__ == "__main__":
    remove_simulation_files()