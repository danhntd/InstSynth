import shutil
import os
import argparse

def copy_directories(src_base, dst_base, directories):
    """
    Copies specified directories from a source base path to a destination base path.
    
    Args:
        src_base (str): Base path for source directories.
        dst_base (str): Base path for destination directories.
        directories (list): List of directory names to copy.
    """
    for directory in directories:
        src_dir = os.path.join(src_base, directory)
        dst_dir = os.path.join(dst_base, directory)
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        print(f"Copied {src_dir} to {dst_dir}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy Cityscapes directories.")
    parser.add_argument(
        "--src_path", 
        type=str, 
        required=True, 
        help="Base source folder for all copied directories."
    )
    parser.add_argument(
        "--des_path", 
        type=str, 
        required=True, 
        help="Base destination folder for all copied directories."
    )
    args = parser.parse_args()
    des_path=args.des_path
    src_path=args.src_path
    

    # Define directories for training, validation, and testing
    train_dirs = [
        "aachen", "bochum", "bremen", "cologne", "darmstadt", "dusseldorf", 
        "hamburg", "hanover", "jena", "krefeld", "monchengladbach", "strasbourg", 
        "stuttgart", "tubingen", "ulm", "weimar", "zurich", "erfurt"
    ]
    test_val_dirs = ["test", "val"]

    # Define source base paths
    src_base_gtFine_train = os.path.join(src_path, "gtFine/train")
    src_base_leftImg8bit_train = os.path.join(src_path, "leftImg8bit/train")
    src_base_gtFine_test_val = os.path.join(src_path, "gtFine")
    src_base_leftImg8bit_test_val = os.path.join(src_path, "leftImg8bit")

    # Define destination base paths using folder_path
    dst_base_gtFine_train = os.path.join(des_path, "gtFine/train")
    dst_base_leftImg8bit_train = os.path.join(des_path, "leftImg8bit/train")
    dst_base_gtFine_test_val = os.path.join(des_path, "gtFine")
    dst_base_leftImg8bit_test_val = os.path.join(des_path, "leftImg8bit")

    # Copy training directories
    copy_directories(src_base_gtFine_train, dst_base_gtFine_train, train_dirs)
    copy_directories(src_base_leftImg8bit_train, dst_base_leftImg8bit_train, train_dirs)

    # Copy testing and validation directories
    copy_directories(src_base_gtFine_test_val, dst_base_gtFine_test_val, test_val_dirs)
    copy_directories(src_base_leftImg8bit_test_val, dst_base_leftImg8bit_test_val, test_val_dirs)