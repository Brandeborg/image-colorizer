import cv2
import re
import os
sep = os.sep

def to_grayscale(in_img_path: str, out_img_path: str) -> None:
    """Converts image at path `in_img_path` to grayscale an saves at path `out_img_path`. 
    If `out_img_path` does not exist, it is created.

    Args:
        in_img_path (str): Path to an existing image file
        out_img_path (str): Path where the grayscale version of the input is saved
    """
    # extract dir parts of path to output file
    dir_path = sep.join(out_img_path.split(sep)[:-1])
    
    # make dirs of they do not exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # convert and save
    rgb_image = cv2.imread(in_img_path)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(out_img_path, gray_image)

    return gray_image

def to_BW(grayscale, out_img_path: str):
    """Converts MatLike grayscale image to black and white an saves at path `out_img_path`. 
    If `out_img_path` does not exist, it is created.

    Args:
        graysacle: Path to an existing image file
        out_img_path (str): Path where the black and white version of the input is saved
    """
    # extract dir parts of path to output file
    dir_path = sep.join(out_img_path.split(sep)[:-1])
    
    # make dirs of they do not exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # convert and save
    _thresh, bw_image = cv2.threshold(grayscale, 127.5, 255, cv2.THRESH_BINARY)
    cv2.imwrite(out_img_path, bw_image)

    return bw_image

def create_input_data(target_dir: str) -> None:
    """Create input image files (grayscale) from target data (colored)

    Args:
        target_dir
    """
    for root, _dirs, files in os.walk(target_dir):
        if files == []:
            continue

        for file in files:
            in_path = sep.join([root, file])
            out_path = re.sub("target", "input_grayscale", in_path)
            
            grayscale = to_grayscale(in_path, out_path)

            out_path = re.sub("target", "input_bw", in_path)
            print(out_path)
            to_BW(grayscale, out_path)

def main():
    create_input_data(f"dataset{sep}target")
    

if __name__ == "__main__":
    main()