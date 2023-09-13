from PIL import Image
import re
import os
sep = os.sep

def to_grayscale(in_img_path: str, out_img_path: str):
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
    img = Image.open(in_img_path).convert('L')
    img.save(out_img_path)

def main():
    input_path = f".{sep}dataset{sep}input{sep}Chapter 1{sep}page 1.png"
    output_path = re.sub("input", "target", input_path)
    
    to_grayscale(input_path, output_path)

if __name__ == "__main__":
    main()