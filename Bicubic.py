#!/usr/bin/env python

"""Upscale images and videos using bicubic interpolation, with folder processing support."""

import argparse
import itertools
import os
import subprocess
import sys
import threading
import glob

from argparse import Namespace
from time import sleep
from typing import Optional, List

import cv2
from rich.progress import track


class Spinner:
    def __init__(self, enter_message: str = "", exit_message: str = "", delay: float = 0.1) -> None:
        self.spinner = itertools.cycle(["-", "/", "|", "\\"])
        self.enter_message = enter_message
        self.exit_message = exit_message
        self.delay = delay
        self.busy = False

        self._screen_lock = None
        self._thread = None

    def spinner_task(self) -> None:
        with self._screen_lock:
            while self.busy:
                sys.stdout.write(next(self.spinner))
                sys.stdout.flush()
                sleep(self.delay)
                sys.stdout.write("\b")
                sys.stdout.flush()

    def __enter__(self) -> None:
        sys.stdout.write(self.enter_message)
        self.busy = True
        self._screen_lock = threading.Lock()
        self._thread = threading.Thread(target=self.spinner_task)
        self._thread.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.busy = False
        with self._screen_lock:
            sys.stdout.write(f"{self.exit_message} \n")
            sys.stdout.flush()


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Upscale images using bicubic interpolation")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        help="paths to individual images to upscale",
    )
    input_group.add_argument(
        "--input_folder",
        type=str,
        help="folder containing images to upscale",
    )

    parser.add_argument(
        "scale",
        type=float,
        help="the number of times to upscale images",
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="save upscaled images to this folder",
    )

    parser.add_argument(
        "--image_extensions",
        type=str,
        nargs="+",
        default=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="file extensions to process when using input_folder (default: jpg, jpeg, png, bmp, tiff)",
    )

    for short_name, long_name in zip(["l", "r", "t", "b"], ["left", "right", "top", "bottom"]):
        parser.add_argument(
            f"-c{short_name}",
            f"--crop_{long_name}",
            type=int,
            default=0,
            nargs="?",
            help=f"{long_name}-crop the resulting image by this number of pixels",
        )

    parser.add_argument(
        "-i",
        "--ignore",
        action="store_true",
        help="If present, ignore errors",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If present, overwrite existing files without prompting",
    )

    args = parser.parse_args()
    return args


def get_image_files_from_folder(folder_path: str, extensions: List[str]) -> List[str]:
    """Get all image files with specified extensions from a folder."""
    image_files = []
    for ext in extensions:
        # Handle extensions with or without the dot
        if not ext.startswith('.'):
            ext = f".{ext}"
        pattern = os.path.join(folder_path, f"*{ext}")
        image_files.extend(glob.glob(pattern))
        # Also try with uppercase extension
        pattern = os.path.join(folder_path, f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern))

    return sorted(image_files)


def upscale_image(
        input_image_path: str,
        output_image_path: str,
        *,
        scale: float = 1,
        crop_left: int = 0,
        crop_right: int = 0,
        crop_top: int = 0,
        crop_bottom: int = 0,
        overwrite: bool = False,
) -> bool:
    """Upscale a single image using bicubic interpolation."""
    if os.path.exists(output_image_path) and not overwrite:
        user = input(
            f'\nThe output path "{output_image_path}" already exists. Do you want to rewrite it? (y/n): '
        )
        if user.lower() != "y":
            print("\nSkipping this file.\n")
            return False

    try:
        image = cv2.imread(input_image_path, flags=cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Error: Could not read image {input_image_path}")
            return False

        if scale > 1:
            image = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        if crop_top > 0 or crop_bottom > 0 or crop_left > 0 or crop_right > 0:
            height, width = image.shape[:2]
            bottom_limit = max(0, height - crop_bottom)
            right_limit = max(0, width - crop_right)
            image = image[crop_top:bottom_limit, crop_left:right_limit]

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

        cv2.imwrite(output_image_path, image)
        return True
    except Exception as e:
        print(f"Error processing {input_image_path}: {str(e)}")
        return False


def main() -> None:
    args = parse_args()

    # Collect input files
    if args.input_folder:
        input_files = get_image_files_from_folder(args.input_folder, args.image_extensions)
        print(f"Found {len(input_files)} images in {args.input_folder}")
    else:
        input_files = args.input_files

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Process files
    success_count = 0
    fail_count = 0

    for input_path in track(input_files, description="Upscaling images:"):
        if not os.path.exists(input_path):
            if args.ignore:
                print(f"Warning: File not found: {input_path}")
                fail_count += 1
                continue
            else:
                raise FileNotFoundError(f"The following path does not exist: {input_path}")

        # Determine output path
        filename = os.path.basename(input_path)
        output_path = os.path.join(args.output_folder, filename)

        output_path = output_path.replace("LR","HR")
        # Upscale the image
        result = upscale_image(
            input_path,
            output_path,
            scale=args.scale,
            crop_left=args.crop_left,
            crop_right=args.crop_right,
            crop_top=args.crop_top,
            crop_bottom=args.crop_bottom,
            overwrite=args.overwrite,
        )

        if result:
            success_count += 1
        else:
            fail_count += 1
            if not args.ignore:
                print(f"Error processing {input_path}")
                break

    print(f"\nProcessing complete: {success_count} files succeeded, {fail_count} files failed")


if __name__ == "__main__":
    main()