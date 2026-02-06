#!/usr/bin/env python
import argparse
import os
from typing import Optional

from simple_video_utils.metadata import video_metadata
from simple_video_utils.frames import read_frames_exact

from pose_format.utils.alphapose import load_alphapose_wholebody_from_json
from pose_format.utils.smplest_x import load_smplestx_pose
from pose_format.utils.sapiens import load_sapiens_wholebody_from_json


def json_to_pose(
        input_path: str,
        output_path: str,
        original_video_path: Optional[str],
        format: str):
    """
    Render pose visualization over a video.

    Parameters
    ----------
    input_path : str
        Path to the input .json file.
    output_path : str
        Path where the output .pose file will be saved.
    original_video_path : str or None, optional
        Path to the original RGB video to obtain metadata.
        If None, it first checks if the .json file already contains the metadata,
        otherwise default values are used.
    """

    kwargs = {}
    if original_video_path is not None:
        # Load video metadata
        print('Obtaining metadata from video ...')
        metadata = video_metadata(original_video_path)
        kwargs["fps"] = metadata.fps
        kwargs["width"] = metadata.width
        kwargs["height"] = metadata.height
    
    # Perform pose estimation
    print('Converting .json to .pose pose-format ...')
    format = format.lower()
    if format == 'smplest-x':
        # SMPLest-X uses spatial dimensions from JSON, not from video
        kwargs.pop("width", None)
        kwargs.pop("height", None)

        pose = load_smplestx_pose(
            input_path=input_path,
            **kwargs
        )

    elif format == 'alphapose':
        pose = load_alphapose_wholebody_from_json(
            input_path=input_path,
            **kwargs
        )
    elif format == 'sapiens':
        pose = load_sapiens_wholebody_from_json(
            input_path=input_path,
            **kwargs  # only includes keys if video metadata was found
        )
    else:
        raise NotImplementedError(f'Pose format {format} not supported')

    # Write
    print('Saving to disk ...')
    with open(output_path, "wb") as f:
        pose.write(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, type=str, help='Path to the input .json file.')
    parser.add_argument('-o', required=True, type=str, help='Path where the output .pose file will be saved.')
    parser.add_argument(
        '--original-video',
        type=str,
        default=None,
        help=(
            "Path to the original RGB video used for metadata extraction. "
            "If None, metadata is taken from the JSON file if available, "
            "otherwise default width/height/FPS values are used."
        )
    )
    parser.add_argument(
        '--format',
        choices=['smplest-x', 'alphapose', 'sapiens'],
        default='sapiens',
        type=str,
        help='Original type of the .json pose estimation'
    )
    args = parser.parse_args()

    if not os.path.exists(args.i):
        raise FileNotFoundError(f"JSON file {args.i} not found")

    print(f"Converting {args.format} -> pose-format from {args.i}")

    json_to_pose(args.i, args.o, args.original_video, args.format)

if __name__ == "__main__":
    main()

# pip install . && json_to_pose -i sapiens.json -o sapiens.pose --format sapiens
# pip install . && json_to_pose -i sapiens.json -o sapiens.pose --original-video video.mp4 --format sapiens

# pip install . && json_to_pose -i alphapose.json -o alphapose.pose --format alphapose
# pip install . && json_to_pose -i alphapose.json -o alphapose.pose --original-video video.mp4 --format alphapose

# pip install . && json_to_pose -i smplest-x.json -o smplest-x.pose --format smplest-x
# pip install . && json_to_pose -i smplest-x.json -o smplest-x.pose --original-video video.mp4 --format smplest-x