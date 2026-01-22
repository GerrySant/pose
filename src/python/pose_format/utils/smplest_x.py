import re
import json
import numpy as np
from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions
from pose_format.utils.openpose import hand_colors

def smplx_components():

    def map_limbs(points, limbs):
        index_map = {name: idx for idx, name in enumerate(points)}
        return [(index_map[a], index_map[b]) for a, b in limbs]

    SMPLX_JOINT_NAMES = (
        # --- Body (25) ---
        "Pelvis",
        "L_Hip", "R_Hip",
        "L_Knee", "R_Knee",
        "L_Ankle", "R_Ankle",
        "Neck",
        "L_Shoulder", "R_Shoulder",
        "L_Elbow", "R_Elbow",
        "L_Wrist", "R_Wrist",
        "L_Big_toe", "L_Small_toe", "L_Heel",
        "R_Big_toe", "R_Small_toe", "R_Heel",
        "L_Ear", "R_Ear",
        "L_Eye", "R_Eye",
        "Nose",

        # --- Left hand (20) ---
        "L_Thumb_1", "L_Thumb_2", "L_Thumb_3", "L_Thumb_4",
        "L_Index_1", "L_Index_2", "L_Index_3", "L_Index_4",
        "L_Middle_1", "L_Middle_2", "L_Middle_3", "L_Middle_4",
        "L_Ring_1", "L_Ring_2", "L_Ring_3", "L_Ring_4",
        "L_Pinky_1", "L_Pinky_2", "L_Pinky_3", "L_Pinky_4",

        # --- Right hand (20) ---
        "R_Thumb_1", "R_Thumb_2", "R_Thumb_3", "R_Thumb_4",
        "R_Index_1", "R_Index_2", "R_Index_3", "R_Index_4",
        "R_Middle_1", "R_Middle_2", "R_Middle_3", "R_Middle_4",
        "R_Ring_1", "R_Ring_2", "R_Ring_3", "R_Ring_4",
        "R_Pinky_1", "R_Pinky_2", "R_Pinky_3", "R_Pinky_4",

        # --- Face (72) ---
        *[f"Face_{i}" for i in range(1, 73)],
    )
    assert len(SMPLX_JOINT_NAMES) == 137

    BODY_LIMBS_NAMES = [
        ("Pelvis", "L_Hip"),
        ("Pelvis", "R_Hip"),
        ("L_Hip", "L_Knee"),
        ("R_Hip", "R_Knee"),
        ("L_Knee", "L_Ankle"),
        ("R_Knee", "R_Ankle"),

        ("Pelvis", "Neck"),
        ("Neck", "L_Shoulder"),
        ("Neck", "R_Shoulder"),
        ("L_Shoulder", "L_Elbow"),
        ("R_Shoulder", "R_Elbow"),
        ("L_Elbow", "L_Wrist"),
        ("R_Elbow", "R_Wrist"),

        ("L_Ankle", "L_Big_toe"),
        ("L_Ankle", "L_Small_toe"),
        ("L_Ankle", "L_Heel"),
        ("R_Ankle", "R_Big_toe"),
        ("R_Ankle", "R_Small_toe"),
        ("R_Ankle", "R_Heel"),

        ("Neck", "Nose"),
        ("Nose", "L_Eye"),
        ("Nose", "R_Eye"),
        ("L_Eye", "L_Ear"),
        ("R_Eye", "R_Ear"),
    ]
    LEFT_HAND_LIMBS_NAMES = [
        ("L_Thumb_1", "L_Thumb_2"), ("L_Thumb_2", "L_Thumb_3"), ("L_Thumb_3", "L_Thumb_4"),
        ("L_Index_1", "L_Index_2"), ("L_Index_2", "L_Index_3"), ("L_Index_3", "L_Index_4"),
        ("L_Middle_1", "L_Middle_2"), ("L_Middle_2", "L_Middle_3"), ("L_Middle_3", "L_Middle_4"),
        ("L_Ring_1", "L_Ring_2"), ("L_Ring_2", "L_Ring_3"), ("L_Ring_3", "L_Ring_4"),
        ("L_Pinky_1", "L_Pinky_2"), ("L_Pinky_2", "L_Pinky_3"), ("L_Pinky_3", "L_Pinky_4"),
    ]
    RIGHT_HAND_LIMBS_NAMES = [
        ("R_Thumb_1", "R_Thumb_2"), ("R_Thumb_2", "R_Thumb_3"), ("R_Thumb_3", "R_Thumb_4"),
        ("R_Index_1", "R_Index_2"), ("R_Index_2", "R_Index_3"), ("R_Index_3", "R_Index_4"),
        ("R_Middle_1", "R_Middle_2"), ("R_Middle_2", "R_Middle_3"), ("R_Middle_3", "R_Middle_4"),
        ("R_Ring_1", "R_Ring_2"), ("R_Ring_2", "R_Ring_3"), ("R_Ring_3", "R_Ring_4"),
        ("R_Pinky_1", "R_Pinky_2"), ("R_Pinky_2", "R_Pinky_3"), ("R_Pinky_3", "R_Pinky_4"),
    ]


    BODY_POINTS = SMPLX_JOINT_NAMES[0:25]
    LEFT_HAND_POINTS = SMPLX_JOINT_NAMES[25:45]
    RIGHT_HAND_POINTS = SMPLX_JOINT_NAMES[45:65]
    FACE_POINTS = SMPLX_JOINT_NAMES[65:137]

    return [
        PoseHeaderComponent(
            name="BODY",
            points=BODY_POINTS,
            limbs=map_limbs(BODY_POINTS, BODY_LIMBS_NAMES),
            colors=[(0, 255, 0)],
            point_format="XYC",
        ),
        PoseHeaderComponent(
            name="LEFT_HAND",
            points=LEFT_HAND_POINTS,
            limbs=map_limbs(LEFT_HAND_POINTS, LEFT_HAND_LIMBS_NAMES),
            colors=[(0, 255, 255)],
            point_format="XYC",
        ),
        PoseHeaderComponent(
            name="RIGHT_HAND",
            points=RIGHT_HAND_POINTS,
            limbs=map_limbs(RIGHT_HAND_POINTS, RIGHT_HAND_LIMBS_NAMES),
            colors=[(255, 128, 0)],
            point_format="XYC",
        ),
        PoseHeaderComponent(
            name="FACE",
            points=FACE_POINTS,
            limbs=[],  # face mesh too dense → usually omitted
            colors=[(255, 255, 255)],
            point_format="XYC",
        ),
    ]

def load_smplestx_pose(
    input_path: str,
    fps: float = 24,
    width: int | None = None,
    height: int | None = None,
) -> Pose:
    """
    Load SMPLest-X JSON and normalize pose so the BODY occupies full canvas.
    """

    with open(input_path, "r") as f:
        data = json.load(f)

    frames = data["frames"]
    num_joints = data["num_joints"]

    # Canvas size
    json_w, json_h = frames[0]["image_size"]
    if width is None:
        width = json_w
    if height is None:
        height = json_h

    BODY_IDX = list(range(25))
    input_width = 192
    input_height = 256

    all_data = []
    all_conf = []

    for frame in frames:
        persons_data = []
        persons_conf = []

        for person in frame["persons"]:
            # --- joints predicted inside crop ---
            joints_crop = np.asarray(person["joints_2d"], dtype=np.float32)

            # --- bbox: x, y, w, h ---
            x0, y0, w, h = map(float, person["bbox"])

            # --- crop → image coordinates ---
            joints_img = joints_crop.copy()
            joints_img[:, 0] = joints_crop[:, 0] * (w / input_width) + x0
            joints_img[:, 1] = joints_crop[:, 1] * (h / input_height) + y0

            # --- BODY bounding box ---
            body = joints_img[BODY_IDX]
            min_xy = body.min(axis=0)
            max_xy = body.max(axis=0)

            bw = max(max_xy[0] - min_xy[0], 1e-6)
            bh = max(max_xy[1] - min_xy[1], 1e-6)

            # --- invisible margin (10%) ---
            margin_ratio = 0.10

            usable_width  = width  * (1.0 - 2 * margin_ratio)
            usable_height = height * (1.0 - 2 * margin_ratio)

            offset_x = width  * margin_ratio
            offset_y = height * margin_ratio

            # --- normalize BODY to usable area ---
            joints_norm = joints_img.copy()
            joints_norm[:, 0] = (joints_img[:, 0] - min_xy[0]) * (usable_width / bw) + offset_x
            joints_norm[:, 1] = (joints_img[:, 1] - min_xy[1]) * (usable_height / bh) + offset_y

            persons_data.append(joints_norm)
            persons_conf.append(np.ones((num_joints,), dtype=np.float32))

        all_data.append(persons_data)
        all_conf.append(persons_conf)

    data_np = np.asarray(all_data, dtype=np.float32)
    conf_np = np.asarray(all_conf, dtype=np.float32)

    header = PoseHeader(
        version=0.2,
        dimensions=PoseHeaderDimensions(width=width, height=height, depth=0),
        components=smplx_components(),
    )

    body = NumPyPoseBody(
        fps=fps,
        data=data_np,
        confidence=conf_np,
    )

    return Pose(header, body)