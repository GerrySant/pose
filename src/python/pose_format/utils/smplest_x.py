import re
import json
import numpy as np
from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions
from pose_format.utils.openpose import hand_colors

SMPLX_BODY_POINTS = [
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
]

SMPLX_GENERAL_HAND_POINTS = [
    "Thumb_1", "Thumb_2", "Thumb_3", "Thumb_4",
    "Index_1", "Index_2", "Index_3", "Index_4",
    "Middle_1", "Middle_2", "Middle_3", "Middle_4",
    "Ring_1", "Ring_2", "Ring_3", "Ring_4",
    "Pinky_1", "Pinky_2", "Pinky_3", "Pinky_4",
]

SMPLX_FACE_POINTS = [f"Face_{i}" for i in range(1, 73)]

SMPLX_LEFT_HAND_POINTS = [f"L_{k}" for k in SMPLX_GENERAL_HAND_POINTS]
SMPLX_RIGHT_HAND_POINTS = [f"R_{k}" for k in SMPLX_GENERAL_HAND_POINTS]

EXTENDED_SMPLX_LEFT_HAND_POINTS = (SMPLX_LEFT_HAND_POINTS + ["L_Wrist_Hand"])
EXTENDED_SMPLX_RIGHT_HAND_POINTS = (SMPLX_RIGHT_HAND_POINTS + ["R_Wrist_Hand"])
EXTENDED_SMPLX_GENERAL_HAND_POINTS = (SMPLX_GENERAL_HAND_POINTS + ["Wrist_Hand"])

SMPLX_JOINT_NAMES = (
    tuple(SMPLX_BODY_POINTS)
    + tuple(SMPLX_LEFT_HAND_POINTS)
    + tuple(SMPLX_RIGHT_HAND_POINTS)
    + tuple(SMPLX_FACE_POINTS)
)

assert len(SMPLX_JOINT_NAMES) == 137

EXTENDED_SMPLX_JOINT_NAMES = (
    tuple(SMPLX_BODY_POINTS)
    + tuple(EXTENDED_SMPLX_LEFT_HAND_POINTS)
    + tuple(EXTENDED_SMPLX_RIGHT_HAND_POINTS)
    + tuple(SMPLX_FACE_POINTS)
)

assert len(EXTENDED_SMPLX_JOINT_NAMES) == 139

orig_name_to_idx = {
    name: i for i, name in enumerate(SMPLX_JOINT_NAMES)
}

final_name_to_idx = {
    name: i for i, name in enumerate(EXTENDED_SMPLX_JOINT_NAMES)
}

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

GENERAL_HAND_LIMBS_NAMES = [
    ("Wrist_Hand", "Thumb_1"), ("Wrist_Hand", "Index_1"), ("Wrist_Hand", "Middle_1"),
    ("Wrist_Hand", "Ring_1"), ("Wrist_Hand", "Pinky_1"),
    ("Thumb_1", "Thumb_2"), ("Thumb_2", "Thumb_3"), ("Thumb_3", "Thumb_4"),
    ("Index_1", "Index_2"), ("Index_2", "Index_3"), ("Index_3", "Index_4"),
    ("Middle_1", "Middle_2"), ("Middle_2", "Middle_3"), ("Middle_3", "Middle_4"),
    ("Ring_1", "Ring_2"), ("Ring_2", "Ring_3"), ("Ring_3", "Ring_4"),
    ("Pinky_1", "Pinky_2"), ("Pinky_2", "Pinky_3"), ("Pinky_3", "Pinky_4"),
    ("Index_1", "Middle_1"), ("Middle_1", "Ring_1"), ("Ring_1", "Pinky_1")
]

def build_smplx_with_hand_wrists(joints, conf):
    """
    Rebuild SMPL-X joints following BODY → LEFT_HAND → RIGHT_HAND → FACE order,
    duplicating wrists as hand roots.
    """
    K_final = len(EXTENDED_SMPLX_JOINT_NAMES)

    xy_ext = np.zeros((K_final, 2), dtype=joints.dtype)
    conf_ext = np.zeros((K_final,), dtype=conf.dtype)

    for i, name in enumerate(EXTENDED_SMPLX_JOINT_NAMES):
        if name == "L_Wrist_Hand":
            src = orig_name_to_idx["L_Wrist"]
        elif name == "R_Wrist_Hand":
            src = orig_name_to_idx["R_Wrist"]
        else:
            src = orig_name_to_idx[name]

        xy_ext[i] = joints[src]
        conf_ext[i] = conf[src]

    return xy_ext, conf_ext


def get_smplx_components():

    def map_limbs(points, limbs):
        index_map = {name: idx for idx, name in enumerate(points)}
        return [(index_map[a], index_map[b]) for a, b in limbs]

    return [
        PoseHeaderComponent(
            name="BODY",
            points=SMPLX_BODY_POINTS,
            limbs=map_limbs(SMPLX_BODY_POINTS, BODY_LIMBS_NAMES),
            colors=[(0, 255, 0)],
            point_format="XYC",
        ),
        PoseHeaderComponent(
            name="LEFT_HAND",
            points=EXTENDED_SMPLX_GENERAL_HAND_POINTS,
            limbs=map_limbs(EXTENDED_SMPLX_GENERAL_HAND_POINTS, GENERAL_HAND_LIMBS_NAMES),
            colors=[(0, 255, 255)],
            point_format="XYC",
        ),
        PoseHeaderComponent(
            name="RIGHT_HAND",
            points=EXTENDED_SMPLX_GENERAL_HAND_POINTS,
            limbs=map_limbs(EXTENDED_SMPLX_GENERAL_HAND_POINTS, GENERAL_HAND_LIMBS_NAMES),
            colors=[(255, 128, 0)],
            point_format="XYC",
        ),
        PoseHeaderComponent(
            name="FACE",
            points=SMPLX_FACE_POINTS,
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
    num_joints = len(EXTENDED_SMPLX_JOINT_NAMES)

    fps = data.get("fps", fps)
    width = data.get("width", width)
    height = data.get("height", height)

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

        # first frames may be empty → initialize safely
        if frame['frame_id'] == 1 and frame['persons'] == []:
            j = frame['frame_id']

            # find first valid frame
            while j < len(frames) and frames[j]["persons"] == []:
                j += 1
            if j < len(frames):
                frame["persons"] = frames[j]["persons"]
            
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

            # persons_data.append(joints_norm)
            # persons_conf.append(np.ones((num_joints,), dtype=np.float32))

            # confidence for original SMPL-X joints (137)
            conf = np.ones((joints_norm.shape[0],), dtype=np.float32)

            # rebuild joints following BODY → LEFT_HAND → RIGHT_HAND → FACE order
            # and duplicate wrists inside hand components
            joints_ext, conf_ext = build_smplx_with_hand_wrists(joints_norm, conf)

            persons_data.append(joints_ext)
            persons_conf.append(conf_ext)

        if len(persons_data) == 0:
            if last_pose is None:
                # first frames may be empty → initialize safely
                pose = np.zeros((K_final, 2), dtype=np.float32)
                conf = np.zeros((K_final,), dtype=np.float32)
            else:
                # copy previous frame
                pose = last_pose.copy()
                conf = last_conf.copy()
        else:
            # enforce single-person assumption
            pose = persons_data[0]
            conf = persons_conf[0]
            last_pose = pose
            last_conf = conf

        all_data.append([pose])
        all_conf.append([conf])

    data_np = np.asarray(all_data, dtype=np.float32)
    conf_np = np.asarray(all_conf, dtype=np.float32)

    header = PoseHeader(
        version=0.2,
        dimensions=PoseHeaderDimensions(width=width, height=height, depth=0),
        components=get_smplx_components(),
    )

    body = NumPyPoseBody(
        fps=fps,
        data=data_np,
        confidence=conf_np,
    )

    return Pose(header, body)