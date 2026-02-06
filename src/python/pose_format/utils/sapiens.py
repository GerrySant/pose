import json
import numpy as np

from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions

GOLIATH_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear",
    "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle", "left_big_toe",
    "left_small_toe", "left_heel", "right_big_toe", "right_small_toe",
    "right_heel", "right_thumb4", "right_thumb3", "right_thumb2",
    "right_thumb_third_joint", "right_forefinger4", "right_forefinger3", "right_forefinger2",
    "right_forefinger_third_joint", "right_middle_finger4", "right_middle_finger3", "right_middle_finger2",
    "right_middle_finger_third_joint", "right_ring_finger4", "right_ring_finger3", "right_ring_finger2",
    "right_ring_finger_third_joint", "right_pinky_finger4", "right_pinky_finger3", "right_pinky_finger2",
    "right_pinky_finger_third_joint", "right_wrist", "left_thumb4", "left_thumb3",
    "left_thumb2", "left_thumb_third_joint", "left_forefinger4", "left_forefinger3",
    "left_forefinger2", "left_forefinger_third_joint", "left_middle_finger4", "left_middle_finger3",
    "left_middle_finger2", "left_middle_finger_third_joint", "left_ring_finger4", "left_ring_finger3",
    "left_ring_finger2", "left_ring_finger_third_joint", "left_pinky_finger4", "left_pinky_finger3",
    "left_pinky_finger2", "left_pinky_finger_third_joint", "left_wrist", "left_olecranon",
    "right_olecranon", "left_cubital_fossa", "right_cubital_fossa", "left_acromion",
    "right_acromion", "neck", "center_of_glabella", "center_of_nose_root",
    "tip_of_nose_bridge", "midpoint_1_of_nose_bridge", "midpoint_2_of_nose_bridge", "midpoint_3_of_nose_bridge",
    "center_of_labiomental_groove", "tip_of_chin", "upper_startpoint_of_r_eyebrow", "lower_startpoint_of_r_eyebrow",
    "end_of_r_eyebrow", "upper_midpoint_1_of_r_eyebrow", "lower_midpoint_1_of_r_eyebrow", "upper_midpoint_2_of_r_eyebrow",
    "upper_midpoint_3_of_r_eyebrow", "lower_midpoint_2_of_r_eyebrow", "lower_midpoint_3_of_r_eyebrow", "upper_startpoint_of_l_eyebrow",
    "lower_startpoint_of_l_eyebrow", "end_of_l_eyebrow", "upper_midpoint_1_of_l_eyebrow", "lower_midpoint_1_of_l_eyebrow",
    "upper_midpoint_2_of_l_eyebrow", "upper_midpoint_3_of_l_eyebrow", "lower_midpoint_2_of_l_eyebrow", "lower_midpoint_3_of_l_eyebrow",
    "l_inner_end_of_upper_lash_line", "l_outer_end_of_upper_lash_line", "l_centerpoint_of_upper_lash_line", "l_midpoint_2_of_upper_lash_line",
    "l_midpoint_1_of_upper_lash_line", "l_midpoint_6_of_upper_lash_line", "l_midpoint_5_of_upper_lash_line", "l_midpoint_4_of_upper_lash_line",
    "l_midpoint_3_of_upper_lash_line", "l_outer_end_of_upper_eyelid_line", "l_midpoint_6_of_upper_eyelid_line", "l_midpoint_2_of_upper_eyelid_line",
    "l_midpoint_5_of_upper_eyelid_line", "l_centerpoint_of_upper_eyelid_line", "l_midpoint_4_of_upper_eyelid_line", "l_midpoint_1_of_upper_eyelid_line",
    "l_midpoint_3_of_upper_eyelid_line", "l_midpoint_6_of_upper_crease_line", "l_midpoint_2_of_upper_crease_line", "l_midpoint_5_of_upper_crease_line",
    "l_centerpoint_of_upper_crease_line", "l_midpoint_4_of_upper_crease_line", "l_midpoint_1_of_upper_crease_line", "l_midpoint_3_of_upper_crease_line",
    "r_inner_end_of_upper_lash_line", "r_outer_end_of_upper_lash_line", "r_centerpoint_of_upper_lash_line", "r_midpoint_1_of_upper_lash_line",
    "r_midpoint_2_of_upper_lash_line", "r_midpoint_3_of_upper_lash_line", "r_midpoint_4_of_upper_lash_line", "r_midpoint_5_of_upper_lash_line",
    "r_midpoint_6_of_upper_lash_line", "r_outer_end_of_upper_eyelid_line", "r_midpoint_3_of_upper_eyelid_line", "r_midpoint_1_of_upper_eyelid_line",
    "r_midpoint_4_of_upper_eyelid_line", "r_centerpoint_of_upper_eyelid_line", "r_midpoint_5_of_upper_eyelid_line", "r_midpoint_2_of_upper_eyelid_line",
    "r_midpoint_6_of_upper_eyelid_line", "r_midpoint_3_of_upper_crease_line", "r_midpoint_1_of_upper_crease_line", "r_midpoint_4_of_upper_crease_line",
    "r_centerpoint_of_upper_crease_line", "r_midpoint_5_of_upper_crease_line", "r_midpoint_2_of_upper_crease_line", "r_midpoint_6_of_upper_crease_line",
    "l_inner_end_of_lower_lash_line", "l_outer_end_of_lower_lash_line", "l_centerpoint_of_lower_lash_line", "l_midpoint_2_of_lower_lash_line",
    "l_midpoint_1_of_lower_lash_line", "l_midpoint_6_of_lower_lash_line", "l_midpoint_5_of_lower_lash_line", "l_midpoint_4_of_lower_lash_line",
    "l_midpoint_3_of_lower_lash_line", "l_outer_end_of_lower_eyelid_line", "l_midpoint_6_of_lower_eyelid_line", "l_midpoint_2_of_lower_eyelid_line",
    "l_midpoint_5_of_lower_eyelid_line", "l_centerpoint_of_lower_eyelid_line", "l_midpoint_4_of_lower_eyelid_line", "l_midpoint_1_of_lower_eyelid_line",
    "l_midpoint_3_of_lower_eyelid_line", "r_inner_end_of_lower_lash_line", "r_outer_end_of_lower_lash_line", "r_centerpoint_of_lower_lash_line",
    "r_midpoint_1_of_lower_lash_line", "r_midpoint_2_of_lower_lash_line", "r_midpoint_3_of_lower_lash_line", "r_midpoint_4_of_lower_lash_line",
    "r_midpoint_5_of_lower_lash_line", "r_midpoint_6_of_lower_lash_line", "r_outer_end_of_lower_eyelid_line", "r_midpoint_3_of_lower_eyelid_line",
    "r_midpoint_1_of_lower_eyelid_line", "r_midpoint_4_of_lower_eyelid_line", "r_centerpoint_of_lower_eyelid_line", "r_midpoint_5_of_lower_eyelid_line",
    "r_midpoint_2_of_lower_eyelid_line", "r_midpoint_6_of_lower_eyelid_line", "tip_of_nose", "bottom_center_of_nose",
    "r_outer_corner_of_nose", "l_outer_corner_of_nose", "inner_corner_of_r_nostril", "outer_corner_of_r_nostril",
    "upper_corner_of_r_nostril", "inner_corner_of_l_nostril", "outer_corner_of_l_nostril", "upper_corner_of_l_nostril",
    "r_outer_corner_of_mouth", "l_outer_corner_of_mouth", "center_of_cupid_bow", "center_of_lower_outer_lip",
    "midpoint_1_of_upper_outer_lip", "midpoint_2_of_upper_outer_lip", "midpoint_1_of_lower_outer_lip", "midpoint_2_of_lower_outer_lip",
    "midpoint_3_of_upper_outer_lip", "midpoint_4_of_upper_outer_lip", "midpoint_5_of_upper_outer_lip", "midpoint_6_of_upper_outer_lip",
    "midpoint_3_of_lower_outer_lip", "midpoint_4_of_lower_outer_lip", "midpoint_5_of_lower_outer_lip", "midpoint_6_of_lower_outer_lip",
    "r_inner_corner_of_mouth", "l_inner_corner_of_mouth", "center_of_upper_inner_lip", "center_of_lower_inner_lip",
    "midpoint_1_of_upper_inner_lip", "midpoint_2_of_upper_inner_lip", "midpoint_1_of_lower_inner_lip", "midpoint_2_of_lower_inner_lip",
    "midpoint_3_of_upper_inner_lip", "midpoint_4_of_upper_inner_lip", "midpoint_5_of_upper_inner_lip", "midpoint_6_of_upper_inner_lip",
    "midpoint_3_of_lower_inner_lip", "midpoint_4_of_lower_inner_lip", "midpoint_5_of_lower_inner_lip", "midpoint_6_of_lower_inner_lip",
    "l_top_end_of_inferior_crus", "l_top_end_of_superior_crus", "l_start_of_antihelix", "l_end_of_antihelix",
    "l_midpoint_1_of_antihelix", "l_midpoint_1_of_inferior_crus", "l_midpoint_2_of_antihelix", "l_midpoint_3_of_antihelix",
    "l_point_1_of_inner_helix", "l_point_2_of_inner_helix", "l_point_3_of_inner_helix", "l_point_4_of_inner_helix",
    "l_point_5_of_inner_helix", "l_point_6_of_inner_helix", "l_point_7_of_inner_helix", "l_highest_point_of_antitragus",
    "l_bottom_point_of_tragus", "l_protruding_point_of_tragus", "l_top_point_of_tragus", "l_start_point_of_crus_of_helix",
    "l_deepest_point_of_concha", "l_tip_of_ear_lobe", "l_midpoint_between_22_15", "l_bottom_connecting_point_of_ear_lobe",
    "l_top_connecting_point_of_helix", "l_point_8_of_inner_helix", "r_top_end_of_inferior_crus", "r_top_end_of_superior_crus",
    "r_start_of_antihelix", "r_end_of_antihelix", "r_midpoint_1_of_antihelix", "r_midpoint_1_of_inferior_crus",
    "r_midpoint_2_of_antihelix", "r_midpoint_3_of_antihelix", "r_point_1_of_inner_helix", "r_point_8_of_inner_helix",
    "r_point_3_of_inner_helix", "r_point_4_of_inner_helix", "r_point_5_of_inner_helix", "r_point_6_of_inner_helix",
    "r_point_7_of_inner_helix", "r_highest_point_of_antitragus", "r_bottom_point_of_tragus", "r_protruding_point_of_tragus",
    "r_top_point_of_tragus", "r_start_point_of_crus_of_helix", "r_deepest_point_of_concha", "r_tip_of_ear_lobe",
    "r_midpoint_between_22_15", "r_bottom_connecting_point_of_ear_lobe", "r_top_connecting_point_of_helix", "r_point_2_of_inner_helix",
    "l_center_of_iris", "l_border_of_iris_3", "l_border_of_iris_midpoint_1", "l_border_of_iris_12",
    "l_border_of_iris_midpoint_4", "l_border_of_iris_9", "l_border_of_iris_midpoint_3", "l_border_of_iris_6",
    "l_border_of_iris_midpoint_2", "r_center_of_iris", "r_border_of_iris_3", "r_border_of_iris_midpoint_1",
    "r_border_of_iris_12", "r_border_of_iris_midpoint_4", "r_border_of_iris_9", "r_border_of_iris_midpoint_3",
    "r_border_of_iris_6", "r_border_of_iris_midpoint_2", "l_center_of_pupil", "l_border_of_pupil_3",
    "l_border_of_pupil_midpoint_1", "l_border_of_pupil_12", "l_border_of_pupil_midpoint_4", "l_border_of_pupil_9",
    "l_border_of_pupil_midpoint_3", "l_border_of_pupil_6", "l_border_of_pupil_midpoint_2", "r_center_of_pupil",
    "r_border_of_pupil_3", "r_border_of_pupil_midpoint_1", "r_border_of_pupil_12", "r_border_of_pupil_midpoint_4",
    "r_border_of_pupil_9", "r_border_of_pupil_midpoint_3", "r_border_of_pupil_6", "r_border_of_pupil_midpoint_2",
]

EXTENDED_GOLIATH_KEYPOINTS = (GOLIATH_KEYPOINTS + ["left_wrist_body", "right_wrist_body"])

BODY_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear",
    "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle", "left_big_toe",
    "left_small_toe", "left_heel", "right_big_toe", "right_small_toe",
    "right_heel", "left_olecranon", "right_olecranon", "left_cubital_fossa",
    "right_cubital_fossa", "left_acromion", "right_acromion", "neck",
    "left_wrist_body", "right_wrist_body",
]

LEFT_HAND_KEYPOINTS = [
    "left_thumb4", "left_thumb3", "left_thumb2", "left_thumb_third_joint",
    "left_forefinger4", "left_forefinger3", "left_forefinger2", "left_forefinger_third_joint",
    "left_middle_finger4", "left_middle_finger3", "left_middle_finger2", "left_middle_finger_third_joint",
    "left_ring_finger4", "left_ring_finger3", "left_ring_finger2", "left_ring_finger_third_joint",
    "left_pinky_finger4", "left_pinky_finger3", "left_pinky_finger2", "left_pinky_finger_third_joint",
    "left_wrist",
]

RIGHT_HAND_KEYPOINTS = [
    "right_thumb4", "right_thumb3", "right_thumb2", "right_thumb_third_joint",
    "right_forefinger4", "right_forefinger3", "right_forefinger2", "right_forefinger_third_joint",
    "right_middle_finger4", "right_middle_finger3", "right_middle_finger2", "right_middle_finger_third_joint",
    "right_ring_finger4", "right_ring_finger3", "right_ring_finger2", "right_ring_finger_third_joint",
    "right_pinky_finger4", "right_pinky_finger3", "right_pinky_finger2", "right_pinky_finger_third_joint",
    "right_wrist",
]

FACE_KEYPOINTS = [
    "center_of_glabella", "center_of_nose_root", "tip_of_nose_bridge", "midpoint_1_of_nose_bridge",
    "midpoint_2_of_nose_bridge", "midpoint_3_of_nose_bridge", "center_of_labiomental_groove", "tip_of_chin",
    "upper_startpoint_of_r_eyebrow", "lower_startpoint_of_r_eyebrow", "end_of_r_eyebrow", "upper_midpoint_1_of_r_eyebrow",
    "lower_midpoint_1_of_r_eyebrow", "upper_midpoint_2_of_r_eyebrow", "upper_midpoint_3_of_r_eyebrow", "lower_midpoint_2_of_r_eyebrow",
    "lower_midpoint_3_of_r_eyebrow", "upper_startpoint_of_l_eyebrow", "lower_startpoint_of_l_eyebrow", "end_of_l_eyebrow",
    "upper_midpoint_1_of_l_eyebrow", "lower_midpoint_1_of_l_eyebrow", "upper_midpoint_2_of_l_eyebrow", "upper_midpoint_3_of_l_eyebrow",
    "lower_midpoint_2_of_l_eyebrow", "lower_midpoint_3_of_l_eyebrow", "l_inner_end_of_upper_lash_line", "l_outer_end_of_upper_lash_line",
    "l_centerpoint_of_upper_lash_line", "l_midpoint_2_of_upper_lash_line", "l_midpoint_1_of_upper_lash_line", "l_midpoint_6_of_upper_lash_line",
    "l_midpoint_5_of_upper_lash_line", "l_midpoint_4_of_upper_lash_line", "l_midpoint_3_of_upper_lash_line", "l_outer_end_of_upper_eyelid_line",
    "l_midpoint_6_of_upper_eyelid_line", "l_midpoint_2_of_upper_eyelid_line", "l_midpoint_5_of_upper_eyelid_line", "l_centerpoint_of_upper_eyelid_line",
    "l_midpoint_4_of_upper_eyelid_line", "l_midpoint_1_of_upper_eyelid_line", "l_midpoint_3_of_upper_eyelid_line", "l_midpoint_6_of_upper_crease_line",
    "l_midpoint_2_of_upper_crease_line", "l_midpoint_5_of_upper_crease_line", "l_centerpoint_of_upper_crease_line", "l_midpoint_4_of_upper_crease_line",
    "l_midpoint_1_of_upper_crease_line", "l_midpoint_3_of_upper_crease_line", "r_inner_end_of_upper_lash_line", "r_outer_end_of_upper_lash_line",
    "r_centerpoint_of_upper_lash_line", "r_midpoint_1_of_upper_lash_line", "r_midpoint_2_of_upper_lash_line", "r_midpoint_3_of_upper_lash_line",
    "r_midpoint_4_of_upper_lash_line", "r_midpoint_5_of_upper_lash_line", "r_midpoint_6_of_upper_lash_line", "r_outer_end_of_upper_eyelid_line",
    "r_midpoint_3_of_upper_eyelid_line", "r_midpoint_1_of_upper_eyelid_line", "r_midpoint_4_of_upper_eyelid_line", "r_centerpoint_of_upper_eyelid_line",
    "r_midpoint_5_of_upper_eyelid_line", "r_midpoint_2_of_upper_eyelid_line", "r_midpoint_6_of_upper_eyelid_line", "r_midpoint_3_of_upper_crease_line",
    "r_midpoint_1_of_upper_crease_line", "r_midpoint_4_of_upper_crease_line", "r_centerpoint_of_upper_crease_line", "r_midpoint_5_of_upper_crease_line",
    "r_midpoint_2_of_upper_crease_line", "r_midpoint_6_of_upper_crease_line", "l_inner_end_of_lower_lash_line", "l_outer_end_of_lower_lash_line",
    "l_centerpoint_of_lower_lash_line", "l_midpoint_2_of_lower_lash_line", "l_midpoint_1_of_lower_lash_line", "l_midpoint_6_of_lower_lash_line",
    "l_midpoint_5_of_lower_lash_line", "l_midpoint_4_of_lower_lash_line", "l_midpoint_3_of_lower_lash_line", "l_outer_end_of_lower_eyelid_line",
    "l_midpoint_6_of_lower_eyelid_line", "l_midpoint_2_of_lower_eyelid_line", "l_midpoint_5_of_lower_eyelid_line", "l_centerpoint_of_lower_eyelid_line",
    "l_midpoint_4_of_lower_eyelid_line", "l_midpoint_1_of_lower_eyelid_line", "l_midpoint_3_of_lower_eyelid_line", "r_inner_end_of_lower_lash_line",
    "r_outer_end_of_lower_lash_line", "r_centerpoint_of_lower_lash_line", "r_midpoint_1_of_lower_lash_line", "r_midpoint_2_of_lower_lash_line",
    "r_midpoint_3_of_lower_lash_line", "r_midpoint_4_of_lower_lash_line", "r_midpoint_5_of_lower_lash_line", "r_midpoint_6_of_lower_lash_line",
    "r_outer_end_of_lower_eyelid_line", "r_midpoint_3_of_lower_eyelid_line", "r_midpoint_1_of_lower_eyelid_line", "r_midpoint_4_of_lower_eyelid_line",
    "r_centerpoint_of_lower_eyelid_line", "r_midpoint_5_of_lower_eyelid_line", "r_midpoint_2_of_lower_eyelid_line", "r_midpoint_6_of_lower_eyelid_line",
    "tip_of_nose", "bottom_center_of_nose", "r_outer_corner_of_nose", "l_outer_corner_of_nose",
    "inner_corner_of_r_nostril", "outer_corner_of_r_nostril", "upper_corner_of_r_nostril", "inner_corner_of_l_nostril",
    "outer_corner_of_l_nostril", "upper_corner_of_l_nostril", "r_outer_corner_of_mouth", "l_outer_corner_of_mouth",
    "center_of_cupid_bow", "center_of_lower_outer_lip", "midpoint_1_of_upper_outer_lip", "midpoint_2_of_upper_outer_lip",
    "midpoint_1_of_lower_outer_lip", "midpoint_2_of_lower_outer_lip", "midpoint_3_of_upper_outer_lip", "midpoint_4_of_upper_outer_lip",
    "midpoint_5_of_upper_outer_lip", "midpoint_6_of_upper_outer_lip", "midpoint_3_of_lower_outer_lip", "midpoint_4_of_lower_outer_lip",
    "midpoint_5_of_lower_outer_lip", "midpoint_6_of_lower_outer_lip", "r_inner_corner_of_mouth", "l_inner_corner_of_mouth",
    "center_of_upper_inner_lip", "center_of_lower_inner_lip", "midpoint_1_of_upper_inner_lip", "midpoint_2_of_upper_inner_lip",
    "midpoint_1_of_lower_inner_lip", "midpoint_2_of_lower_inner_lip", "midpoint_3_of_upper_inner_lip", "midpoint_4_of_upper_inner_lip",
    "midpoint_5_of_upper_inner_lip", "midpoint_6_of_upper_inner_lip", "midpoint_3_of_lower_inner_lip", "midpoint_4_of_lower_inner_lip",
    "midpoint_5_of_lower_inner_lip", "midpoint_6_of_lower_inner_lip", "l_top_end_of_inferior_crus", "l_top_end_of_superior_crus",
    "l_start_of_antihelix", "l_end_of_antihelix", "l_midpoint_1_of_antihelix", "l_midpoint_1_of_inferior_crus",
    "l_midpoint_2_of_antihelix", "l_midpoint_3_of_antihelix", "l_point_1_of_inner_helix", "l_point_2_of_inner_helix",
    "l_point_3_of_inner_helix", "l_point_4_of_inner_helix", "l_point_5_of_inner_helix", "l_point_6_of_inner_helix",
    "l_point_7_of_inner_helix", "l_highest_point_of_antitragus", "l_bottom_point_of_tragus", "l_protruding_point_of_tragus",
    "l_top_point_of_tragus", "l_start_point_of_crus_of_helix", "l_deepest_point_of_concha", "l_tip_of_ear_lobe",
    "l_midpoint_between_22_15", "l_bottom_connecting_point_of_ear_lobe", "l_top_connecting_point_of_helix", "l_point_8_of_inner_helix",
    "r_top_end_of_inferior_crus", "r_top_end_of_superior_crus", "r_start_of_antihelix", "r_end_of_antihelix",
    "r_midpoint_1_of_antihelix", "r_midpoint_1_of_inferior_crus", "r_midpoint_2_of_antihelix", "r_midpoint_3_of_antihelix",
    "r_point_1_of_inner_helix", "r_point_8_of_inner_helix", "r_point_3_of_inner_helix", "r_point_4_of_inner_helix",
    "r_point_5_of_inner_helix", "r_point_6_of_inner_helix", "r_point_7_of_inner_helix", "r_highest_point_of_antitragus",
    "r_bottom_point_of_tragus", "r_protruding_point_of_tragus", "r_top_point_of_tragus", "r_start_point_of_crus_of_helix",
    "r_deepest_point_of_concha", "r_tip_of_ear_lobe", "r_midpoint_between_22_15", "r_bottom_connecting_point_of_ear_lobe",
    "r_top_connecting_point_of_helix", "r_point_2_of_inner_helix", "l_center_of_iris", "l_border_of_iris_3",
    "l_border_of_iris_midpoint_1", "l_border_of_iris_12", "l_border_of_iris_midpoint_4", "l_border_of_iris_9",
    "l_border_of_iris_midpoint_3", "l_border_of_iris_6", "l_border_of_iris_midpoint_2", "r_center_of_iris",
    "r_border_of_iris_3", "r_border_of_iris_midpoint_1", "r_border_of_iris_12", "r_border_of_iris_midpoint_4",
    "r_border_of_iris_9", "r_border_of_iris_midpoint_3", "r_border_of_iris_6", "r_border_of_iris_midpoint_2",
    "l_center_of_pupil", "l_border_of_pupil_3", "l_border_of_pupil_midpoint_1", "l_border_of_pupil_12",
    "l_border_of_pupil_midpoint_4", "l_border_of_pupil_9", "l_border_of_pupil_midpoint_3", "l_border_of_pupil_6",
    "l_border_of_pupil_midpoint_2", "r_center_of_pupil", "r_border_of_pupil_3", "r_border_of_pupil_midpoint_1",
    "r_border_of_pupil_12", "r_border_of_pupil_midpoint_4", "r_border_of_pupil_9", "r_border_of_pupil_midpoint_3",
    "r_border_of_pupil_6", "r_border_of_pupil_midpoint_2",
]

BODY_LIMBS_NAMES = [
    ("left_ankle", "left_knee"), ("left_knee", "left_hip"), ("right_ankle", "right_knee"), ("right_knee", "right_hip"),
    ("left_hip", "right_hip"), ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"), ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"), ("left_elbow", "left_wrist_body"), ("right_elbow", "right_wrist_body"),
    ("left_eye", "right_eye"), ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"), ("right_eye", "right_ear"), 
    ("left_ear", "left_shoulder"), ("right_ear", "right_shoulder"), ("left_ankle", "left_big_toe"), ("left_ankle", "left_small_toe"),
    ("left_ankle", "left_heel"), ("right_ankle", "right_big_toe"), ("right_ankle", "right_small_toe"), ("right_ankle", "right_heel"),
]

LEFT_HAND_LIMBS_NAMES = [
    ("left_wrist", "left_thumb_third_joint"), ("left_thumb_third_joint", "left_thumb2"), ("left_thumb2", "left_thumb3"), ("left_thumb3", "left_thumb4"),
    ("left_wrist", "left_forefinger_third_joint"), ("left_forefinger_third_joint", "left_forefinger2"), ("left_forefinger2", "left_forefinger3"), ("left_forefinger3", "left_forefinger4"),
    ("left_wrist", "left_middle_finger_third_joint"), ("left_middle_finger_third_joint", "left_middle_finger2"), ("left_middle_finger2", "left_middle_finger3"), ("left_middle_finger3", "left_middle_finger4"),
    ("left_wrist", "left_ring_finger_third_joint"), ("left_ring_finger_third_joint", "left_ring_finger2"), ("left_ring_finger2", "left_ring_finger3"), ("left_ring_finger3", "left_ring_finger4"),
    ("left_wrist", "left_pinky_finger_third_joint"), ("left_pinky_finger_third_joint", "left_pinky_finger2"), ("left_pinky_finger2", "left_pinky_finger3"), ("left_pinky_finger3", "left_pinky_finger4"),
]

RIGHT_HAND_LIMBS_NAMES = [
    ("right_wrist", "right_thumb_third_joint"), ("right_thumb_third_joint", "right_thumb2"), ("right_thumb2", "right_thumb3"), ("right_thumb3", "right_thumb4"),
    ("right_wrist", "right_forefinger_third_joint"), ("right_forefinger_third_joint", "right_forefinger2"), ("right_forefinger2", "right_forefinger3"), ("right_forefinger3", "right_forefinger4"),
    ("right_wrist", "right_middle_finger_third_joint"), ("right_middle_finger_third_joint", "right_middle_finger2"), ("right_middle_finger2", "right_middle_finger3"), ("right_middle_finger3", "right_middle_finger4"),
    ("right_wrist", "right_ring_finger_third_joint"), ("right_ring_finger_third_joint", "right_ring_finger2"), ("right_ring_finger2", "right_ring_finger3"), ("right_ring_finger3", "right_ring_finger4"),
    ("right_wrist", "right_pinky_finger_third_joint"), ("right_pinky_finger_third_joint", "right_pinky_finger2"), ("right_pinky_finger2", "right_pinky_finger3"), ("right_pinky_finger3", "right_pinky_finger4"),
]

GENERIC_HAND_KEYPOINTS = [element.replace("right_","") for element in RIGHT_HAND_KEYPOINTS]

# Global index lookup
name_to_global = {name: i for i, name in enumerate(EXTENDED_GOLIATH_KEYPOINTS)}

LEFT_WRIST_IDX = name_to_global["left_wrist"]
RIGHT_WRIST_IDX = name_to_global["right_wrist"]

LEFT_WRIST_BODY_IDX = name_to_global["left_wrist_body"]
RIGHT_WRIST_BODY_IDX = name_to_global["right_wrist_body"]

# --------------------------------------------------
# PoseHeader components
# --------------------------------------------------

def get_sapiens_components():
    """
    Create PoseHeader components for Sapiens (Goliath),
    split into BODY_SAPIENS / FACE_SAPIENS / LEFT_HAND_SAPIENS / RIGHT_HAND_SAPIENS.
    """

    def build_limbs(keypoint_names, named_limbs):
        # Global -> local index mapping
        global_indices = [name_to_global[k] for k in keypoint_names]
        global_to_local = {
            g: i for i, g in enumerate(global_indices)
        }
        limbs = []
        # Build limbs only if both endpoints belong to this component
        for link in named_limbs:
            a = link[0]
            b = link[1]
            if a in name_to_global and b in name_to_global:
                ga, gb = name_to_global[a], name_to_global[b]
                if ga in global_to_local and gb in global_to_local:
                    limbs.append(
                        (global_to_local[ga], global_to_local[gb])
                    )

        return limbs

    body_limbs = build_limbs(BODY_KEYPOINTS, BODY_LIMBS_NAMES)
    l_hand_limbs = build_limbs(LEFT_HAND_KEYPOINTS, LEFT_HAND_LIMBS_NAMES)
    r_hand_limbs = build_limbs(RIGHT_HAND_KEYPOINTS, RIGHT_HAND_LIMBS_NAMES)

    components = [
        PoseHeaderComponent(
            name="BODY_SAPIENS",
            points=BODY_KEYPOINTS,
            limbs=body_limbs,  
            colors=[(0, 255, 0)] * len(body_limbs),
            point_format="XYC"
        ),

        PoseHeaderComponent(
            name="FACE_SAPIENS",
            points=FACE_KEYPOINTS,
            limbs=[],   # WholeBody face mesh is huge, usually omitted
            colors=[(255,255,255)],
            point_format="XYC"
        ),

        PoseHeaderComponent(
            name="LEFT_HAND_SAPIENS",
            points=GENERIC_HAND_KEYPOINTS,
            limbs=l_hand_limbs,
            colors=[(255, 0, 0)] * len(l_hand_limbs),
            point_format="XYC"
        ),

        PoseHeaderComponent(
            name="RIGHT_HAND_SAPIENS",
            points=GENERIC_HAND_KEYPOINTS,
            limbs=r_hand_limbs,
            colors=[(0, 0, 255)] * len(r_hand_limbs),
            point_format="XYC"
        ),
    ]

    return components


def load_sapiens_wholebody_from_json(
    input_path: str,
    version: float = 0.1,
    fps: float = 24,
    width: int = 1000,
    height: int = 1000,
    depth: int = 0,
) -> Pose:
    """
    Load Sapiens Goliath poses into pose-format.

    Parameters
    ----------
    input_path : str
        Path to Sapiens JSON file.

    Returns
    -------
    Pose
        Pose-format Pose object.
    """
    print("Loading pose with Sapiens (Goliath)...")

    frames, metadata = load_sapiens_json(input_path)

    # Override metadata if present
    fps = metadata.get("fps", fps)
    width = metadata.get("width", width)
    height = metadata.get("height", height)

    num_keypoints = metadata.get("num_keypoints", len(GOLIATH_KEYPOINTS))
    assert num_keypoints == len(GOLIATH_KEYPOINTS), \
        "Mismatch between JSON and GOLIATH_KEYPOINTS"

    frames_xy = []
    frames_conf = []

    for frame in frames:
        xy, conf = parse_sapiens_frame(frame, GOLIATH_KEYPOINTS)
        xy, conf = duplicate_wrists(xy, conf)
        xy, conf = reorder_sapiens_keypoints(xy, conf)
        frames_xy.append(xy)
        frames_conf.append(conf)

    # Convert to arrays
    xy_data = np.stack(frames_xy, axis=0)      # (T, K, 2)
    conf_data = np.stack(frames_conf, axis=0)  # (T, K)

    # Add people dimension (single person)
    xy_data = xy_data[:, None, :, :]            # (T, 1, K, 2)
    conf_data = conf_data[:, None, :]           # (T, 1, K)

    # Build header
    header = PoseHeader(
        version=version,
        dimensions=PoseHeaderDimensions(
            width=width,
            height=height,
            depth=depth,
        ),
        components=get_sapiens_components(),
    )

    # Build body
    body = NumPyPoseBody(
        fps=fps,
        data=xy_data,
        confidence=conf_data,
    )

    return Pose(header, body)

def load_sapiens_json(json_path):
    """
    Load a Sapiens (Goliath) pose JSON file.

    The JSON file is expected to contain pose estimates for a single person
    across multiple video frames, stored in a name-based keypoint format.

    Expected JSON structure
    -----------------------
    {
      "metadata": {
        "fps": float,                # Frames per second of the source video
        "width": int,                # Frame width in pixels
        "height": int,               # Frame height in pixels
        "num_keypoints": int,        # Number of keypoints (len(GOLIATH_KEYPOINTS))
      },

      "frames": [
        {
          "frame": int,              # Frame index (0-based)
          "keypoints": {
            "<keypoint_name>": [x, y, score],
            ...
          }
        },
        ...
      ]
    }

    Notes
    -----
    - Keypoints are stored as a dictionary mapping keypoint names to
      (x, y, confidence) triplets.
    - Missing keypoints may be omitted from a frame's "keypoints" dictionary
      and should be treated as (0, 0, 0).

    Parameters
    ----------
    json_path : str
        Path to the Sapiens pose JSON file.

    Returns
    -------
    frames : list of dict
        List of per-frame pose annotations, sorted by frame index.
    metadata : dict
        Metadata dictionary describing the video and keypoint layout.
    """

    with open(json_path, "r") as f:
        raw = json.load(f)

    assert isinstance(raw, dict) and "frames" in raw, \
        "Invalid Sapiens JSON format"

    frames = raw["frames"]
    metadata = raw.get("metadata", {})

    # Sort frames just in case
    frames = sorted(frames, key=lambda x: x.get("frame", 0))

    return frames, metadata


def parse_sapiens_frame(frame, keypoint_names):
    """
    Parse a single Sapiens frame using named keypoints.

    Returns
    -------
    xy : (K, 2)
    conf : (K,)
    """

    kpts = frame["keypoints"]
    K = len(keypoint_names)

    xy = np.zeros((K, 2), dtype=np.float32)
    conf = np.zeros((K,), dtype=np.float32)

    for i, name in enumerate(keypoint_names):
        if name in kpts:
            x, y, score = kpts[name]
            xy[i, 0] = x
            xy[i, 1] = y
            conf[i] = score
        else:
            # missing keypoint → already zeroed
            pass

    return xy, conf


def duplicate_wrists(xy, conf):
    """
    Duplicate wrist keypoints to create separate BODY wrist entries.

    This function extends the original set of keypoints by duplicating the
    left and right wrist positions. The duplicated wrist keypoints are added
    as dedicated BODY wrist points (``left_wrist_body`` and
    ``right_wrist_body``), while preserving the original wrist keypoints
    used by the LEFT_HAND and RIGHT_HAND components.

    Motivation
    ----------
    In the Sapiens (Goliath) layout, wrist keypoints conceptually belong to
    both the body skeleton and the hand skeletons. However, the pose_format
    library expects BODY, LEFT_HAND, and RIGHT_HAND components to be
    independent and self-contained.

    By duplicating the wrist keypoints:
    - The BODY component includes its own wrist joints, enabling correct
      body skeleton connectivity and more accurate pose drawing.
    - The LEFT_HAND and RIGHT_HAND components retain their original wrist
      joints as hand roots.
    - Component-specific normalization and transformation utilities in
      pose_format can be applied correctly without sharing indices across
      components.

    Parameters
    ----------
    xy : np.ndarray, shape (K, 2)
        Array of 2D keypoint coordinates (x, y) in the global Goliath order.
    conf : np.ndarray, shape (K,)
        Array of confidence scores corresponding to each keypoint.

    Returns
    -------
    xy_ext : np.ndarray, shape (K + 2, 2)
        Extended array of keypoint coordinates including duplicated BODY
        wrist keypoints.
    conf_ext : np.ndarray, shape (K + 2,)
        Extended array of confidence scores including duplicated BODY
        wrist keypoints.

    Notes
    -----
    - The indices of the duplicated wrist keypoints are determined by
      ``LEFT_WRIST_BODY_IDX`` and ``RIGHT_WRIST_BODY_IDX``.
    - This operation does not modify the original hand wrist keypoints;
      it only creates additional BODY-specific wrist entries.
    """
    K = xy.shape[0]

    xy_ext = np.zeros((K + 2, 2), dtype=xy.dtype)
    conf_ext = np.zeros((K + 2,), dtype=conf.dtype)

    # copy original
    xy_ext[:K] = xy
    conf_ext[:K] = conf

    # duplicate wrists
    xy_ext[LEFT_WRIST_BODY_IDX] = xy[LEFT_WRIST_IDX]
    xy_ext[RIGHT_WRIST_BODY_IDX] = xy[RIGHT_WRIST_IDX]

    conf_ext[LEFT_WRIST_BODY_IDX] = conf[LEFT_WRIST_IDX]
    conf_ext[RIGHT_WRIST_BODY_IDX] = conf[RIGHT_WRIST_IDX]

    return xy_ext, conf_ext


def reorder_sapiens_keypoints(xy, conf):
    """
    Reorder Sapiens/Goliath keypoints to:
        [BODY, FACE, LEFT_HAND, RIGHT_HAND]

    Assumes:
    - xy, conf are indexed according to EXTENDED_GOLIATH_KEYPOINTS
      (i.e. after duplicate_wrists)

    Parameters
    ----------
    xy : np.ndarray, shape (K, 2)
    conf : np.ndarray, shape (K,)

    Returns
    -------
    xy_reordered : np.ndarray, shape (K, 2)
    conf_reordered : np.ndarray, shape (K,)
    """

    reordered_names = (
        BODY_KEYPOINTS
        + FACE_KEYPOINTS
        + LEFT_HAND_KEYPOINTS
        + RIGHT_HAND_KEYPOINTS
    )

    name_to_idx = {
        name: i for i, name in enumerate(EXTENDED_GOLIATH_KEYPOINTS)
    }

    if set(reordered_names) != set(EXTENDED_GOLIATH_KEYPOINTS):
        missing = set(EXTENDED_GOLIATH_KEYPOINTS) - set(reordered_names)
        extra = set(reordered_names) - set(EXTENDED_GOLIATH_KEYPOINTS)
        raise ValueError(
            f"Keypoint mismatch.\nMissing: {missing}\nExtra: {extra}"
        )

    reorder_indices = np.array(
        [name_to_idx[name] for name in reordered_names],
        dtype=np.int64
    )

    xy_reordered = xy[reorder_indices]
    conf_reordered = conf[reorder_indices]

    return xy_reordered, conf_reordered
