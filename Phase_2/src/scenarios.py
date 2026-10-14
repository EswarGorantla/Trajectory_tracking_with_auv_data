from typing import Dict, Any, List

def get_canonical_scenarios() -> Dict[str, Dict[str, Any]]:
    """Returns the standardized, reproducible benchmark scenario suite."""
    scenarios = {
        "point_to_point_return": {
            "name": "point_to_point_return",
            "desc": "Round-Trip transit A -> B -> A with in-place 180° turnaround at Point B",
            "start": [0.0, 0.0, 20.0],
            "goal": [60.0, 50.0, 30.0],
            "boundaries": {"north_min": -10.0, "north_max": 100.0, "east_min": -10.0, "east_max": 100.0, "depth_min": 10.0, "depth_max": 60.0},
            "obstacles": [],
            "current": [0.0, 0.0, 0.0],
            "return_to_start": True
        },
        "basic_A_to_B": {
            "name": "basic_A_to_B",
            "desc": "Baseline 3D transit from (0,0,20) to (100,80,40)",
            "start": [0.0, 0.0, 20.0],
            "goal": [100.0, 80.0, 40.0],
            "boundaries": {"north_min": 0.0, "north_max": 120.0, "east_min": 0.0, "east_max": 100.0, "depth_min": 10.0, "depth_max": 60.0},
            "obstacles": [],
            "current": [0.0, 0.0, 0.0],
            "return_to_start": True
        },
        "case_1_diagonal": {
            "name": "case_1_diagonal",
            "desc": "Diagonal climbing dive across volume",
            "start": [0.0, 0.0, 20.0],
            "goal": [100.0, 80.0, 40.0],
            "boundaries": {"north_min": 0.0, "north_max": 120.0, "east_min": 0.0, "east_max": 100.0, "depth_min": 10.0, "depth_max": 60.0},
            "obstacles": [],
            "current": [0.0, 0.0, 0.0]
        },
        "case_2_corridor": {
            "name": "case_2_corridor",
            "desc": "Interior boundary transit with Eastward bias",
            "start": [20.0, 10.0, 30.0],
            "goal": [80.0, 90.0, 50.0],
            "boundaries": {"north_min": 0.0, "north_max": 120.0, "east_min": 0.0, "east_max": 100.0, "depth_min": 10.0, "depth_max": 60.0},
            "obstacles": [],
            "current": [0.0, 0.0, 0.0]
        },
        "case_3_ascent": {
            "name": "case_3_ascent",
            "desc": "South-East ascent transit (High depth to shallow)",
            "start": [100.0, 20.0, 50.0],
            "goal": [20.0, 80.0, 20.0],
            "boundaries": {"north_min": 0.0, "north_max": 120.0, "east_min": 0.0, "east_max": 100.0, "depth_min": 10.0, "depth_max": 60.0},
            "obstacles": [],
            "current": [0.0, 0.0, 0.0]
        },
        "boundary_corridor": {
            "name": "boundary_corridor",
            "desc": "Constrained corridor with East wall at 70m",
            "start": [10.0, 10.0, 25.0],
            "goal": [85.0, 55.0, 35.0],
            "boundaries": {"north_min": 0.0, "north_max": 120.0, "east_min": 0.0, "east_max": 70.0, "depth_min": 10.0, "depth_max": 60.0},
            "obstacles": [],
            "current": [0.0, 0.0, 0.0]
        },
        "obstacle_avoidance": {
            "name": "obstacle_avoidance",
            "desc": "Cylindrical obstacle (r=8m, safe=12m) centered at (50,40) blocking LOS",
            "start": [0.0, 0.0, 20.0],
            "goal": [100.0, 80.0, 40.0],
            "boundaries": {"north_min": 0.0, "north_max": 120.0, "east_min": 0.0, "east_max": 100.0, "depth_min": 10.0, "depth_max": 60.0},
            "obstacles": [
                {"north": 50.0, "east": 40.0, "radius": 8.0, "margin": 4.0, "depth_min": 0.0, "depth_max": 80.0}
            ],
            "current": [0.0, 0.0, 0.0]
        },
        "robustness_current_noise": {
            "name": "robustness_current_noise",
            "desc": "0.20 m/s ocean cross-current drift plus 0.5m acoustic USBL jitter",
            "start": [0.0, 0.0, 20.0],
            "goal": [100.0, 80.0, 40.0],
            "boundaries": {"north_min": 0.0, "north_max": 120.0, "east_min": 0.0, "east_max": 100.0, "depth_min": 10.0, "depth_max": 60.0},
            "obstacles": [],
            "current": [0.15, -0.12, 0.05]
        },
        "stress_narrow_gap": {
            "name": "stress_narrow_gap",
            "desc": "Two obstacles forming a narrow 8m corridor directly on transit line",
            "start": [0.0, 0.0, 20.0],
            "goal": [100.0, 80.0, 40.0],
            "boundaries": {"north_min": 0.0, "north_max": 120.0, "east_min": 0.0, "east_max": 100.0, "depth_min": 10.0, "depth_max": 60.0},
            "obstacles": [
                {"north": 50.0, "east": 34.0, "radius": 5.0, "margin": 3.0, "depth_min": 0.0, "depth_max": 80.0},
                {"north": 50.0, "east": 48.0, "radius": 5.0, "margin": 3.0, "depth_min": 0.0, "depth_max": 80.0}
            ],
            "current": [0.0, 0.0, 0.0]
        }
    }
    return scenarios
