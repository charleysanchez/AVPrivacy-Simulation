#!/usr/bin/env python3
# generate_carla_demo.py
#
# Spawns an ego vehicle and 5–10 pedestrians near it, makes them walk around,
# and saves RGB (+ optional depth) frames to a session folder.
# Works with CARLA 0.9.11 and Python 3.7.

import os
import sys
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict

# ------------------ Make CARLA importable from ../carla_sim/Carla911 ------------------
def _add_carla_to_path():
    carla_root = os.environ.get("CARLA_ROOT")
    if not carla_root:
        # Assume this script lives in .../AVPrivacy-Simulation/scripts
        here = Path(__file__).resolve()
        guess = here.parents[1] / "carla_sim" / "Carla911"
        if (guess / "PythonAPI").exists():
            carla_root = str(guess)

    if not carla_root:
        return False

    pyapi = Path(carla_root) / "PythonAPI"
    dist = pyapi / "carla" / "dist"

    # Match current Python minor (e.g., py3.7)
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    eggs = sorted(dist.glob(f"carla-*py{ver}-*.egg"))
    if eggs:
        sys.path.append(str(eggs[0]))
    sys.path.append(str(pyapi / "carla"))
    sys.path.append(str(pyapi / "carla" / "agents"))
    return True

if not _add_carla_to_path():
    print("ERROR: CARLA Python API not found. "
          "Set CARLA_ROOT or place CARLA at ../carla_sim/Carla911")
    sys.exit(1)

import carla  # noqa: E402
import numpy as np  # noqa: E402

# ------------------ CLI ------------------
ap = argparse.ArgumentParser("Spawn ego + nearby pedestrians and record frames")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=2000)
ap.add_argument("--town", default="Town03", help="Map name (e.g., Town03)")
ap.add_argument("--num-peds", type=int, default=8, help="# pedestrians to spawn (5–10 recommended)")
ap.add_argument("--radius", type=float, default=12.0, help="Spawn pedestrians within this radius (m)")
ap.add_argument("--min-dist", type=float, default=3.0, help="Keep pedestrians at least this far from ego (m)")
ap.add_argument("--fps", type=float, default=20.0, help="Simulation FPS (sync mode)")
ap.add_argument("--duration", type=float, default=15.0, help="Recording duration (seconds)")
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--out-root", default="../data/carla_demo", help="Output root (relative to this script)")
ap.add_argument("--save-depth", action="store_true", help="Also save linearized depth as 16-bit PNG")
ap.add_argument("--ego-autopilot", action="store_true", help="Enable autopilot for ego")
args = ap.parse_args()

# ------------------ Helpers ------------------
def depth_image_to_meters(image: carla.Image) -> np.ndarray:
    """
    Convert CARLA depth image to meters (approx).
    CARLA depth is encoded in 24 bits: [R,G,B] ~ [0..1].
    """
    w, h = image.width, image.height
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((h, w, 4))
    # Normalize to [0,1]
    depth_norm = (array[:, :, 2].astype(np.float32) * 65536.0 +
                  array[:, :, 1].astype(np.float32) * 256.0 +
                  array[:, :, 0].astype(np.float32)) / (256.0**3 - 1.0)
    # Default far plane in CARLA is ~1000 m for depth camera; scale accordingly.
    depth_m = 1000.0 * depth_norm
    return depth_m

def rand_near(loc: carla.Location, rng: np.random.RandomState, r_min: float, r_max: float) -> carla.Location:
    """Pick a random nearby target location in a donut [r_min, r_max] around loc."""
    r = rng.uniform(r_min, r_max)
    theta = rng.uniform(0, 2*np.pi)
    return carla.Location(x=loc.x + r * np.cos(theta),
                          y=loc.y + r * np.sin(theta),
                          z=loc.z)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ------------------ Connect & world setup ------------------
client = carla.Client(args.host, args.port)
client.set_timeout(10.0)

world = client.get_world()
if args.town and (world.get_map().name.split('/')[-1] != args.town):
    world = client.load_world(args.town)

original_settings = world.get_settings()
settings = carla.WorldSettings(
    no_rendering_mode=False,
    synchronous_mode=True,
    fixed_delta_seconds=1.0 / args.fps
)
world.apply_settings(settings)

tm = client.get_trafficmanager()
tm.set_synchronous_mode(True)

blueprints = world.get_blueprint_library()

spawn_points = world.get_map().get_spawn_points()
rng = np.random.RandomState(7)

# ------------------ Spawn ego vehicle ------------------
vehicle_bp = blueprints.filter("vehicle.tesla.model3")[0] if blueprints.filter("vehicle.tesla.model3") else random.choice(blueprints.filter("vehicle.*"))
ego_tf = rng.choice(spawn_points)
ego = world.try_spawn_actor(vehicle_bp, ego_tf)
if not ego:
    # fallback
    for tf in spawn_points:
        ego = world.try_spawn_actor(vehicle_bp, tf)
        if ego:
            break
if not ego:
    print("ERROR: Could not spawn ego vehicle.")
    # Restore settings
    world.apply_settings(original_settings)
    sys.exit(1)

ego.set_autopilot(args.ego_autopilot, tm.get_port())

# ------------------ Attach sensors ------------------
cam_bp = blueprints.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", str(args.width))
cam_bp.set_attribute("image_size_y", str(args.height))
cam_bp.set_attribute("fov", "90")
cam_tf = carla.Transform(carla.Location(x=0.4, z=1.6))  # roof-ish
cam_rgb = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

depth_actor = None
if args.save_depth:
    depth_bp = blueprints.find("sensor.camera.depth")
    depth_bp.set_attribute("image_size_x", str(args.width))
    depth_bp.set_attribute("image_size_y", str(args.height))
    depth_bp.set_attribute("fov", "90")
    depth_tf = carla.Transform(carla.Location(x=0.45, z=1.6))
    depth_actor = world.spawn_actor(depth_bp, depth_tf, attach_to=ego)

# ------------------ Output dirs (session-style) ------------------
root = Path(__file__).resolve().parents[1] / Path(args.out_root)
ts = time.strftime("%Y-%m-%d_%H-%M-%S")
session_dir = root / f"session_{ts}_carla"
orig_dir = session_dir / "original"
depth_dir = session_dir / "depth"
ensure_dir(orig_dir)
if args.save_depth:
    ensure_dir(depth_dir)

print(f"Saving frames to: {session_dir}")

# ------------------ Spawn pedestrians near ego ------------------
walker_bps = blueprints.filter("walker.pedestrian.*")
controller_bp = blueprints.find("controller.ai.walker")

def pick_nav_point_near(center: carla.Location, r_min: float, r_max: float, tries=30):
    for _ in range(tries):
        loc = rand_near(center, rng, r_min, r_max)
        nav = world.get_random_location_from_navigation()
        # Prefer nav points close to our sampled loc and near ego
        if nav and nav.distance(center) <= r_max + 5.0 and nav.distance(center) >= r_min * 0.5:
            return nav
    return None

walkers = []
controllers = []
to_spawn = max(1, min(20, args.num_peds))

ego_loc = ego.get_location()
for _ in range(to_spawn):
    nav = pick_nav_point_near(ego_loc, args.min_dist, args.radius)
    if not nav:
        continue
    walker_bp = rng.choice(walker_bps)
    # speed attribute (0 = walk, 1 = run ratio)
    if walker_bp.has_attribute('is_invincible'):
        walker_bp.set_attribute('is_invincible', 'false')
    walker = world.try_spawn_actor(walker_bp, carla.Transform(nav, carla.Rotation(yaw=rng.uniform(-180, 180))))
    if walker:
        ctrl = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        if ctrl:
            walkers.append(walker)
            controllers.append(ctrl)

world.tick()

# Start controllers and set initial goals near ego
for w, c in zip(walkers, controllers):
    c.start()
    # Set a walking speed (m/s). If walker_bp supports 'speed', you can sample it; else set a typical ~1.3 m/s
    try:
        c.set_max_speed(1.4 + rng.uniform(-0.3, 0.3))
    except Exception:
        pass
    target = pick_nav_point_near(ego_loc, args.min_dist, args.radius) or ego_loc
    c.go_to_location(target)

print(f"Spawned ego + {len(walkers)} pedestrians near the vehicle (r ≤ {args.radius} m).")

# ------------------ Sensor callbacks ------------------
frame_written = defaultdict(bool)

def save_rgb(image: carla.Image):
    # Name by simulation frame for easy alignment
    out = orig_dir / f"frame_{image.frame}.png"
    image.save_to_disk(str(out), carla.ColorConverter.Raw)

def save_depth(image: carla.Image):
    # Convert to meters and save as 16-bit PNG (millimeters) for precision
    depth_m = depth_image_to_meters(image)
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    out = depth_dir / f"frame_{image.frame}.png"
    import cv2
    cv2.imwrite(str(out), depth_mm)

cam_rgb.listen(save_rgb)
if depth_actor:
    depth_actor.listen(save_depth)

# ------------------ Drive + keep pedestrians near ego ------------------
n_frames = int(args.duration * args.fps)
reassign_interval = int(2.5 * args.fps)  # retarget every ~2.5s to stay near ego

try:
    for frame in range(n_frames):
        world.tick()

        # Every few seconds, refresh targets around the *current* ego location
        if frame % reassign_interval == 0:
            ego_loc = ego.get_location()
            for c in controllers:
                tgt = pick_nav_point_near(ego_loc, args.min_dist, args.radius) or ego_loc
                try:
                    c.go_to_location(tgt)
                except RuntimeError:
                    pass

        if frame % int(args.fps) == 0:
            print(f"[{frame:05d}/{n_frames}] recording…")

finally:
    print("Cleaning up actors…")
    # Stop listeners
    try:
        cam_rgb.stop()
    except Exception:
        pass
    if depth_actor:
        try:
            depth_actor.stop()
        except Exception:
            pass

    # Stop controllers
    for c in controllers:
        try:
            c.stop()
        except Exception:
            pass

    # Destroy actors
    actor_ids = [a.id for a in controllers + walkers + [cam_rgb] + ([depth_actor] if depth_actor else []) + [ego] if a]
    for a in controllers + walkers + ([cam_rgb] if cam_rgb else []) + ([depth_actor] if depth_actor else []) + ([ego] if ego else []):
        try:
            a.destroy()
        except Exception:
            pass

    # Restore world settings
    world.apply_settings(original_settings)
    tm.set_synchronous_mode(False)

    print(f"Saved to: {session_dir}")