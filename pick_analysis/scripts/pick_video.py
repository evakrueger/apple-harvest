#!/usr/bin/env python3
"""
Render a pick bag as a video: gripper webcam in the centre, vacuum / flex / ToF / force drawn live on either side,
and a green line at the pick time annotated in ../labels/PICK_TIMES_2026.progress.json.

  python3 pick_video.py <bag path> [--start s] [--speed x] [--labels ../labels/PICK_TIMES_2026.progress.json] [--out x.mp4]

Times are seconds from the bag start using receive times, the same zero as label_bags.py.
Flex = ||flex / 4|| and force = |F| with a 21-sample moving mean, as in rf_pick_classifier.py.
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray, Int32

PICK_ANALYSIS = Path(__file__).resolve().parent.parent  # pick_analysis/: scripts/, labels/, plots/, videos/
CAM = '/usb_cam_image_raw'
TOPICS = {'/vacuum_pressure': Float32, '/flex_sensor_data': Float32MultiArray, '/tof_sensor_data': Int32,
          '/force_torque_sensor_broadcaster/wrench': WrenchStamped, CAM: Image}
FPS = 30
W, H, DPI = 1920, 1080, 100
FONT = 20  # base font size (matplotlib's default is 10)


def to_rgb(msg):
    raw = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    enc = msg.encoding.lower()
    if enc in ('yuv422_yuy2', 'yuyv'):
        return cv2.cvtColor(raw[:, :msg.width * 2].reshape(msg.height, msg.width, 2), cv2.COLOR_YUV2RGB_YUY2)
    img = raw[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if enc == 'bgr8' else img.copy()


def read_bag(path, start_ns):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3'), rosbag2_py.ConverterOptions('', ''))
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(TOPICS)))
    data = {t: ([], []) for t in TOPICS}
    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        msg = deserialize_message(raw, TOPICS[topic])
        if topic == CAM:
            val = to_rgb(msg)
        elif topic == '/flex_sensor_data':
            val = float(np.linalg.norm(np.asarray(msg.data, dtype=float) / 4.0))
        elif topic == '/force_torque_sensor_broadcaster/wrench':
            f = msg.wrench.force
            val = float(np.sqrt(f.x ** 2 + f.y ** 2 + f.z ** 2))
        else:
            val = float(msg.data)
        data[topic][0].append((t_ns - start_ns) / 1e9)
        data[topic][1].append(val)
    out = {t: (np.array(ts), vals) for t, (ts, vals) in data.items()}
    ft, fv = out['/force_torque_sensor_broadcaster/wrench']
    fv = np.asarray(fv)
    # trailing 21-sample mean, like the classifier's live buffer
    csum = np.cumsum(np.insert(fv, 0, 0.0))
    idx = np.arange(1, len(fv) + 1)
    lo = np.maximum(idx - 21, 0)
    out['/force_torque_sensor_broadcaster/wrench'] = (ft, (csum[idx] - csum[lo]) / (idx - lo))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('bag')
    p.add_argument('--labels', default=str(PICK_ANALYSIS / 'labels' / 'PICK_TIMES_2026.progress.json'))
    p.add_argument('--out')
    p.add_argument('--start', type=float, default=0.0, help='start the video at this time (s from bag start)')
    p.add_argument('--gamma', type=float, default=0.75, help='webcam gamma; below 1 brightens, 1 leaves it as recorded')
    p.add_argument('--speed', type=float, default=1.0, help='playback speed (0.5 = half speed)')
    args = p.parse_args()

    entry = json.loads(Path(args.labels).read_text())[args.bag]
    bag = entry['bag']
    t_mark = entry['time_s']
    out_path = args.out or str(PICK_ANALYSIS / 'videos' / (Path(args.bag).stem + '_pick.mp4'))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    data = read_bag(args.bag, bag['start_ns'])
    cam_t, cam_imgs = data[CAM]
    lut = np.clip(255 * (np.arange(256) / 255) ** args.gamma, 0, 255).astype(np.uint8)
    cam_imgs = [cv2.LUT(img, lut) for img in cam_imgs]
    series = [  # (topic, title, unit, colour)
        ('/vacuum_pressure', 'Vacuum pressure  (kPa)', '', '#1565c0'),
        ('/tof_sensor_data', 'ToF  (mm)', '', '#6a1b9a'),
        ('/flex_sensor_data', 'Flex  ‖flex / 4‖  (°)', '', '#ef6c00'),
        ('/force_torque_sensor_broadcaster/wrench', 'Force  |F|  (N)', '', '#c62828'),
    ]

    plt.rcParams['font.size'] = FONT
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor='white')
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.45, 1], left=0.055, right=0.985, top=0.94, bottom=0.09,
                          wspace=0.2, hspace=0.42)
    ax_cam = fig.add_subplot(gs[:, 1])
    ax_cam.set_axis_off()
    im = ax_cam.imshow(cam_imgs[0])
    clock = ax_cam.text(0.5, -0.04, '', transform=ax_cam.transAxes, ha='center', va='top', fontsize=2.2 * FONT,
                        family='monospace')

    t_end = bag['duration_s']
    lines, cursors, pick_lines = [], [], []
    for (topic, title, unit, color), slot in zip(series, [gs[0, 0], gs[1, 0], gs[0, 2], gs[1, 2]]):
        ax = fig.add_subplot(slot)
        ts, vals = data[topic]
        vals = np.asarray(vals)
        shown = vals[(ts >= args.start) & (ts <= t_end)]
        lo, hi = np.nanmin(shown), np.nanmax(shown)
        pad = 0.08 * (hi - lo or 1.0)
        ax.set_xlim(args.start, t_end)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title, fontsize=1.4 * FONT, loc='left')
        ax.set_xlabel('time from bag start (s)')
        if unit:
            ax.set_ylabel(unit)
        ax.grid(alpha=0.3)
        pick_lines.append(ax.axvline(t_mark, color='#2e7d32', lw=3, visible=False))  # shown once t passes the pick
        line, = ax.plot([], [], color=color, lw=2.2)
        keep = ts >= args.start
        lines.append((line, ts[keep], vals[keep]))
        cursors.append(ax.axvline(0, color='#888', lw=1, ls='--'))

    ffmpeg = subprocess.Popen(
        ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{W}x{H}', '-r', str(FPS),
         '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-movflags', '+faststart', out_path],
        stdin=subprocess.PIPE)
    t0, t1 = args.start, max(t_end, cam_t[-1])
    n_frames = int((t1 - t0) / args.speed * FPS) + 1
    hold = FPS  # hold the last frame for a second
    for k in range(n_frames + hold):
        t = min(t0 + k * args.speed / FPS, t1)
        i = max(np.searchsorted(cam_t, t, side='right') - 1, 0)
        im.set_data(cam_imgs[i])
        for (line, ts, vals), cursor in zip(lines, cursors):
            j = np.searchsorted(ts, t, side='right')
            line.set_data(ts[:j], vals[:j])
            cursor.set_xdata([t, t])
        clock.set_text(f't = {t:6.2f} s')
        clock.set_color('#2e7d32' if t >= t_mark else 'black')
        for pick_line in pick_lines:
            pick_line.set_visible(t >= t_mark)
        fig.canvas.draw()
        ffmpeg.stdin.write(fig.canvas.buffer_rgba())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(out_path)


if __name__ == '__main__':
    main()
