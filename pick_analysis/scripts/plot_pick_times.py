#!/usr/bin/env python3
"""
Plot the sensor data of every labeled pick bag, one PDF per year.

  python3 plot_pick_times.py [--years 2025 2026] [--labels ../labels] [--out ../plots] [--summary-only]

Reads <labels>/PICK_TIMES_<year>.progress.json (written by label_bags.py / label_bags_2025.py) and writes
<out>/PICK_TIMES_<year>_plots.pdf:
  - one page per pick/slip bag (N/A bags are left out), in the order they were recorded, titled with the bag
    filename: vacuum pressure, force magnitude |F|, its derivative d|F|/dt, the flex sensors and ToF, against
    seconds from bag start, with the marked pick/slip time drawn as a vertical line
  - two summary pages: every pick overlaid, then every slip overlaid, each curve shifted so its marked time is t = 0
and <out>/PICK_TIMES_<years>_summary.pdf: the same two summary pages with all years together, colored by year, then
the same two again with every curve in one color (picks green, slips red).

  --summary-only   only write the combined summary PDF

Each bag's signals are cached in <out>/.cache/ (refreshed if the bag changes), so reruns don't re-read the bags.

|F| = sqrt(Fx^2 + Fy^2 + Fz^2) from /force_torque_sensor_broadcaster/wrench, as recorded (no tare). d|F|/dt is taken
after resampling |F| to 500 Hz and smoothing it with a 50 ms moving average, since the raw signal is too noisy to
differentiate.
"""

import argparse
import datetime
import glob
import json
import os
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from rclpy.serialization import deserialize_message
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float32, Float32MultiArray, Int32

TOPICS = {
    'pressure': ('/vacuum_pressure', Float32),
    'force': ('/force_torque_sensor_broadcaster/wrench', WrenchStamped),
    'flex': ('/flex_sensor_data', Float32MultiArray),
    'tof': ('/tof_sensor_data', Int32),
}
PICK_ANALYSIS = Path(__file__).resolve().parent.parent  # pick_analysis/: scripts/, labels/, plots/, videos/
DERIV_RATE = 500.0  # Hz, |F| is resampled to this before differentiating
DERIV_SMOOTH = 0.05  # s, moving-average window before differentiating

# (key, y label) for the five plots, top to bottom
PANELS = [('pressure', 'Vacuum pressure'), ('force', '|F| (N)'), ('dforce', 'd|F|/dt (N/s)'),
          ('flex', 'Flex sensors'), ('tof', 'ToF (mm)')]
EVENT_COLORS = {'pick': 'tab:green', 'slip': 'tab:red'}
YEAR_COLORS = ['tab:blue', 'tab:orange', 'tab:purple', 'tab:brown']
FLEX_COLORS = ['tab:blue', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']


def read_signals(bag_path, start_ns):
    """{key: (t in s from bag start, values)}; flex values are (n, channels), padded with NaN."""
    raw = {key: ([], []) for key in TOPICS}
    for db in sorted(glob.glob(os.path.join(bag_path, '*.db3'))):
        con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
        ids = dict(con.execute('select name, id from topics'))
        for key, (topic, msg_type) in TOPICS.items():
            if topic not in ids:
                continue
            for ts, data in con.execute('select timestamp, data from messages where topic_id = ? order by timestamp',
                                        (ids[topic],)):
                msg = deserialize_message(data, msg_type)
                if key == 'force':
                    f = msg.wrench.force
                    value = (f.x * f.x + f.y * f.y + f.z * f.z) ** 0.5
                elif key == 'flex':
                    value = list(msg.data)
                else:
                    value = msg.data
                raw[key][0].append((ts - start_ns) / 1e9)
                raw[key][1].append(value)
        con.close()

    signals = {}
    for key, (t, values) in raw.items():
        if not t:
            continue
        if key == 'flex':
            width = max(len(v) for v in values)
            values = [v + [np.nan] * (width - len(v)) for v in values]
        signals[key] = (np.asarray(t), np.asarray(values, dtype=float))
    if 'force' in signals:
        signals['dforce'] = derivative(*signals['force'])
    return signals


def load_signals(entry, cache_dir):
    """read_signals, cached per bag as .npz; the cache is redone if any of the bag's files is newer."""
    b = entry['bag']
    cache = Path(cache_dir) / (b['name'] + '.npz')
    bag_mtime = max((os.path.getmtime(f) for f in glob.glob(os.path.join(b['path'], '*'))), default=0)
    if cache.exists() and cache.stat().st_mtime >= bag_mtime:
        with np.load(cache) as z:
            keys = {k[:-2] for k in z.files}
            return {k: (z[k + '_t'], z[k + '_y']) for k in keys}
    signals = read_signals(b['path'], b['start_ns'])
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.stem + '.tmp.npz')
    np.savez(tmp, **{f'{k}_{part}': arr for k, (t, y) in signals.items() for part, arr in (('t', t), ('y', y))})
    os.replace(tmp, cache)
    return signals


def derivative(t, y):
    if len(t) < 10 or t[-1] - t[0] < 4 * DERIV_SMOOTH:
        return np.array([]), np.array([])
    tu = np.arange(t[0], t[-1], 1.0 / DERIV_RATE)
    n = max(1, int(round(DERIV_SMOOTH * DERIV_RATE)))
    smooth = np.convolve(np.interp(tu, t, y), np.ones(n) / n, mode='same')
    d = np.gradient(smooth, tu)
    return tu[n:-n], d[n:-n]  # the moving average is biased where its window runs off the ends


def plot_signal(ax, key, t, y, flex_by_channel=True, **kw):
    if key == 'flex':
        if flex_by_channel:
            kw.pop('color', None)
        for ch in range(y.shape[1]):
            ax.plot(t, y[:, ch], **({'color': FLEX_COLORS[ch % len(FLEX_COLORS)]} if flex_by_channel else {}), **kw)
    else:
        ax.plot(t, y, color=kw.pop('color', 'tab:blue'), **kw)


def new_page(title, subtitle):
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(8.5, 11), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.985)
    fig.text(0.5, 0.955, subtitle, ha='center', fontsize=9, color='0.3')
    for ax, (_, label) in zip(axes, PANELS):
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.935, bottom=0.05, hspace=0.12)
    return fig, axes


def flex_legend(ax, n_channels):
    handles = [plt.Line2D([], [], color=FLEX_COLORS[i % len(FLEX_COLORS)], label=f'flex {i}')
               for i in range(n_channels)]
    ax.legend(handles=handles, fontsize=7, ncol=n_channels, loc='upper left')


def bag_page(pdf, entry, signals):
    b = entry['bag']
    label, t_event = entry['label'], entry['time_s']
    details = [f'{label.upper()} at {t_event:.2f} s', f'{datetime.datetime.fromtimestamp(b["start_ns"] / 1e9):%Y-%m-%d %H:%M:%S}']
    if b.get('batch') is not None:
        details.append(f'batch {b["batch"]}, apple {b["apple"]}')
    if b.get('pull_type'):
        details.append(b['pull_type'])
    fig, axes = new_page(b['name'], '   |   '.join(details))
    for ax, (key, _) in zip(axes, PANELS):
        if key in signals and len(signals[key][0]):
            plot_signal(ax, key, *signals[key], linewidth=0.8)
            if key == 'flex':
                flex_legend(ax, signals[key][1].shape[1])
        else:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center', va='center', color='0.5')
        ax.axvline(t_event, color=EVENT_COLORS[label], linewidth=1.2, linestyle='--')
    axes[-1].set_xlabel('Time from bag start (s)', fontsize=9)
    axes[0].legend(handles=[plt.Line2D([], [], color=EVENT_COLORS[label], linestyle='--', label=label)],
                   fontsize=7, loc='upper right')
    pdf.savefig(fig)
    plt.close(fig)


def summary_page(pdf, title, label, groups, flex_by_channel=True):
    """groups: [(name, color, [(entry, signals)])] for one event type, overlaid with each curve shifted so its
    marked time is t = 0. With one group flex is colored by channel; with several, everything by group."""
    fig, axes = new_page(title, f'every {label} overlaid, shifted so the marked {label} time is t = 0')
    if not any(items for _, _, items in groups):
        for ax in axes:
            ax.text(0.5, 0.5, f'no {label}s', transform=ax.transAxes, ha='center', va='center', color='0.5')
    n_flex = 0
    # biggest group first, so a smaller one isn't buried under it
    groups = sorted(groups, key=lambda g: -len(g[2]))
    for ax, (key, _) in zip(axes, PANELS):
        for _, color, items in groups:
            alpha = float(np.clip(10 / max(len(items), 1), 0.12, 0.8))  # fainter the more curves overlap
            for entry, signals in items:
                if key in signals and len(signals[key][0]):
                    t, y = signals[key]
                    plot_signal(ax, key, t - entry['time_s'], y, flex_by_channel=flex_by_channel, color=color,
                                linewidth=0.7, alpha=alpha)
                    if key == 'flex':
                        n_flex = max(n_flex, y.shape[1])
        ax.axvline(0, color='k', linewidth=1, linestyle='--')
    if n_flex and flex_by_channel:
        flex_legend(axes[3], n_flex)
    if len(groups) > 1:
        axes[0].legend(handles=[plt.Line2D([], [], color=color, linewidth=2, label=f'{name} (n = {len(items)})')
                                for name, color, items in sorted(groups, key=lambda g: g[0])],
                       fontsize=8, loc='lower left')
    axes[-1].set_xlabel(f'Time from {label} (s)', fontsize=9)
    save_as_image(pdf, fig)


def save_as_image(pdf, fig, dpi=200):
    """Puts the page in the PDF as one picture. Hundreds of overlaid curves would make a huge, slow vector page,
    and matplotlib's per-artist rasterizing came out as broken dotted lines in PDF viewers."""
    fig.set_dpi(dpi)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    page = plt.figure(figsize=fig.get_size_inches())
    plt.close(fig)
    ax = page.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(img, aspect='auto', interpolation='antialiased')
    pdf.savefig(page)
    plt.close(page)


def labeled_entries(year, labels_dir):
    """The pick/slip labels for one year, in recording order."""
    progress = Path(labels_dir) / f'PICK_TIMES_{year}.progress.json'
    return sorted((e for e in json.loads(progress.read_text()).values() if e['label'] in EVENT_COLORS),
                  key=lambda e: e['bag']['start_ns'])


def make_pdf(year, labels_dir, out_dir):
    out = Path(out_dir) / f'PICK_TIMES_{year}_plots.pdf'
    entries = labeled_entries(year, labels_dir)
    by_label = {label: [] for label in EVENT_COLORS}
    with PdfPages(out) as pdf:
        for i, entry in enumerate(entries, 1):
            signals = load_signals(entry, Path(out_dir) / '.cache')
            bag_page(pdf, entry, signals)
            by_label[entry['label']].append((entry, signals))
            print(f'\r{year}: {i}/{len(entries)} {entry["bag"]["name"]}', end='', flush=True)
        for label, items in by_label.items():
            summary_page(pdf, f'{year}: all {label}s (n = {len(items)})', label,
                         [(year, EVENT_COLORS[label], items)])
        pdf.infodict()['Title'] = f'PICK_TIMES_{year} plots'
    print(f'\n{year}: {len(entries)} bag pages + 2 summary pages ->', out)


def make_combined_summary(years, labels_dir, out_dir):
    out = Path(out_dir) / f'PICK_TIMES_{"_".join(years)}_summary.pdf'
    loaded = {}
    for year in years:
        entries = labeled_entries(year, labels_dir)
        loaded[year] = []
        for i, entry in enumerate(entries, 1):
            loaded[year].append((entry, load_signals(entry, Path(out_dir) / '.cache')))
            print(f'\rsummary: {year} {i}/{len(entries)}', end='', flush=True)
    with PdfPages(out) as pdf:
        for label in EVENT_COLORS:
            groups = [(year, YEAR_COLORS[i % len(YEAR_COLORS)], [x for x in loaded[year] if x[0]['label'] == label])
                      for i, year in enumerate(years)]
            n = sum(len(items) for _, _, items in groups)
            summary_page(pdf, f'{" and ".join(years)}: all {label}s (n = {n})', label, groups, flex_by_channel=False)
        for label in EVENT_COLORS:  # the same, years not told apart
            items = [x for year in years for x in loaded[year] if x[0]['label'] == label]
            summary_page(pdf, f'{" and ".join(years)}: all {label}s (n = {len(items)})', label,
                         [(' and '.join(years), EVENT_COLORS[label], items)], flex_by_channel=False)
        pdf.infodict()['Title'] = f'PICK_TIMES {" and ".join(years)} summary'
    print('\nsummary: picks + slips by year, then picks + slips in one color ->', out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--years', nargs='+', default=['2025', '2026'])
    parser.add_argument('--labels', default=str(PICK_ANALYSIS / 'labels'),
                        help='folder with PICK_TIMES_<year>.progress.json')
    parser.add_argument('--out', default=str(PICK_ANALYSIS / 'plots'), help='folder the PDFs are written to')
    parser.add_argument('--summary-only', action='store_true', help='only write the combined summary PDF')
    args = parser.parse_args()
    if not args.summary_only:
        for year in args.years:
            make_pdf(year, args.labels, args.out)
    make_combined_summary(args.years, args.labels, args.out)


if __name__ == '__main__':
    main()
