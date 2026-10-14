# pick_analysis

Hand-labeled pick/slip times for the Prosser pick bags (Oct 2025 and Oct 2026), and the plots made from them.
Kept apart from the harvest packages next to it: it isn't a ROS package (`COLCON_IGNORE`), so colcon skips it, and
nothing here is run by the robot. The signal cache in `plots/.cache/` is git-ignored.

```
scripts/   label_bags.py        labeling GUI for the 2026 bags (2 cameras)
           label_bags_2025.py   labeling GUI for the 2025 bags (gripper camera, sometimes the mast camera)
           plot_pick_times.py   sensor plots of every labeled bag + summary pages
           pick_video.py        video of one bag with its sensor traces
labels/    PICK_TIMES_2025.xlsx, PICK_TIMES_2026.xlsx   the labels (one row per pick/slip; N/A bags left out)
           *.progress.json      what the GUIs save and resume from, every bag including N/A; the .xlsx is made from it
plots/     PICK_TIMES_<year>_plots.pdf         one page per bag, then all picks / all slips overlaid
           PICK_TIMES_2025_2026_summary.pdf    picks and slips of both years overlaid, by year and in one color
           .cache/                             each bag's signals, so replotting doesn't re-read the bags
videos/    output of pick_video.py
```

## Labeling

```bash
python3 scripts/label_bags.py        # bags on /mnt/data from Oct 6/7 2026
python3 scripts/label_bags_2025.py   # bags in /mnt/data/eva_rosbags/prosser_2025
```

Mark pick `P`, mark slip `S`, restart videos `R`, next bag `N`, N/A `X`, back `B`; `Space` plays/pauses, `←`/`→` step a
frame. One event per bag. Each bag is saved when you press Next or N/A, and the GUI reopens at the first unlabeled bag.
`--list` prints the bags it found.

Times are seconds from the bag's recorded start (receive timestamps). The 2025 times from the old live plotter used a
different zero (first `/position_*` or `/tof_sensor_data` message), so they are not directly comparable to these.

## Plotting

```bash
python3 scripts/plot_pick_times.py                  # per-year PDFs + combined summary
python3 scripts/plot_pick_times.py --summary-only   # just the combined summary
python3 scripts/pick_video.py <bag path>            # 2026 bags; writes videos/<bag>_pick.mp4
```

Plots: vacuum pressure, |F| (force magnitude from the F/T sensor, not tared), d|F|/dt (after a 50 ms moving average),
the flex sensor channels, and ToF. Summary pages shift each curve so its marked pick/slip is at t = 0.
