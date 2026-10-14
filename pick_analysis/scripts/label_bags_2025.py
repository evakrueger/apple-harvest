#!/usr/bin/env python3
"""
Bag labeler for last year's (2025) Prosser pick bags: same window, buttons and shortcuts as label_bags.py, adapted
to how those bags were recorded.

  python3 label_bags_2025.py [--roots /mnt/data/eva_rosbags/prosser_2025] [--dates 2025-10-28 ...]
                             [--out ../labels/PICK_TIMES_2025.xlsx]

Differences from label_bags.py:
  - the gripper camera (/image_raw) is in every bag and always shown on the left; the mast camera
    (/camera/mast_camera/color/image_raw) is only in some, and is shown on the right when the bag has it
  - the bags sit in day folders (e.g. "Day 1 Tuesday/eva_day1/") with no batch_*/apple_* folders or
    pick_metadata.yaml, so the spreadsheet has the folder instead of batch, apple and pull type
  - every final_approach_and_pick_* bag under --roots is labeled, unless --dates is given

Times are seconds from the bag's start, plus the absolute epoch time, as in label_bags.py.
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

from PyQt5 import QtWidgets

import label_bags as lb

LABELS = lb.PICK_ANALYSIS / 'labels'
GRIPPER_CAMERA = '/image_raw'
COLUMNS = ['Bag filename', 'Date captured', 'Folder', 'Pick/Slip', 'Pick/Slip time (s from bag start)',
           'Pick/Slip time (epoch s)', 'Bag path']


class AnyDate:
    def __contains__(self, date):
        return True


def find_bags(roots, dates):
    bags = lb.find_bags(roots, dates)
    for bag in bags:  # gripper camera first, so it stays on the left whether or not the mast camera is there
        bag.image_topics.sort(key=lambda t: (t != GRIPPER_CAMERA, t))
    return bags


class LabelStore2025(lb.LabelStore):
    def __init__(self, xlsx_path, roots):
        super().__init__(xlsx_path)
        self.roots = [os.path.abspath(r) for r in roots]

    def folder(self, bag_path):
        """The bag's folder relative to the root it was found under, e.g. 'Day 1 Tuesday/eva_day1'."""
        parent = os.path.dirname(bag_path)
        for root in self.roots:
            if parent == root or parent.startswith(root + os.sep):
                return os.path.relpath(parent, root)
        return parent

    def write_xlsx(self):
        from openpyxl import Workbook
        from openpyxl.styles import Font

        wb = Workbook()
        ws = wb.active
        ws.title = self.xlsx_path.stem[:31]  # the tab is named after the file, e.g. PICK_TIMES_2026
        ws.append(COLUMNS)
        for cell in ws[1]:
            cell.font = Font(bold=True)
        entries = sorted((e for e in self.labels.values() if e['label'] != 'na'),
                         key=lambda e: e['bag']['start_ns'])
        for e in entries:
            b = e['bag']
            ws.append([b['name'], datetime.datetime.fromtimestamp(b['start_ns'] / 1e9), self.folder(b['path']),
                       e['label'], round(e['time_s'], 3), round(e['time_ns'] / 1e9, 3), b['path']])
            ws.cell(ws.max_row, 2).number_format = 'yyyy-mm-dd hh:mm:ss'
            ws.cell(ws.max_row, 6).number_format = '0.000'
        for col, width in zip('ABCDEFG', [44, 20, 28, 10, 16, 18, 100]):
            ws.column_dimensions[col].width = width
        ws.freeze_panes = 'A2'
        ws.auto_filter.ref = ws.dimensions
        tmp = self.xlsx_path.with_name('.' + self.xlsx_path.name + '.tmp')
        wb.save(tmp)
        os.replace(tmp, self.xlsx_path)


class Labeler2025(lb.Labeler):
    def __init__(self, bags, store):
        super().__init__(bags, store)
        self.setWindowTitle('Bag labeler 2025')

    def go_to(self, i):
        super().go_to(i)
        bag = self.bags[i]
        # only show the second video when this bag has a second camera; otherwise give the whole width to the first
        two = len(bag.image_topics) > 1
        self.views[1].setVisible(two)
        self.view_titles[1].setVisible(two)
        self.layout().itemAt(1).layout().setStretch(1, 1 if two else 0)  # the videos row
        self.info.setText(
            f'<b>{bag.name}</b> &nbsp;|&nbsp; folder: <b>{self.store.folder(bag.path)}</b> &nbsp;|&nbsp; '
            f'{bag.start:%Y-%m-%d %H:%M:%S}, {bag.duration_s:.1f} s<br><small>{bag.path}</small>')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--roots', nargs='+', default=['/mnt/data/eva_rosbags/prosser_2025'],
                        help='directories searched for bags')
    parser.add_argument('--dates', nargs='+', help='only bags started on these (local) dates; default: all')
    parser.add_argument('--out', default=str(LABELS / 'PICK_TIMES_2025.xlsx'),
                        help='spreadsheet to write (progress JSON goes next to it)')
    parser.add_argument('--list', action='store_true', help='print the bags that would be labeled and exit')
    args = parser.parse_args()

    missing = [r for r in args.roots if not os.path.isdir(r)]
    if missing:
        sys.exit(f'Not found: {", ".join(missing)} (put the 2025 bags there or pass --roots)')
    dates = {datetime.date.fromisoformat(d) for d in args.dates} if args.dates else AnyDate()
    print('Searching for bags in', ', '.join(args.roots), '...', flush=True)
    bags = find_bags(args.roots, dates)
    print(f'Found {len(bags)} pick bags')
    if args.list:
        for b in bags:
            print(f'{b.start:%Y-%m-%d %H:%M:%S}  {",".join(b.image_topics) or "(no images)":14s} {b.path}')
        return
    if not bags:
        sys.exit('Nothing to label')

    app = QtWidgets.QApplication(sys.argv)
    window = Labeler2025(bags, LabelStore2025(args.out, args.roots))
    window.show()
    code = app.exec_()
    os._exit(code)  # labels are already saved; don't wait for a prefetch still reading a bag


if __name__ == '__main__':
    main()
