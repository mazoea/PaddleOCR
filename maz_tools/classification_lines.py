import os
import sys
import json
import argparse
import numpy as np
import cv2
import logging

from coords import bbox

_logger = logging.getLogger("classification_lines")

def classify_page(shaded_lines: list, lines: dict, page_h, page_w) -> bool:
    """
        Classify page as containing zebra lines if there are enough shaded lines,
        and they are well distributed across the page height.
        1) if more ten 30% of all lines are shaded lines, then classify as zebra lines
        2) if there is minimum 5times pattern zebra, classic, zebra lines in situated
        in 3/5 of the page height classify as zebra lines
    """

    # summary of lines which are wider than half of page width
    all_lines = [bbox.union(*line_bbs) for line_bbs in lines.values()]
    wide_lines = [line_bb for line_bb in all_lines if line_bb.width > page_w * 0.5]
    if len(wide_lines) == 0:
        return False
    shaded_ratio = len(shaded_lines) / len(wide_lines)
    if 0.3 < shaded_ratio:
        return True

    wide_lines.sort(key=lambda bb: bb.mid_y())
    start_test = page_h * 0.2
    end_test = page_h * 0.8
    wide_lines_in_region = [line_bb for line_bb in wide_lines if start_test < line_bb.mid_y() < end_test]
    # check for pattern zebra, classic, zebra lines in 3/5 of the page height
    shaded_lines_sorted = sorted(shaded_lines, key=lambda bb: bb.mid_y())
    pattern_count = 0
    last_shaded_mid_y = -1
    for shaded_bb in shaded_lines_sorted:
        shaded_mid_y = shaded_bb.mid_y()
        if last_shaded_mid_y < 0:
            last_shaded_mid_y = shaded_mid_y
            pattern_count = 1
            continue
        # find next classic line between last and current shaded line
        found_classic = 0
        for line_bb in wide_lines_in_region:
            line_mid_y = line_bb.mid_y()
            if last_shaded_mid_y < line_mid_y < shaded_mid_y:
                if line_bb not in shaded_lines:
                    found_classic += 1
        if 0 == found_classic:
            last_shaded_mid_y = shaded_mid_y

        if found_classic < 3:
            pattern_count += 1  # we found zebra-classic pattern
        last_shaded_mid_y = shaded_mid_y
    # each pattern is 2 shaded lines and 1 classic line, so we need at least 2 patterns to classify
    if 5 <= pattern_count:
            return True

    return False


def analyze_gap_region(gap_roi: np.ndarray, debug):
    if gap_roi.size == 0:
        return False, False

    # Threshold to binary
    if debug:
        cv2.imwrite("d:/tmp/gap_roi.png", gap_roi)
    #_, binary = cv2.threshold(gap_roi, 240, 255, cv2.THRESH_BINARY_INV)
    _, binary = cv2.threshold(gap_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if debug:
        cv2.imwrite("d:/tmp/gap_roi.png", 255-binary)

    # Calculate the percentage of dark pixels
    dark_pixels = np.sum(binary == 255)
    total_pixels = binary.size
    dark_ratio = dark_pixels / total_pixels

    # Heuristic thresholds
    frag_threshold = 0.1  # 10% dark pixels indicates fragment

    is_fragment = dark_ratio > frag_threshold
    if is_fragment:
        dilated_img = cv2.dilate(binary,  np.ones((3, 3), np.uint8), iterations=1)
        kernel = np.ones((3,3), np.uint8)
        dilated_img = cv2.morphologyEx(dilated_img, cv2.MORPH_CLOSE, kernel)
        if debug:
            cv2.imwrite("d:/tmp/gap_roi.png", 255-dilated_img)

        # try to detect white rectangles, because if there are in, there will be no fragment nor noise
        contours, _ = cv2.findContours(255-dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if area < 0.25 * total_pixels:
                continue
            # if high is higher then half of gap_roi height, consider it
            if h < gap_roi.shape[0] * 0.5:
                continue
            # Found a white rectangle
            is_fragment = False
            break

    # try to detect noise in the way, that we dilate and close the binary image, then we look again for countours
    is_noise = False
    #if not is_fragment:
    if dark_ratio < frag_threshold:
        # by smart to establish kernel according to size of gap_roi
        h, w = binary.shape[:2]
        if h < w:
            h = min(max(3, h // 3), 30)
            w = min(max(3, (w // 7)), 30)
        else:
            w = min(max(3, w // 3), 30)
            h = min(max(3, (h // 7)), 30)
        dilated_img = cv2.dilate(binary,  np.ones((h, w), np.uint8), iterations=1)
        kernel = np.ones((3,3), np.uint8)
        dilated_img = cv2.morphologyEx(dilated_img, cv2.MORPH_CLOSE, kernel)
        if debug:
            cv2.imwrite("d:/tmp/gap_roi.png", 255-dilated_img)

        # count dark pixels again
        dark_pixels = np.sum(dilated_img == 255)
        dark_ratio = dark_pixels / total_pixels
        noise_dark_ratio = 0.2
        if noise_dark_ratio < dark_ratio:
            is_noise = True
            # check contours
            contours, _ = cv2.findContours(255-dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            white_area = sum([cv2.contourArea(cnt) for cnt in contours])
            if 0.10 * total_pixels < white_area:
                for cnt in contours:
                    #area = cv2.contourArea(cnt)
                    x, y, w, h = cv2.boundingRect(cnt)
                    #if area < 0.10 * total_pixels:
                    #    continue
                    # if high is higher then half of gap_roi height, consider it
                    if h < gap_roi.shape[0] * 0.4:
                        continue
                    # Found a white rectangle
                    is_noise = False
                    break

    return is_fragment, is_noise

def process(input_path: str, input_bbs: str, debug: bool = False):

    res = {"input_path": input_path, "zebra_doc": False, "found_zebra_lines": 0, "total_lines": 0}
    # Load image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        _logger.error(f"Error: Unable to load image '{input_path}'")
        return

    # Load bounding boxes
    with open(input_bbs, 'r') as f:
        bbs_data = json.load(f)

    bbs = [bbox.default_bbox(d) for d in bbs_data["bboxes"]]

    # scale bbs to image size
    h_img, w_img = img.shape[:2]
    bbs_h = bbs_data["img_h"]
    bbs_w = bbs_data["img_w"]
    scale_x = w_img / bbs_w
    scale_y = h_img / bbs_h
    bbs = [bb.scale(scale_x) for bb in bbs]

    # aprox lines from bbs according their mid_y and height
    avg_h = np.average([bb.height for bb in bbs])
    lines = {}
    for bb in bbs:
        mid_y = bb.mid_y()
        found_line = False
        for line_key in lines:
            line_bbs = lines[line_key]
            line_mid_y = np.average([b.mid_y() for b in line_bbs])
            if abs(mid_y - line_mid_y) < avg_h * 0.5:
                lines[line_key].append(bb)
                found_line = True
                break
        if not found_line:
            new_line_key = len(lines)
            lines[new_line_key] = [bb]

    # For each line, compute combined bbox and visualize it
    #lines_bbs = [bbox.union(*line_bbs) for line_bbs in lines.values()]
    #for i, line_bb in enumerate(lines_bbs):
    #    x1, y1, x2, y2 = int(line_bb.xl), int(line_bb.yt), int(line_bb.xr), int(line_bb.yb)
    #    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #    cv2.putText(img, f"Line {i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    #output_viz_path = f"{input_path}_lines_viz.png"
    #cv2.imwrite(output_viz_path, img)

    # analyze each line where are more words, we are interesting in the areas between the words
    # if this areas are dark fragments or noise
    shaded_lines = []
    for i, line_bbs in enumerate(lines.values()):
        # check if there are at least 2 bbs to analyze gaps
        if len(line_bbs) < 2:
            continue
        h = bbox.union(*line_bbs).height
        l_bb = bbox.union(*line_bbs)
        x1_cut, y1_cut, x2_cut, y2_cut = l_bb.xl, l_bb.yt, l_bb.xr, l_bb.yb
        line_img = img[int(y1_cut):int(y2_cut), int(x1_cut):int(x2_cut)]
        if debug:
            cv2.imwrite("d:/tmp/gap_roi.png", line_img)

        # sort bbs by xl
        line_bbs = sorted(line_bbs, key=lambda bb: bb.xl)
        # analyze gaps between bbs
        clean_gaps_w = 0
        dirty_gaps_w = 0
        for j in range(len(line_bbs) - 1):
            bb1 = line_bbs[j]
            bb2 = line_bbs[j + 1]
            gap_x1 = int(bb1.xr)
            gap_x2 = int(bb2.xl)
            gap_y1 = int(min(bb1.yt, bb2.yt))
            gap_y2 = int(max(bb1.yb, bb2.yb))
            diff_x = gap_x2 - gap_x1
            tol_x = h* 0.5
            if gap_x2 <= gap_x1 or diff_x <= tol_x:
                continue  # no gap
            gap_roi = img[gap_y1:gap_y2, gap_x1:gap_x2]
            is_frag, is_noise = analyze_gap_region(gap_roi, debug)
            # get width of gap
            gap_width = gap_x2 - gap_x1
            if is_frag or is_noise:
                dirty_gaps_w += gap_width
            else:
                clean_gaps_w += gap_width
        # calcul ratio
        total_gaps_w = clean_gaps_w + dirty_gaps_w
        if 0 < total_gaps_w:
            dirty_ratio = dirty_gaps_w / total_gaps_w
            if 0.5 < dirty_ratio:
                if debug:
                    _logger.debug(f"Line {i} classified as shaded line (dirty ratio: {dirty_ratio:.2f})")
                shaded_lines.append(bbox.union(*line_bbs))

    if debug:
        for i, line_bb in enumerate(shaded_lines):
            x1, y1, x2, y2 = int(line_bb.xl), int(line_bb.yt), int(line_bb.xr), int(line_bb.yb)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Line {i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        output_viz_path = f"{input_path}_lines_viz_after.png"
        cv2.imwrite(output_viz_path, img)

    # lets classify page
    is_ = classify_page(shaded_lines, lines, h_img, w_img)
    res["zebra_doc"] = is_
    res["found_zebra_lines"] = len(shaded_lines)
    res["total_lines"] = len(lines)

    for k, v in res.items():
        _logger.critical(f"{k}: {v}")



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Analyze image for zebra lines.")
    arg_parser.add_argument("--input_path", type=str, help="Path to the input image and bbs is expected near by with _detection.png.bbs.json.")
    arg_parser.add_argument("--debug", type=bool, default=False, help="Enable debug mode.", required=False)

    args = arg_parser.parse_args()

    input_path = args.input_path
    debug = args.debug

    input_bbs = f"{input_path}_detection.png.bbs.json"

    # Check if input file exists
    if not os.path.exists(input_path) and not os.path.exists(input_bbs):
        _logger.error(f"Error: Input files '{input_path}' does not exist!")
        sys.exit(1)

    process(input_path, input_bbs, debug)
