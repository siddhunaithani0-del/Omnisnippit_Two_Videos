import os
import cv2
import numpy as np
import imageio.v2 as imageio
import streamlit as st
from moviepy.editor import (
    VideoFileClip,
    concatenate_videoclips,
    AudioFileClip,
)
import moviepy.video.fx.all as vfx
from ultralytics import YOLO
from streamlit_sortables import sort_items  # for drag & drop reordering

PREVIEW_DIR = "previews"
os.makedirs(PREVIEW_DIR, exist_ok=True)

# Do NOT trim shots shorter than this
MIN_PROTECT_SECONDS = 0.7

# Target aspect for vertical reels
TARGET_ASPECT = 9 / 16

# YOLO model
yolo = YOLO("yolov8n.pt")

st.set_page_config(layout="wide")
# ---------------- FULL-WIDTH STREAMLIT TOP BAR OVERRIDE ----------------
st.markdown(
    """
    <style>
    /* Override default Streamlit header */
    header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #ff0077, #ff8800) !important;
        height: 60px !important;
    }

    /* Hide default Streamlit logo area */
    header[data-testid="stHeader"]::before {
        content: "OMNISNIPPET";
        position: absolute;
        left: 24px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 20px;
        font-weight: 800;
        letter-spacing: 2px;
        color: white;
    }

    /* Push app content below colored header */
    .block-container {
        padding-top: 90px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎬 OMNISNIPPET – 2 Video Smart Mixer")


# ==========================================================
#        COMPOSITION ANALYSIS (PERSON + PRODUCT)
# ==========================================================

def sample_frames(video_path, start, end, num_samples=4):
    """Sample a few frames evenly from a shot using OpenCV only."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0

    if fps <= 0 or total_frames <= 0:
        cap.release()
        return []

    duration = total_frames / fps

    # clamp start/end to video duration
    s = max(0.0, min(start, duration))
    e = max(0.0, min(end, duration))

    if e <= s:
        cap.release()
        return []

    # avoid sampling exactly the very last frame
    frame_margin = 1.0 / fps
    e = max(s + frame_margin, e - 1e-3)

    times = np.linspace(s, e, num_samples)
    frames = []

    for t in times:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return frames


def detect_person_product(frame_rgb):
    """
    Detect person, laptop and OTHER objects in a frame using YOLO.

    Priority:
        1) person
        2) laptop
        3) other objects (treated as product / toy / focus object)
    """
    H, W, _ = frame_rgb.shape
    res = yolo.predict(frame_rgb, verbose=False)[0]

    persons, laptops, products = [], [], []
    for box in res.boxes:
        cls_id = int(box.cls[0])
        label = yolo.names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        if label == "person":
            persons.append((x1, y1, x2, y2))
        elif label == "laptop":
            laptops.append((x1, y1, x2, y2))
        else:
            products.append((x1, y1, x2, y2))

    return persons, laptops, products, (W, H)


def biggest_box(boxes):
    """Return the largest bounding box."""
    if not boxes:
        return None
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    return boxes[int(np.argmax(areas))]


def union_boxes(boxes):
    """Return union of multiple boxes."""
    if not boxes:
        return None
    xs1 = [b[0] for b in boxes]
    ys1 = [b[1] for b in boxes]
    xs2 = [b[2] for b in boxes]
    ys2 = [b[3] for b in boxes]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def compute_vertical_crop(W, H, key_box):
    """Compute vertical 9:16 crop around key region (person + product)."""
    target_w = int(H * TARGET_ASPECT)

    # Already vertical/narrow → no horizontal crop
    if target_w >= W:
        return [0, 0, W, H]

    if key_box is None:
        x_center = W // 2
    else:
        x1k, y1k, x2k, y2k = key_box
        x_center = (x1k + x2k) / 2

    x1 = int(x_center - target_w / 2)
    x2 = x1 + target_w

    # Clamp inside image
    if x1 < 0:
        x1 = 0
        x2 = target_w
    if x2 > W:
        x2 = W
        x1 = W - target_w

    return [x1, 0, x2, H]


def composition_ok(crop_box, person_box):
    """
    Framing rules (only when a PERSON is present):
    - Person fully inside crop
    - Person height between 35–80% of frame height
    - Head near upper third of frame
    """
    if person_box is None or crop_box is None:
        return False

    cx1, cy1, cx2, cy2 = crop_box
    px1, py1, px2, py2 = person_box
    crop_h = cy2 - cy1
    crop_w = cx2 - cx1

    # Transform person box into crop coordinates
    px1c = px1 - cx1
    px2c = px2 - cx1
    py1c = py1 - cy1
    py2c = py2 - cy1

    # Person must be fully inside crop
    if px1c < 0 or px2c > crop_w or py1c < 0 or py2c > crop_h:
        return False

    person_h = py2c - py1c
    scale = person_h / crop_h

    if scale < 0.35 or scale > 0.8:
        return False

    head_y = py1c
    if not (0.1 * crop_h <= head_y <= 0.45 * crop_h):
        return False

    return True


def analyze_shot(video_path, start, end):
    """
    Analyze one shot of one video.
    Priority target:
        1) person
        2) laptop
        3) other objects (product / toy / prop)
    """
    frames = sample_frames(video_path, start, end, num_samples=4)
    if not frames:
        return None, False

    all_persons = []
    all_laptops = []
    all_products = []
    W = H = None

    for f in frames:
        persons, laptops, products, (W, H) = detect_person_product(f)
        all_persons.extend(persons)
        all_laptops.extend(laptops)
        all_products.extend(products)

    if W is None or H is None:
        return None, False

    main_person = biggest_box(all_persons)
    main_laptop = biggest_box(all_laptops)
    main_product = biggest_box(all_products)

    if main_person is not None:
        key_boxes = [main_person]
        if main_laptop is not None:
            key_boxes.append(main_laptop)
        key_boxes.extend(all_products)
    elif main_laptop is not None:
        key_boxes = [main_laptop]
        if main_product is not None:
            key_boxes.append(main_product)
    elif main_product is not None:
        key_boxes = [main_product]
    else:
        crop_box = compute_vertical_crop(W, H, None)
        return crop_box, False

    key_region = union_boxes(key_boxes)
    crop_box = compute_vertical_crop(W, H, key_region)

    is_good = composition_ok(crop_box, main_person) if main_person is not None else False
    return crop_box, is_good


def apply_crop_to_clip(clip, crop_box):
    """Apply spatial crop to a MoviePy clip (no resize here)."""
    if crop_box is None:
        return clip
    x1, y1, x2, y2 = crop_box
    return vfx.crop(clip, x1=x1, y1=y1, x2=x2, y2=y2)


def compute_square_crop_from_box(W, H, crop_box):
    """
    1:1 crop using same YOLO-driven center.
    """
    side = int(min(W, H))

    if crop_box is None:
        x1 = (W - side) // 2
        y1 = (H - side) // 2
        return [x1, y1, x1 + side, y1 + side]

    cx1, cy1, cx2, cy2 = crop_box
    x_center = (cx1 + cx2) / 2.0

    x1 = int(x_center - side / 2.0)
    if x1 < 0:
        x1 = 0
    if x1 + side > W:
        x1 = W - side

    y1 = (H - side) // 2
    return [x1, y1, x1 + side, y1 + side]


def compute_16_9_center_crop(W, H):
    """
    Center crop to 16:9 for YouTube style export.
    """
    if W <= 0 or H <= 0:
        return [0, 0, W, H]

    target_aspect = 16.0 / 9.0
    current_aspect = W / H

    # Already 16:9-ish
    if abs(current_aspect - target_aspect) < 1e-3:
        return [0, 0, W, H]

    # Too wide -> crop width
    if current_aspect > target_aspect:
        new_w = int(H * target_aspect)
        x1 = (W - new_w) // 2
        return [x1, 0, x1 + new_w, H]
    # Too tall -> crop height
    else:
        new_h = int(W / target_aspect)
        y1 = (H - new_h) // 2
        return [0, y1, W, y1 + new_h]


# --------------------------------------------------------
# SHOT POST-PROCESSING – MERGE ONLY SHORT SHOTS
# --------------------------------------------------------
def merge_short_shots(shots, min_len_sec=0.5):
    if not shots:
        return []

    merged = [[shots[0][0], shots[0][1]]]

    for s, e in shots[1:]:
        dur = e - s
        if dur < min_len_sec:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    return [(s, e) for s, e in merged]


# --------------------------------------------------------
# TRUE SHOT DETECTION (HARD CUTS)
# --------------------------------------------------------
def detect_shots(video_path, threshold=25.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    shots = []
    prev_gray = None
    shot_start = 0.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            score = np.mean(diff)

            if score > threshold:
                t = frame_idx / fps
                shots.append((shot_start, t))
                shot_start = t

        prev_gray = gray
        frame_idx += 1

    duration = frame_idx / fps
    shots.append((shot_start, duration))
    cap.release()

    shots = merge_short_shots(shots, min_len_sec=0.5)
    return shots


# --------------------------------------------------------
# PREVIEW & THUMB GENERATION – PER VIDEO
# --------------------------------------------------------
def generate_previews(video_path, shot_meta, prefix):
    """
    shot_meta: list of dicts with keys:
        start, end, crop, is_good, video_path
    prefix: string to distinguish video1/video2 files

    Returns:
        preview_paths: mp4 previews
        thumb_paths:   jpg thumbnails (mid-frame)
    """
    base = VideoFileClip(video_path)
    preview_paths = []
    thumb_paths = []

    for i, shot in enumerate(shot_meta):
        s = float(shot["start"])
        e = float(shot["end"])
        out_path = os.path.join(PREVIEW_DIR, f"{prefix}_shot_{i:03d}.mp4")
        thumb_path = os.path.join(PREVIEW_DIR, f"{prefix}_shot_{i:03d}.jpg")

        if os.path.exists(out_path):
            os.remove(out_path)
        if os.path.exists(thumb_path):
            os.remove(thumb_path)

        s = max(0.0, s)
        e = min(e, base.duration)
        if e <= s:
            continue

        # smaller height so selection section is compact
        clip = base.subclip(s, e).without_audio().resize(height=320)

        clip.write_videofile(
            out_path,
            codec="libx264",
            audio=False,
            verbose=False,
            logger=None,
        )
        clip.close()

        # thumbnail from mid-frame
        mid_t = (s + e) / 2.0
        frame = base.get_frame(mid_t)  # RGB
        imageio.imwrite(thumb_path, frame)

        preview_paths.append(out_path)
        thumb_paths.append(thumb_path)

    base.close()
    return preview_paths, thumb_paths


# --------------------------------------------------------
# FINAL RENDER – MIX 2 VIDEOS WITH SINGLE AUDIO SOURCE
# --------------------------------------------------------
def render_final(
    selected_shots,
    outname,
    target_duration,
    vertical_9_16=False,
    square_1_1=False,
    youtube_16_9=False,
    audio_source="video1",  # "video1", "video2", "external"
    audio_path=None,
    video1_path=None,
    video2_path=None,
):
    """
    selected_shots: list of dicts with keys:
        video_path, start, end, crop, is_good
    """
    if not selected_shots:
        st.error("No shots selected.")
        return

    # open each source video once
    video_paths = sorted(list({sh["video_path"] for sh in selected_shots}))
    clips_by_path = {vp: VideoFileClip(vp) for vp in video_paths}

    # clamp shots
    valid_shots = []
    for sh in selected_shots:
        vp = sh["video_path"]
        base = clips_by_path[vp]
        s = max(0.0, float(sh["start"]))
        e = min(float(sh["end"]), base.duration)
        if e > s:
            new_shot = dict(sh)
            new_shot["start"] = s
            new_shot["end"] = e
            valid_shots.append(new_shot)

    if not valid_shots:
        st.error("No valid shots after clamping.")
        for b in clips_by_path.values():
            b.close()
        return

    orig_lengths = [sh["end"] - sh["start"] for sh in valid_shots]
    total_original = sum(orig_lengths)

    # ---------- Duration logic ----------
    if target_duration <= 0 or target_duration >= total_original:
        segments = [(sh, sh["start"], sh["end"]) for sh in valid_shots]
        st.info("No trimming required.")
    else:
        protected_indices = [i for i, L in enumerate(orig_lengths) if L < MIN_PROTECT_SECONDS]
        adjustable_indices = [i for i, L in enumerate(orig_lengths) if L >= MIN_PROTECT_SECONDS]

        protected_duration = sum(orig_lengths[i] for i in protected_indices)
        adjustable_total = total_original - protected_duration

        if protected_duration >= target_duration or adjustable_total <= 0:
            scale = target_duration / total_original
            st.info(
                f"Total selected duration {total_original:.2f}s > target "
                f"{target_duration:.2f}s → trimming ALL shots with scale {scale:.3f}"
            )
            new_lengths = [L * scale for L in orig_lengths]
        else:
            scale = (target_duration - protected_duration) / adjustable_total
            st.info(
                f"Total selected duration {total_original:.2f}s > target {target_duration:.2f}s "
                f"→ trimming only shots ≥ {MIN_PROTECT_SECONDS:.2f}s with scale {scale:.3f}"
            )
            new_lengths = orig_lengths[:]
            for i in adjustable_indices:
                new_lengths[i] = orig_lengths[i] * scale

        diff = target_duration - sum(new_lengths)
        if abs(diff) > 0.05 and adjustable_indices:
            last_i = adjustable_indices[-1]
            new_lengths[last_i] = max(0.05, new_lengths[last_i] + diff)

        segments = []
        for sh, new_len in zip(valid_shots, new_lengths):
            vp = sh["video_path"]
            base = clips_by_path[vp]
            s = sh["start"]
            new_end = min(s + new_len, base.duration)
            if new_end > s:
                segments.append((sh, s, new_end))

    # ---------- Build clips (video only, no per-clip audio) ----------
    clips = []
    for sh, s, e in segments:
        vp = sh["video_path"]
        base = clips_by_path[vp]
        sub = base.subclip(s, e).without_audio()

        if vertical_9_16 and sh.get("crop") is not None:
            sub = apply_crop_to_clip(sub, sh["crop"]).resize(height=1920)
        elif square_1_1:
            w, h = sub.w, sub.h
            square_box = compute_square_crop_from_box(w, h, sh.get("crop"))
            x1, y1, x2, y2 = square_box
            sub = vfx.crop(sub, x1=x1, y1=y1, x2=x2, y2=y2).resize(height=1080)
        elif youtube_16_9:
            w, h = sub.w, sub.h
            x1, y1, x2, y2 = compute_16_9_center_crop(w, h)
            sub = vfx.crop(sub, x1=x1, y1=y1, x2=x2, y2=y2).resize(height=1080)

        clips.append(sub)

    if not clips:
        st.error("Nothing to render after trimming.")
        for b in clips_by_path.values():
            b.close()
        return

    final = concatenate_videoclips(clips, method="compose")

    # ---------- Single audio source ----------
    audio_clip = None
    try:
        if audio_source == "video1" and video1_path:
            base_audio_vid = VideoFileClip(video1_path)
            if base_audio_vid.audio:
                audio_clip = base_audio_vid.audio.subclip(0, final.duration)
        elif audio_source == "video2" and video2_path:
            base_audio_vid = VideoFileClip(video2_path)
            if base_audio_vid.audio:
                audio_clip = base_audio_vid.audio.subclip(0, final.duration)
        elif audio_source == "external" and audio_path:
            ext_audio = AudioFileClip(audio_path)
            audio_clip = ext_audio.subclip(0, final.duration)
    except Exception as e:
        st.warning(f"Audio loading error: {e}")
        audio_clip = None

    try:
        if audio_clip:
            final = final.set_audio(audio_clip)
            final.write_videofile(outname, codec="libx264", audio_codec="aac")
        else:
            final.write_videofile(outname, codec="libx264", audio=False)
    except Exception as e:
        print("Error writing video, trying silent render:", e)
        final.write_videofile(outname, codec="libx264", audio=False)

    # cleanup
    if audio_clip:
        audio_clip.close()
    try:
        base_audio_vid.close()
    except Exception:
        pass

    final.close()
    for c in clips:
        c.close()
    for b in clips_by_path.values():
        b.close()


# --------------------------------------------------------
# UI – CLEAN LAYOUT
# --------------------------------------------------------
st.markdown("### 1. Upload your two source videos")

col_up1, col_up2 = st.columns(2)
video1_path = None
video2_path = None

with col_up1:
    st.subheader("🎥 Video 1")
    uploaded_video1 = st.file_uploader("Upload Video 1", type=["mp4", "mov", "mkv"], key="v1")
    if uploaded_video1:
        video1_path = "input1.mp4"
        with open(video1_path, "wb") as f:
            f.write(uploaded_video1.read())
        st.markdown("**Preview – Video 1**")
        st.video(video1_path, format="video/mp4")

with col_up2:
    st.subheader("🎥 Video 2")
    uploaded_video2 = st.file_uploader("Upload Video 2", type=["mp4", "mov", "mkv"], key="v2")
    if uploaded_video2:
        video2_path = "input2.mp4"
        with open(video2_path, "wb") as f:
            f.write(uploaded_video2.read())
        st.markdown("**Preview – Video 2**")
        st.video(video2_path, format="video/mp4")

st.markdown("---")

if video1_path and video2_path and st.button("🔍 Analyze & Detect Shots for BOTH"):
    with st.spinner("Detecting shots and analyzing composition for both videos..."):
        # -------- Video 1 --------
        raw_shots1 = detect_shots(video1_path)
        shot_meta1 = []
        for (s, e) in raw_shots1:
            crop_box, is_good = analyze_shot(video1_path, s, e)
            shot_meta1.append(
                {
                    "video_path": video1_path,
                    "start": s,
                    "end": e,
                    "crop": crop_box,
                    "is_good": is_good,
                }
            )
        previews1, thumbs1 = generate_previews(video1_path, shot_meta1, prefix="v1")

        # -------- Video 2 --------
        raw_shots2 = detect_shots(video2_path)
        shot_meta2 = []
        for (s, e) in raw_shots2:
            crop_box, is_good = analyze_shot(video2_path, s, e)
            shot_meta2.append(
                {
                    "video_path": video2_path,
                    "start": s,
                    "end": e,
                    "crop": crop_box,
                    "is_good": is_good,
                }
            )
        previews2, thumbs2 = generate_previews(video2_path, shot_meta2, prefix="v2")

    st.session_state["shots_v1"] = shot_meta1
    st.session_state["previews_v1"] = previews1
    st.session_state["thumbs_v1"] = thumbs1
    st.session_state["shots_v2"] = shot_meta2
    st.session_state["previews_v2"] = previews2
    st.session_state["thumbs_v2"] = thumbs2

# -------- Shot selection UI --------
if "shots_v1" in st.session_state or "shots_v2" in st.session_state:
    st.markdown("### 2. Select shots from each video")

    if "selection_order" not in st.session_state:
        st.session_state["selection_order"] = []

    current_selected_ids = []
    id_to_info = {}

    col_v1, col_v2 = st.columns(2)

    # ------------- VIDEO 1 SHOTS -------------
    with col_v1:
        with st.container(border=True):
            st.markdown("#### ▶ Video 1 Shots")
            if "shots_v1" in st.session_state:
                shots = st.session_state["shots_v1"]
                previews = st.session_state["previews_v1"]
                if shots:
                    shot_cols = st.columns(2)  # 2-shot grid per row
                    for i, (shot, preview) in enumerate(zip(shots, previews)):
                        shot_id = f"v1-{i}"
                        with shot_cols[i % 2]:
                            st.video(preview)
                            s = shot["start"]
                            e = shot["end"]
                            dur = e - s
                            comp_tag = "✅ GOOD" if shot["is_good"] else "⚠️ WEAK"
                            st.caption(
                                f"{s:.2f}s → {e:.2f}s  | len {dur:.2f}s | {comp_tag}"
                            )
                            checked = st.checkbox(
                                f"Use V1 Shot {i}", key=f"cb_v1_{i}"
                            )
                            if checked:
                                current_selected_ids.append(shot_id)
                                id_to_info[shot_id] = ("v1", i)

    # ------------- VIDEO 2 SHOTS -------------
    with col_v2:
        with st.container(border=True):
            st.markdown("#### ▶ Video 2 Shots")
            if "shots_v2" in st.session_state:
                shots = st.session_state["shots_v2"]
                previews = st.session_state["previews_v2"]
                if shots:
                    shot_cols = st.columns(2)
                    for i, (shot, preview) in enumerate(zip(shots, previews)):
                        shot_id = f"v2-{i}"
                        with shot_cols[i % 2]:
                            st.video(preview)
                            s = shot["start"]
                            e = shot["end"]
                            dur = e - s
                            comp_tag = "✅ GOOD" if shot["is_good"] else "⚠️ WEAK"
                            st.caption(
                                f"{s:.2f}s → {e:.2f}s  | len {dur:.2f}s | {comp_tag}"
                            )
                            checked = st.checkbox(
                                f"Use V2 Shot {i}", key=f"cb_v2_{i}"
                            )
                            if checked:
                                current_selected_ids.append(shot_id)
                                id_to_info[shot_id] = ("v2", i)

    # ----- maintain selection order (first selected = first in timeline) -----
    order = st.session_state["selection_order"]
    # remove ids that are no longer selected
    order = [sid for sid in order if sid in current_selected_ids]
    # append newly selected ids at the end
    for sid in current_selected_ids:
        if sid not in order:
            order.append(sid)
    st.session_state["selection_order"] = order

    # ---------- Reorder all selected shots ----------
    st.markdown("---")
    st.markdown("### 3. Build & reorder your final timeline")

    combined_labels = []
    label_to_data = {}
    total_selected_duration = 0.0

    # Build labels in current selection order
    for sid in order:
        src, idx = id_to_info.get(sid, (None, None))
        if src is None:
            continue
        if src == "v1":
            sh = st.session_state["shots_v1"][idx]
            thumb = st.session_state["thumbs_v1"][idx]
            tag = f"V1-{idx}"
        else:
            sh = st.session_state["shots_v2"][idx]
            thumb = st.session_state["thumbs_v2"][idx]
            tag = f"V2-{idx}"

        s, e = sh["start"], sh["end"]
        dur = e - s
        label = f"{tag} | {dur:.2f}s | {s:.2f}→{e:.2f}"

        combined_labels.append(label)
        label_to_data[label] = (sh, thumb, tag)
        total_selected_duration += dur

    ordered_shots = []

    if combined_labels:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric("Total selected duration", f"{total_selected_duration:.2f} s")
        with c2:
            st.caption("Drag the blocks horizontally to set the final order (left = first):")

        # SORTABLE BAR – HORIZONTAL
        sorted_labels = sort_items(combined_labels, direction="horizontal")

        # Build ordered_shots list according to sorted_labels
        for lbl in sorted_labels:
            sh, thumb, tag = label_to_data[lbl]
            ordered_shots.append(sh)

        # Thumbnail timeline (bigger, easier to see)
        if sorted_labels:
            st.markdown("#### Timeline preview")
            n_cols = 6
            cols = st.columns(n_cols)
            for i, lbl in enumerate(sorted_labels):
                sh, thumb, tag = label_to_data[lbl]
                col = cols[i % n_cols]
                with col:
                    st.image(thumb, use_column_width=True)
                    st.caption(tag)

        with st.expander("View ordered timeline list"):
            for i, lbl in enumerate(sorted_labels):
                st.write(f"{i+1}. {lbl}")
    else:
        st.info("Select at least one shot from Video 1 or Video 2 to build a timeline.")

    # ---------- Audio selection ----------
    st.markdown("---")
    st.markdown("### 4. Choose audio for final video")

    audio_choice = st.radio(
        "Audio source",
        ["Use Video 1 audio", "Use Video 2 audio", "Upload separate audio"],
        index=0,
    )

    audio_source = "video1"
    external_audio_path = None

    if audio_choice == "Use Video 1 audio":
        audio_source = "video1"
    elif audio_choice == "Use Video 2 audio":
        audio_source = "video2"
    else:
        audio_source = "external"
        audio_upload = st.file_uploader(
            "Upload audio file (mp3 / wav / m4a)", type=["mp3", "wav", "m4a"], key="audio"
        )
        if audio_upload:
            ext = os.path.splitext(audio_upload.name)[1]
            if not ext:
                ext = ".mp3"
            external_audio_path = "external_audio" + ext
            with open(external_audio_path, "wb") as f:
                f.write(audio_upload.read())
            st.caption(f"Loaded external audio: {audio_upload.name}")

    # ---------- Final Output Settings ----------
    st.markdown("---")
    st.markdown("### 5. Final output settings")

    target_dur = st.number_input("Target Duration (seconds)", value=10.0)
    output_name = st.text_input("Output File Name", value="final_mix_output.mp4")

    vertical_flag = st.checkbox(
        "Export as vertical 9:16 (Reels / TikTok format)",
        value=True,
    )

    square_flag = st.checkbox(
        "Export as square 1:1 (Feed / WhatsApp)",
        value=False,
    )

    youtube_flag = st.checkbox(
        "Export as 16:9 (YouTube)",
        value=False,
    )

    selected_formats = sum([vertical_flag, square_flag, youtube_flag])
    if selected_formats > 1:
        st.warning("Multiple output formats selected. Priority: 9:16 > 1:1 > 16:9.")

    if st.button("🚀 Render Final Mixed Video"):
        if not ordered_shots:
            st.error("Select at least one shot and set an order.")
        else:
            total_selected_duration_chk = sum(
                float(sh["end"]) - float(sh["start"]) for sh in ordered_shots
            )

            if total_selected_duration_chk < float(target_dur):
                st.warning(
                    f"Selected shots total duration is only "
                    f"{total_selected_duration_chk:.2f}s, which is less than "
                    f"the target duration {float(target_dur):.2f}s. "
                    "Please select more shots or reduce the target duration."
                )
            else:
                with st.spinner("Rendering final mixed video..."):
                    render_final(
                        ordered_shots,
                        output_name,
                        target_dur,
                        vertical_9_16=vertical_flag,
                        square_1_1=(square_flag and not vertical_flag),
                        youtube_16_9=(youtube_flag and not vertical_flag and not square_flag),
                        audio_source=audio_source,
                        audio_path=external_audio_path,
                        video1_path=video1_path,
                        video2_path=video2_path,
                    )

                st.success("Final mixed video rendered!")
                st.video(output_name)
# --------------------------------------------------------
# HIGHLIGHTED TAGLINE (LAST LINE)
# --------------------------------------------------------
st.markdown(
    """
    <div style="
        background:linear-gradient(90deg,#ff0077,#ff8800);
        color:white;
        padding:16px;
        border-radius:12px;
        text-align:center;
        font-size:22px;
        font-weight:700;
        margin-top:40px;
    ">
        More speed. More control. More efficiency.
    </div>
    """,
    unsafe_allow_html=True,
)
