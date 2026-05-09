import h5py
import cv2
import numpy as np
import argparse
import os


def edit_log(filepath, episode=0, lazy=False):
    try:
        f = h5py.File(filepath, 'r')
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    images_ds = f['images']           # (E, T_max, H, W, 3) uint8
    actions_ds = f['actions']         # (E, T_max, A) float32

    el = f['episode_lengths']
    episode_lengths = el[:] if el.shape != () else np.array([el[()]])

    num_episodes = images_ds.shape[0]
    T_max = images_ds.shape[1]

    if episode < 0 or episode >= num_episodes:
        print(f"Episode {episode} out of range (shard has {num_episodes} episode(s))")
        f.close()
        return

    T = int(episode_lengths[episode])
    actions = actions_ds[episode, :T]
    images = images_ds[episode, :T] if not lazy else None

    if 'is_demo' in f:
        ds = f['is_demo']
        if ds.ndim == 1:
            arr = ds[:].astype(np.uint8)
            is_demo = np.zeros(T, dtype=np.uint8)
            is_demo[:min(T, len(arr))] = arr[:min(T, len(arr))]
        else:
            is_demo = ds[episode, :T].astype(np.uint8).copy()
            if is_demo.ndim > 1:
                is_demo = is_demo.squeeze()
    else:
        is_demo = np.zeros(T, dtype=np.uint8)

    max_act = float(np.abs(actions[:, :2]).max())
    if max_act < 1e-6:
        max_act = 1.0
    ARROW_PX = 150

    H5_PANEL_SIZE = 768
    TIMELINE_H = 80
    TOTAL_W = H5_PANEL_SIZE

    state = {"cursor": 0, "running": True, "save": False}
    timeline_base = np.zeros((TIMELINE_H, TOTAL_W, 3), dtype=np.uint8) + 30
    sel = {"start": None, "end": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and y > H5_PANEL_SIZE:
            ratio = x / TOTAL_W
            state["cursor"] = max(0, min(int(ratio * T), T - 1))

    window_name = f"Demo Labeler - {os.path.basename(filepath)} [ep {episode}/{num_episodes - 1}]"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    print("\n--- CONTROLS ---")
    print("[ A / D or Arrows ] : Scrub  (hold Shift via [ J / L ] for x10)")
    print("[ [ / ] ]           : Mark range start / end")
    print("[ R ]               : Reset selection")
    print("[ M ]               : Toggle is_demo on cursor or selected range")
    print("[ S / Enter ]       : Save and quit")
    print("[ Q / Esc ]         : Quit without saving")

    while state["running"]:
        i = state["cursor"]

        img_raw = images[i] if images is not None else images_ds[episode, i]
        h5_img = cv2.resize(img_raw, (H5_PANEL_SIZE, H5_PANEL_SIZE), interpolation=cv2.INTER_NEAREST)

        a = actions[i]
        cx, cy = H5_PANEL_SIZE // 2, H5_PANEL_SIZE // 2
        vec_x = int((a[0] / max_act) * ARROW_PX)
        vec_y = int(-(a[1] / max_act) * ARROW_PX)
        if abs(vec_x) > 2 or abs(vec_y) > 2:
            cv2.arrowedLine(h5_img, (cx, cy), (cx + vec_x, cy + vec_y),
                            (255, 255, 0), 4, tipLength=0.2)

        if is_demo[i]:
            cv2.putText(h5_img, "DEMO", (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 4)

        timeline_view = timeline_base.copy()

        demo_mask = is_demo > 0
        if demo_mask.any():
            xs = (np.arange(T)[demo_mask] / T * TOTAL_W).astype(int)
            for x in xs:
                cv2.line(timeline_view, (x, TIMELINE_H // 2), (x, TIMELINE_H), (0, 200, 0), 1)

        if sel["start"] is not None:
            s = sel["start"]
            e = sel["end"] if sel["end"] is not None else i
            sd, ed = min(s, e), max(s, e)
            x1 = int((sd / T) * TOTAL_W)
            x2 = int((ed / T) * TOTAL_W)
            cv2.rectangle(timeline_view, (x1, 10), (x2, TIMELINE_H), (0, 255, 255), -1)

        cursor_x = int((i / T) * TOTAL_W)
        cv2.line(timeline_view, (cursor_x, 0), (cursor_x, TIMELINE_H), (0, 255, 0), 2)

        n_demo = int(is_demo.sum())
        pct = 100.0 * n_demo / max(T, 1)
        stats = f"Frame: {i}/{T - 1} | Demo: {n_demo}/{T} ({pct:.1f}%) | Ep {episode}/{num_episodes - 1}"
        cv2.putText(timeline_view, stats, (10, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)

        full_ui = np.vstack((h5_img, timeline_view))
        cv2.imshow(window_name, full_ui)

        key = cv2.waitKey(20) & 0xFF

        if key == 27 or key == ord('q'):
            state["running"] = False
        elif key == ord('d') or key == 83:
            state["cursor"] = min(i + 1, T - 1)
        elif key == ord('a') or key == 81:
            state["cursor"] = max(i - 1, 0)
        elif key == ord('l'):
            state["cursor"] = min(i + 10, T - 1)
        elif key == ord('j'):
            state["cursor"] = max(i - 10, 0)
        elif key == ord('['):
            sel["start"] = i
            sel["end"] = None
        elif key == ord(']'):
            if sel["start"] is not None:
                sel["end"] = i
        elif key == ord('r'):
            sel["start"] = None
            sel["end"] = None
        elif key == ord('m'):
            if sel["start"] is not None:
                e = sel["end"] if sel["end"] is not None else i
                s, ee = min(sel["start"], e), max(sel["start"], e)
                is_demo[s:ee + 1] = 1 - is_demo[s:ee + 1]
            else:
                is_demo[i] = 1 - is_demo[i]
        elif key == ord('s') or key == 13:
            state["save"] = True
            state["running"] = False

    f.close()
    cv2.destroyAllWindows()

    if state["save"]:
        save_is_demo(filepath, episode, is_demo, T_max, num_episodes)


def save_is_demo(filepath, episode, is_demo, T_max, num_episodes):
    print(f"\n--- SAVING is_demo to {filepath} (episode {episode}) ---")
    with h5py.File(filepath, 'r+') as f:
        if 'is_demo' in f:
            ds = f['is_demo']
            if ds.shape != (num_episodes, T_max):
                print(f"Existing is_demo shape {ds.shape} != ({num_episodes}, {T_max}). Recreating.")
                full = np.zeros((num_episodes, T_max), dtype=np.uint8)
                if ds.ndim == 2 and ds.shape[0] == num_episodes:
                    full[:, :min(T_max, ds.shape[1])] = ds[:, :min(T_max, ds.shape[1])]
                del f['is_demo']
                full[episode, :len(is_demo)] = is_demo
                f.create_dataset('is_demo', data=full, dtype='uint8')
            else:
                ds[episode, :len(is_demo)] = is_demo
        else:
            full = np.zeros((num_episodes, T_max), dtype=np.uint8)
            full[episode, :len(is_demo)] = is_demo
            f.create_dataset('is_demo', data=full, dtype='uint8')
    n_demo = int(is_demo.sum())
    print(f"Saved: {n_demo}/{len(is_demo)} demo frames ({100 * n_demo / max(len(is_demo), 1):.1f}%).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label is_demo frames in a pushT shard.")
    parser.add_argument("file", help="Path to .h5 shard file")
    parser.add_argument("--episode", "-e", type=int, default=0,
                        help="Episode index within the shard (default: 0)")
    parser.add_argument("--lazy", action="store_true",
                        help="Read frames from disk on demand instead of loading the episode into RAM")
    args = parser.parse_args()
    edit_log(args.file, episode=args.episode, lazy=args.lazy)
