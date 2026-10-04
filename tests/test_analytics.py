from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from ipcam.analytics import (
    AnalyticsEngine,
    ByteTrackConfig,
    ByteTracker,
    EventKind,
    HeuristicConfig,
    HeuristicFilter,
    KalmanBoxFilter,
    PolygonZone,
    Track,
    Tripwire,
    build_rules,
    iou_matrix,
    points_in_polygon,
    segments_intersect,
)
from ipcam.analytics.rules import TrackView


def box(cx: float, foot_y: float, h: float, aspect: float = 0.4) -> list[float]:
    w = h * aspect
    return [cx - w / 2, foot_y - h, cx + w / 2, foot_y]


def view(tid: int, foot: tuple[float, float], t: float, *, valid: bool = True, observed: bool = True,
         h: float = 60.0, maturing: bool = False) -> TrackView:
    b = box(foot[0], foot[1], h)
    return TrackView(tid, t, foot, tuple(b), (0.0, 0.0), h, 0.9, observed, valid, (), maturing)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- geometry

def test_segment_intersection_respects_extent() -> None:
    a, b = (0.0, 0.0), (10.0, 0.0)
    assert segments_intersect((5, -1), (5, 1), a, b)
    assert not segments_intersect((15, -1), (15, 1), a, b)    # crosses the line, not the segment
    assert not segments_intersect((5, 1), (6, 2), a, b)


def test_ray_casting_non_convex_and_vertices() -> None:
    u_shape = np.array([[0, 0], [30, 0], [30, 30], [20, 30], [20, 10], [10, 10], [10, 30], [0, 30]], float)
    pts = np.array([[5, 20], [15, 20], [25, 20], [15, 5], [-1, 5], [15, 0.0001]])
    assert points_in_polygon(pts, u_shape).tolist() == [True, False, True, True, False, True]
    # ray passing exactly through vertices (y = 10 hits (20,10) and (10,10))
    assert points_in_polygon(np.array([[5, 10], [15, 10]]), u_shape).tolist() == [True, False]


def test_buffered_iou_bridges_gap() -> None:
    a = np.array([[0, 0, 10, 30]], float)
    b = np.array([[14, 0, 24, 30]], float)
    assert iou_matrix(a, b)[0, 0] == 0.0
    assert iou_matrix(a, b, buffer=0.3)[0, 0] > 0.0


# --------------------------------------------------------------------------- tracker

def run_track(seq: list[tuple[float, list[list[float]]]], cfg: ByteTrackConfig | None = None) -> list[list[int]]:
    tracker = ByteTracker(cfg)
    ids = []
    for t, dets in seq:
        out = tracker.update(np.array(dets, float) if dets else np.zeros((0, 5)), t)
        ids.append([tr.track_id for tr in out.tracks])
    return ids


def test_low_confidence_occlusion_keeps_identity() -> None:
    seq = []
    for i in range(60):
        conf = 0.25 if 20 <= i < 35 else 0.9           # behind a tree: score collapses
        seq.append((i * 0.04, [box(100 + i * 3, 300, 80) + [conf]]))
    ids = run_track(seq)
    assert {i for frame in ids[1:] for i in frame} == {1}
    assert all(frame == [1] for frame in ids[1:])     # never dropped, stage 2 held it


def test_full_occlusion_reidentified_from_lost_pool() -> None:
    seq = []
    for i in range(80):
        dets = [] if 30 <= i < 50 else [box(100 + i * 4, 300, 80) + [0.9]]   # 0.8 s fully hidden
        seq.append((i * 0.04, dets))
    ids = run_track(seq)
    seen = {i for frame in ids for i in frame}
    assert seen == {1}
    assert ids[-1] == [1]


def test_small_fast_target_with_dropped_frames() -> None:
    """15 px person at 150 px/s, processed at irregular 60-200 ms gaps: plain IoU is 0."""
    rng = np.random.default_rng(3)
    t, x, seq = 0.0, 50.0, []
    while x < 600:
        seq.append((t, [box(x, 200, 15) + [0.85]]))
        dt = float(rng.uniform(0.06, 0.2))
        t += dt
        x += 150 * dt
    ids = run_track(seq)
    assert {i for frame in ids for i in frame} == {1}


def test_crossing_pedestrians_do_not_swap() -> None:
    seq = []
    for i in range(70):
        a = box(100 + i * 5, 300, 90)
        b = box(450 - i * 5, 302, 90)
        seq.append((i * 0.04, [a + [0.9], b + [0.9]]))
    tracker = ByteTracker()
    for t, dets in seq:
        out = tracker.update(np.array(dets), t)
    final = {tr.track_id: tr.foot[0] for tr in out.tracks}
    assert final == pytest.approx({1: 100 + 69 * 5, 2: 450 - 69 * 5}, abs=10)


def simulate_scene(seed: int, n_people: int = 8, seconds: float = 20.0):
    """Ground truth + noisy detector: misses, occlusion dips, false positives, frame drops."""
    rng = np.random.default_rng(seed)
    people = []
    for k in range(n_people):
        h = float(rng.uniform(18, 140))
        people.append({
            "id": k, "x": float(rng.uniform(50, 1200)), "y": float(rng.uniform(150, 680)), "h": h,
            "vx": float(rng.uniform(-1.2, 1.2)) * h, "vy": float(rng.uniform(-0.3, 0.3)) * h,
            "t0": float(rng.uniform(0, 6)), "t1": float(rng.uniform(12, seconds)),
        })
    frames, t = [], 0.0
    while t < seconds:
        gt, dets = [], []
        for p in people:
            if not p["t0"] <= t <= p["t1"]:
                continue
            dt_alive = t - p["t0"]
            cx = p["x"] + p["vx"] * dt_alive + 3 * np.sin(dt_alive * 2)
            fy = p["y"] + p["vy"] * dt_alive
            b = box(cx, fy, p["h"])
            gt.append((p["id"], b))
            r = rng.random()
            if r < 0.08:
                continue                                  # missed detection
            conf = float(rng.uniform(0.2, 0.5)) if r < 0.18 else float(rng.uniform(0.65, 0.95))
            noise = rng.normal(0, 0.03 * p["h"], 4)
            dets.append(list(np.array(b) + noise) + [conf])
        for _ in range(rng.poisson(0.3)):                 # low-score clutter
            fx, fyy = rng.uniform(0, 1280), rng.uniform(100, 720)
            dets.append(box(fx, fyy, rng.uniform(15, 60), 1.0) + [float(rng.uniform(0.15, 0.4))])
        frames.append((t, gt, dets))
        t += float(rng.choice([0.04, 0.04, 0.04, 0.08, 0.12]))   # drop-oldest gaps
    return frames


def mota(frames, tracker: ByteTracker) -> tuple[float, int]:
    fn = fp = idsw = n_gt = 0
    last_match: dict[int, int] = {}
    for t, gt, dets in frames:
        out = tracker.update(np.array(dets) if dets else np.zeros((0, 5)), t)
        n_gt += len(gt)
        trs = out.tracks
        if gt and trs:
            iou = iou_matrix(np.array([b for _, b in gt]), np.stack([tr.box for tr in trs]))
            r, c = linear_sum_assignment(1 - iou)
            pairs = [(i, j) for i, j in zip(r, c) if iou[i, j] >= 0.5]
        else:
            pairs = []
        fn += len(gt) - len(pairs)
        fp += len(trs) - len(pairs)
        for i, j in pairs:
            gid, tid = gt[i][0], trs[j].track_id
            if gid in last_match and last_match[gid] != tid:
                idsw += 1
            last_match[gid] = tid
    return 1 - (fn + fp + idsw) / max(n_gt, 1), idsw


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_mota_on_synthetic_cctv_scene(seed: int) -> None:
    score, idsw = mota(simulate_scene(seed), ByteTracker())
    assert score > 0.85, f"MOTA {score:.3f}, idsw {idsw}"
    assert idsw <= 4


def test_tracker_throughput() -> None:
    frames = simulate_scene(9, n_people=40)
    tracker = ByteTracker()
    t0 = time.perf_counter()
    for t, _, dets in frames:
        tracker.update(np.array(dets) if dets else np.zeros((0, 5)), t)
    per_frame_ms = (time.perf_counter() - t0) * 1000 / len(frames)
    assert per_frame_ms < 25, per_frame_ms


# --------------------------------------------------------------------------- tripwire

def walk(wire: Tripwire, path: list[tuple[float, float]], tid: int = 1, **kw: object) -> list:
    events = []
    for i, p in enumerate(path):
        events += wire.evaluate([view(tid, p, i * 0.1, **kw)], i * 0.1)  # type: ignore[arg-type]
    return events


def test_tripwire_direction() -> None:
    wire = Tripwire("gate", (0, 100), (200, 100), inside=(100, 200), alarm_on="in")
    ev_in = walk(wire, [(100, 50 + i * 10) for i in range(11)], tid=1)
    ev_out = walk(wire, [(100, 150 - i * 10) for i in range(11)], tid=2)
    assert [e.direction for e in ev_in] == ["in"] and ev_in[0].kind is EventKind.TRIPWIRE
    assert ev_out == []
    both = Tripwire("g2", (0, 100), (200, 100), inside=(100, 200), alarm_on="both")
    assert [e.direction for e in walk(both, [(100, 150 - i * 10) for i in range(11)])] == ["out"]


def test_tripwire_hysteresis_and_segment_extent() -> None:
    wire = Tripwire("gate", (0, 100), (200, 100), inside=(100, 200), alarm_on="both")
    jitter = [(100, 100 + (2 if i % 2 else -2)) for i in range(30)]   # standing on the line
    assert walk(wire, jitter) == []
    around = [(250, 50), (260, 100), (250, 150)]                      # walks past the end
    assert walk(wire, around, tid=5) == []


def test_tripwire_pending_until_track_validated() -> None:
    wire = Tripwire("gate", (0, 100), (200, 100), inside=(100, 200), alarm_on="in")
    assert wire.evaluate([view(1, (100, 80), 0.0, valid=False, maturing=True)], 0.0) == []
    assert wire.evaluate([view(1, (100, 120), 0.1, valid=False, maturing=True)], 0.1) == []
    ev = wire.evaluate([view(1, (100, 125), 0.4)], 0.4)
    assert len(ev) == 1 and ev[0].details["crossed_at"] == 0.1
    # a track rejected by heuristics (e.g. a cat) never alarms
    assert walk(Tripwire("g", (0, 100), (200, 100), inside=(100, 200)),
                [(100, 50 + i * 10) for i in range(11)], valid=False) == []


# --------------------------------------------------------------------------- polygon zone

SQUARE = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], float)


def test_intrusion_min_dwell_and_exit() -> None:
    z = PolygonZone("r", SQUARE, min_dwell_s=0.5, exit_grace_s=0.5, loiter_s=None)
    evs = []
    for i in range(20):
        p = (200, 200) if i < 10 else (400, 400)
        evs += z.evaluate([view(1, p, i * 0.1)], i * 0.1)
    kinds = [e.kind for e in evs]
    assert kinds == [EventKind.INTRUSION, EventKind.ZONE_EXIT]
    assert evs[0].t == pytest.approx(0.5)
    flick = PolygonZone("r2", SQUARE, min_dwell_s=0.5, loiter_s=None)
    assert flick.evaluate([view(2, (200, 200), 0.0)], 0.0) == []      # single-frame border jitter
    assert flick.evaluate([view(2, (400, 400), 0.1)], 0.1) == []


def test_loitering_radius() -> None:
    still = PolygonZone("z", SQUARE, loiter_s=15, loiter_radius_px=30)
    walker = PolygonZone("z", SQUARE, loiter_s=15, loiter_radius_px=30)
    ev_still, ev_walk = [], []
    for i in range(170):
        t = i * 0.1
        ev_still += still.evaluate([view(1, (200 + 10 * np.sin(t), 200), t)], t)
        ev_walk += walker.evaluate([view(1, (120 + (i % 160), 200), t)], t)
    assert [e.kind for e in ev_still] == [EventKind.INTRUSION, EventKind.LOITERING]
    assert ev_still[1].dwell_s >= 15
    assert EventKind.LOITERING not in [e.kind for e in ev_walk]


def test_zone_state_frozen_during_occlusion() -> None:
    z = PolygonZone("z", SQUARE, loiter_s=3, loiter_radius_px=30, exit_grace_s=0.5)
    evs = []
    for i in range(40):
        t = i * 0.1
        observed = not 10 <= i < 25            # 1.5 s behind a pillar
        evs += z.evaluate([view(1, (200, 200), t, observed=observed)], t)
    assert [e.kind for e in evs] == [EventKind.INTRUSION, EventKind.LOITERING]


# --------------------------------------------------------------------------- heuristics + engine

def test_heuristics_reject_animals_and_glare() -> None:
    hf = HeuristicFilter(HeuristicConfig(min_age_s=0.0))
    tracker = ByteTracker(ByteTrackConfig(min_hits=1))
    cat = None
    for i in range(10):
        out = tracker.update(np.array([[100 + i * 4, 300, 160 + i * 4, 330, 0.9]]), i * 0.04)  # wide box
        cat = out.tracks[0]
    assert "aspect" in hf.assess(cat).reasons

    # Headlight glare / bird: a "person" whose foot point jumps 10 body heights per frame.
    kf = KalmanBoxFilter()
    b0 = np.array(box(100, 300, 30))
    glare = Track(99, *kf.initiate(b0), b0, 0.9, 0, 0.0, 50)
    for i in range(1, 10):
        b = np.array(box(100 + (i % 2) * 300, 300, 30))
        glare._record(i * 0.04, b, 0.9, observed=True)
    glare.t_state = 0.36
    assert "implausible_speed" in hf.assess(glare).reasons

    walker = Track(98, *kf.initiate(b0), b0, 0.9, 0, 0.0, 50)
    for i in range(1, 10):
        walker._record(i * 0.04, np.array(box(100 + i * 1.2, 300, 30)), 0.9, observed=True)
    walker.t_state = 0.36
    assert hf.assess(walker).valid


def test_engine_end_to_end_with_config() -> None:
    cfg = {"rules": [
        {"type": "tripwire", "name": "gate", "a": [0.0, 0.5], "b": [1.0, 0.5], "inside": [0.5, 0.9]},
        {"type": "zone", "name": "yard", "polygon": [[0.5, 0.55], [1, 0.55], [1, 1], [0.5, 1]], "loiter_s": None},
    ]}
    rules = build_rules(cfg, 640, 360)
    engine = AnalyticsEngine(rules=rules)
    events = []
    for i in range(80):
        t = i * 0.04
        res = engine.update(np.array([box(400, 120 + i * 2.5, 70) + [0.9]]), t)
        events += res.events
    kinds = [(e.kind, e.rule) for e in events]
    assert (EventKind.TRIPWIRE, "gate") in kinds
    assert (EventKind.INTRUSION, "yard") in kinds
    assert engine.timing.summary().p95 < 20


def test_config_rejects_self_intersecting_zone() -> None:
    with pytest.raises(ValueError):
        build_rules({"rules": [{"type": "zone", "name": "bow", "polygon": [[0, 0], [1, 1], [1, 0], [0, 1]]}]},
                    100, 100)
