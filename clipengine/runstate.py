"""In-memory state of one run: its sources and clips as rows with a state, and its events.

No database: the rows live in this process for the length of a run, and engine.run writes what
it made to the output folder (clip_NN.mp4, clip_NN.json, run.json). The functions mirror a job
store's (add / update / read rows) so every engine step reads and moves rows the same way.

States:
  source: new -> harvested -> selected -> done | failed
  clip:   selected -> rendered | failed
"""
import copy
import json
import time

_SOURCES = {}          # source_id -> row
_CLIPS = {}            # clip_id -> row
_EVENTS = []
_NEXT_ID = [1]

SOURCE_DEFAULTS = {"url": None, "title": None, "local_path": None, "duration_s": None,
                   "content_type": "auto", "vertical": "general", "channel": None, "game": None,
                   "niche": None, "state": "new", "error": None}
CLIP_DEFAULTS = {"start_s": None, "end_s": None, "score": None, "layout": "general", "hook": None,
                 "state": "selected", "words_path": None, "features_json": None,
                 "boundary_type": None, "error": None, "out_path": None}


def _now():
    return time.time()


def reset():
    """Forget every row and event (a new run, or a test)."""
    _SOURCES.clear()
    _CLIPS.clear()
    _EVENTS.clear()
    _NEXT_ID[0] = 1


def init():
    """Nothing to set up: kept so callers can start a run the same way they reset it."""
    return None


# ---- sources ----
def add_source(source_id, url=None, title=None, local_path=None, duration_s=None,
               content_type="auto", vertical="general", channel=None, game=None, **extra):
    """Register a source (a second add of the same id is ignored); returns its id."""
    if source_id not in _SOURCES:
        row = {**SOURCE_DEFAULTS, **extra, "source_id": source_id, "url": url, "title": title,
               "local_path": local_path, "duration_s": duration_s, "content_type": content_type,
               "vertical": vertical, "channel": channel, "game": game,
               "created_at": _now(), "updated_at": _now()}
        _SOURCES[source_id] = row
    return source_id


def update_source(source_id, **kw):
    row = _SOURCES.get(source_id)
    if row is not None:
        row.update(kw, updated_at=_now())


def source(source_id):
    row = _SOURCES.get(source_id)
    return copy.deepcopy(row) if row is not None else None


def sources_in(state):
    return [copy.deepcopy(r) for r in _SOURCES.values() if r.get("state") == state]


# ---- clips ----
def _features_str(features_json):
    if features_json is not None and not isinstance(features_json, str):
        # numpy scalars -> native via .item()
        return json.dumps(features_json, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return features_json


def add_clip(source_id, start_s, end_s, score=None, layout="general", hook=None,
             words_path=None, features_json=None, boundary_type=None):
    """Add a selected clip; returns its clip_id, or 0 when the source already has a clip that
    starts at `start_s` (the same source moment is never clipped twice)."""
    if any(c["source_id"] == source_id and c["start_s"] == start_s for c in _CLIPS.values()):
        return 0
    cid = _NEXT_ID[0]
    _NEXT_ID[0] += 1
    _CLIPS[cid] = {**CLIP_DEFAULTS, "clip_id": cid, "source_id": source_id, "start_s": start_s,
                   "end_s": end_s, "score": score, "layout": layout, "hook": hook,
                   "words_path": words_path, "features_json": _features_str(features_json),
                   "boundary_type": boundary_type, "state": "selected",
                   "created_at": _now(), "updated_at": _now()}
    return cid


def update_clip(clip_id, **kw):
    """Update a clip row. A new start_s that another clip of the same source already starts at is
    refused (ValueError), as a unique (source, start) key refuses it."""
    row = _CLIPS.get(clip_id)
    if row is None:
        return
    if "start_s" in kw and any(c["clip_id"] != clip_id and c["source_id"] == row["source_id"]
                               and c["start_s"] == kw["start_s"] for c in _CLIPS.values()):
        raise ValueError(f"clip {clip_id}: another clip of {row['source_id']} starts at "
                         f"{kw['start_s']}")
    if "features_json" in kw:
        kw["features_json"] = _features_str(kw["features_json"])
    row.update(kw, updated_at=_now())


def clip(clip_id):
    row = _CLIPS.get(clip_id)
    return copy.deepcopy(row) if row is not None else None


def clips_in(state, limit=None):
    rows = [copy.deepcopy(r) for k, r in sorted(_CLIPS.items()) if r.get("state") == state]
    return rows[:int(limit)] if limit else rows


def clips_for_source(source_id):
    return [copy.deepcopy(r) for r in sorted(_CLIPS.values(),
                                             key=lambda c: (c["start_s"] or 0.0, c["clip_id"]))
            if r["source_id"] == source_id]


# ---- events ----
def log_event(level, stage, message, source_id=None, clip_id=None):
    _EVENTS.append({"ts": round(_now(), 3), "level": level, "stage": stage,
                    "source_id": source_id, "clip_id": clip_id, "message": message})


def events():
    return list(_EVENTS)
