"""The hook never refuses a render. Built from hook-placement refusals on the gamestack v3 header
(rows 0-200), each with the text bands of its own error string: "best y=100 still covers N px of
text bands [...]". The header check counted text in the game band under the header (the hook
block, padded by TEXT_PAD 24, reaches row 210); only one case has text inside the header itself
(rows 0-132). The hooks are synthetic titles of the same length (two lines at the header's size,
as the refused ones were).
"""
import pytest

from clipengine.render import captions

HEADER = (0, 200)
# (case, hook, covered px in the error string, text bands in the error string)
REFUSALS = [
    ("under_1", "Why he can't wear the new shoes today", 5,
     [(205, 233), (280, 362), (997, 1056), (1186, 1218), (1343, 1360), (1388, 1418), (1872, 1874)]),
    ("under_2", "The kitted car already got taken away", 10, [(200, 288), (318, 558), (585, 640)]),
    ("header_text", "Viewer says the host ignored the sub", 132, [(0, 132), (1408, 1422)]),
    ("edge_196", "He hates bags, but this one is carbon?", 14, [(196, 490), (900, 926)]),
    ("under_3", "A love song cut short by a surprise", 5, [(205, 565)]),
    ("under_4", "Asking a rival shop for its suppliers", 10,
     [(200, 210), (400, 406), (470, 504), (547, 602), (1259, 1304), (1336, 1361), (1408, 1422),
      (1872, 1874)]),
    ("edge_198", "The greatest driving stunt ever made?", 12, [(198, 218), (1394, 1422)]),
    ("under_5", "She scored the same goal nine times", 10, [(200, 412), (1408, 1422), (1872, 1876)]),
    ("under_6", "Holding off a dog and a goose at once", 8, [(202, 218), (885, 907), (1347, 1416)]),
    ("under_7", "Following the music led right to her", 8, [(202, 218), (885, 908), (1346, 1416)]),
    ("under_8", "Ten nominations and only one win", 6, [(204, 409), (1416, 1422), (1552, 1603)]),
]
IDS = [r[0] for r in REFUSALS]


def _block(line, size):
    y = int(line.split("\\pos(540,")[1].split(")")[0])
    h = (line.count("\\N") + 1) * size * captions.HOOK_LINE_H
    return y, h


def _first_choice(hook):
    """The header hook the old code placed (largest size of <= 2 lines inside the header)."""
    text = " ".join(hook.split()).upper()
    room = HEADER[1] - HEADER[0] - 2 * captions.HOOK_BOX_PAD
    fits = [f for f in captions.hook_fits(text, max_lines=captions.HOOK_BOX_LINES)
            if len(f[0]) * f[1] * captions.HOOK_LINE_H <= room]
    lines, size = fits[0] if fits else captions.wrap_hook(text, max_lines=captions.HOOK_BOX_LINES)
    return lines, size


@pytest.mark.parametrize("clip,hook,covered,bands", REFUSALS, ids=IDS)
def test_each_refusal_reproduces_on_the_old_rule(clip, hook, covered, bands):
    lines, size = _first_choice(hook)
    old = captions.band_overlap(100, len(lines) * size * captions.HOOK_LINE_H, bands,
                                captions.TEXT_PAD)
    assert abs(old - covered) <= 1, (clip, old)


@pytest.mark.parametrize("clip,hook,covered,bands", REFUSALS, ids=IDS)
def test_the_header_hook_never_refuses_and_covers_no_header_text(clip, hook, covered, bands):
    info = {}
    line, size = captions.hook_event(hook, "gamestack", 21.5, box=HEADER, text_avoid=bands,
                                     info=info)
    assert line, clip
    y, h = _block(line, size)
    assert HEADER[0] + captions.HOOK_BOX_PAD <= y - h / 2 and y + h / 2 <= HEADER[1] - captions.HOOK_BOX_PAD
    inside = captions._inside(bands, HEADER)
    if info["fallback"] == "plate":
        assert clip == "header_text" and "\\p1" in info["plate"]   # the only one with header text
    else:
        assert captions.band_overlap(y, h, inside, captions.TEXT_PAD) == 0, clip
    assert ",0:00:00.00,0:00:21.50,Hook," in line and "\\fad(120,0)" in line


def test_text_in_the_band_under_the_header_is_not_the_headers():
    # every case whose text sits at row >= 200: the first choice stands, unchanged (2 lines,
    # centred)
    for clip, hook, _c, bands in REFUSALS:
        if min(a for a, _b in bands) < HEADER[1]:
            continue
        info = {}
        line, size = captions.hook_event(hook, "gamestack", 21.5, box=HEADER, text_avoid=bands,
                                         info=info)
        lines, first = _first_choice(hook)
        assert info["fallback"] is None and size == first and "\\pos(540,100)" in line, clip
        assert line.count("\\N") + 1 == len(lines)


def test_text_just_inside_the_header_edge_moves_or_shrinks_the_hook():
    # edge_196 (196-490) and edge_198 (198-218): 2-4 rows of the band's text box reach into the
    # header
    for clip, hook, _c, bands in REFUSALS:
        if clip not in ("edge_196", "edge_198"):
            continue
        info = {}
        line, size = captions.hook_event(hook, "gamestack", 21.5, box=HEADER, text_avoid=bands,
                                         info=info)
        assert info["fallback"] in ("one_line", "smaller") and info["covered_px"] > 0, clip
        y, h = _block(line, size)
        assert y + h / 2 + captions.TEXT_PAD <= min(a for a, _b in bands)


def test_header_text_everywhere_gets_a_plate_off_the_frame_edges(tmp_path):
    _clip, hook, _c, bands = next(r for r in REFUSALS if r[0] == "header_text")
    info = {}
    out = captions.build_ass([], tmp_path / "c.ass", hook=hook, layout="gamestack", dur=21.5,
                             text_avoid=bands, hook_box=HEADER, hook_info=info)
    assert info["fallback"] == "plate" and info["covered_px"] == 132
    ev = [l for l in out.read_text().splitlines() if l.startswith("Dialogue:")]
    plate = [l for l in ev if "\\p1" in l]
    hook_ev = [l for l in ev if ",Hook," in l and "\\p1" not in l]
    assert len(plate) == 1 and len(hook_ev) == 1
    assert plate[0].startswith("Dialogue: 0,") and hook_ev[0].startswith("Dialogue: 1,")
    assert ",0:00:00.00,0:00:21.50,Hook," in plate[0]      # under the hook, as long as it
    x0, y0 = map(int, plate[0].split("\\pos(")[1].split(")")[0].split(","))
    w, _l, _w, h = map(int, plate[0].split("}m 0 0 l ")[1].split("{")[0].split(" ")[:4])
    assert x0 >= captions.HOOK_MARGIN - captions.HOOK_PLATE_PAD and x0 + w == captions.PLAY_W - x0
    assert y0 >= HEADER[0] + captions.HOOK_BOX_PAD / 2
    assert y0 + h <= HEADER[1] - captions.HOOK_BOX_PAD / 2


def test_a_header_too_small_for_any_hook_drops_it():
    info = {}
    line, _ = captions.hook_event("Viewer says the host ignored the sub", "gamestack", 21.5,
                                  box=(0, 60), text_avoid=[(0, 40)], info=info)
    assert line is None and info["fallback"] == "none"


# ---------------------------------------------------------------- the free-placement path
OLD_FREE = [   # three older refusals (track / hudband renders), their error strings' text bands
    ("free_1", "The leader's plan: push the whole table", [(0, 35), (73, 1130), (1490, 1823)]),
    ("free_2", "The word she never wanted to say",
     [(320, 322), (400, 408), (612, 622), (686, 718), (748, 854), (912, 934), (982, 1002)]),
    ("free_3", "The pick everyone hates always wins",
     [(34, 104), (132, 146), (360, 436), (622, 650), (718, 732), (771, 807), (843, 1002),
      (1038, 1093), (1147, 1167), (1220, 1447), (1498, 1614), (1826, 1902)]),
]


@pytest.mark.parametrize("clip,hook,bands", OLD_FREE, ids=[r[0] for r in OLD_FREE])
def test_free_placement_never_refuses_and_never_covers_text(clip, hook, bands):
    info = {}
    line, size = captions.hook_event(hook, "track", 20.0, text_avoid=bands, info=info)
    if line is None:
        assert info["fallback"] == "none" and info["covered_px"] > 0
    else:
        y, h = _block(line, size)
        assert captions.band_overlap(y, h, bands, captions.TEXT_PAD) == 0
        assert info["fallback"] in (None, "one_line", "smaller")
    assert info["plate"] is None                     # no plate off a header / top band
