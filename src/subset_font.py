"""Subset LXGW WenKai (SIL OFL 1.1) to the glyphs this video needs.

The subset is renamed to "PaperHand Subset" as required by the OFL for
modified versions (the original Reserved Font Names are not used).

Usage:
    uv run --group dev python src/subset_font.py LXGWWenKai-Regular.ttf subtitles.srt
"""

import sys
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "fonts" / "PaperHand-Subset.ttf"
EXTRA = "好想爱这个世界啊华晨宇·补全结尾版 0123456789,.!?，。！？、：；“”‘’（）…—-"


def main() -> None:
    src, srt = sys.argv[1], sys.argv[2]
    text = Path(srt).read_text(encoding="utf-8-sig") + EXTRA
    font = TTFont(src)
    opts = subset.Options()
    opts.name_IDs = []
    opts.notdef_outline = True
    sub = subset.Subsetter(opts)
    sub.populate(text=text)
    sub.subset(font)
    name = font["name"]
    for rec_id, value in ((1, "PaperHand Subset"), (2, "Regular"), (3, "PaperHand Subset Regular"),
                          (4, "PaperHand Subset Regular"), (6, "PaperHandSubset-Regular")):
        name.setName(value, rec_id, 3, 1, 0x409)
    font.save(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
