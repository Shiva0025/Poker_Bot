#!/usr/bin/env python3
"""Pokerbots: Discard Hold'em bot.

Single-file bot: exact table-driven hand evaluator, offline-computed 3-card
preflop equity tables, range-weighted Monte Carlo postflop equity with
discard inference, adaptive opponent model and EV-driven bet selection.
All protocol output goes to stdout; diagnostics go to stderr only.
"""
import sys
import json
import math
import random
import time
import zlib
import base64
import array

_T0 = time.time()

# ============================================================ evaluator
# Cards are ints r*4+s (r: 0=2 .. 12=A, s: 0..3 = s h d c).
RANKS = "23456789TJQKA"
SUITS = "shdc"


def c2i(s):
    return RANKS.index(s[0]) * 4 + SUITS.index(s[1])


I2C = [RANKS[c >> 2] + SUITS[c & 3] for c in range(52)]
KEY = [(1 << (3 * (c >> 2))) | (1 << (39 + 3 * (c & 3))) for c in range(52)]
BIT = [1 << (c >> 2) for c in range(52)]
M39 = (1 << 39) - 1

FSUIT = [-1] * 4096
for _k in range(4096):
    for _s in range(4):
        if (_k >> (3 * _s)) & 7 >= 5:
            FSUIT[_k] = _s

STRH = [-1] * 8192
for _m in range(8192):
    for _hi in range(12, 3, -1):
        _w = 31 << (_hi - 4)
        if _m & _w == _w:
            STRH[_m] = _hi
            break
    else:
        if _m & 0b1000000001111 == 0b1000000001111:
            STRH[_m] = 3

# number of ranks that would complete a straight (only when no straight yet)
SOUTS = [0] * 8192
for _m in range(8192):
    if STRH[_m] < 0:
        n = 0
        for _x in range(13):
            if not (_m >> _x) & 1 and STRH[_m | (1 << _x)] >= 0:
                n += 1
        SOUTS[_m] = n


def _enc(cat, lst):
    v = cat
    for r in lst:
        v = (v << 4) | r
    return v << (4 * (5 - len(lst)))


FV = [0] * 8192
for _m in range(8192):
    if bin(_m).count("1") >= 5:
        _sh = STRH[_m]
        if _sh >= 0:
            FV[_m] = _enc(8, [_sh])
        else:
            FV[_m] = _enc(5, [r for r in range(12, -1, -1) if (_m >> r) & 1][:5])


def _nonflush(k):
    quads = []
    trips = []
    pairs = []
    present = []
    mask = 0
    for r in range(12, -1, -1):
        c = (k >> (3 * r)) & 7
        if c:
            mask |= 1 << r
            present.append(r)
            if c >= 4:
                quads.append(r)
            elif c == 3:
                trips.append(r)
            elif c == 2:
                pairs.append(r)
    if quads:
        q = quads[0]
        return _enc(7, [q] + [r for r in present if r != q][:1])
    if trips and (len(trips) > 1 or pairs):
        t = trips[0]
        return _enc(6, [t, max(trips[1:] + pairs)])
    sh = STRH[mask]
    if sh >= 0:
        return _enc(4, [sh])
    if trips:
        t = trips[0]
        return _enc(3, [t] + [r for r in present if r != t][:2])
    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        return _enc(2, [p1, p2] + [r for r in present if r != p1 and r != p2][:1])
    if pairs:
        p = pairs[0]
        return _enc(1, [p] + [r for r in present if r != p][:3])
    return _enc(0, present[:5])


class _RankTable(dict):
    def __missing__(self, k):
        v = _nonflush(k)
        self[k] = v
        return v


RT = _RankTable()


def evaluate(cards):
    k = 0
    for c in cards:
        k += KEY[c]
    s = FSUIT[k >> 39]
    if s < 0:
        return RT[k & M39]
    m = 0
    for c in cards:
        if c & 3 == s:
            m |= BIT[c]
    return FV[m]


def ev_key(k, cards):
    """Evaluate given precomputed key sum of the same cards."""
    s = FSUIT[k >> 39]
    if s < 0:
        return RT[k & M39]
    m = 0
    for c in cards:
        if c & 3 == s:
            m |= BIT[c]
    return FV[m]


def _fill_rank_table(sizes=(7,), deadline=None):
    import time as _t
    out = []

    def rec(r, left, k):
        if left == 0:
            out.append(k)
            return
        if r < 0:
            return
        for c in range(min(4, left), -1, -1):
            rec(r - 1, left - c, k + c * (1 << (3 * r)))

    for n in sizes:
        out.clear()
        rec(12, n, 0)
        for i, k in enumerate(out):
            if k not in RT:
                RT[k] = _nonflush(k)
            if deadline is not None and (i & 1023) == 0 and _t.time() > deadline:
                return


# made-hand + draw strength proxy in [0,1]; used for ranges and opponent discards
_CAT_BASE = (0.10, 0.38, 0.66, 0.78, 0.86, 0.90, 0.95, 0.99, 1.0)
_CAT_SPAN = (0.12, 0.25, 0.12, 0.08, 0.04, 0.04, 0.03, 0.0, 0.0)


def hscore(a, b, board, bk, nboard):
    k = bk + KEY[a] + KEY[b]
    sk = k >> 39
    s = FSUIT[sk]
    if s < 0:
        v = RT[k & M39]
    else:
        m = BIT[a] if a & 3 == s else 0
        if b & 3 == s:
            m |= BIT[b]
        for c in board:
            if c & 3 == s:
                m |= BIT[c]
        v = FV[m]
    cat = v >> 20
    sc = _CAT_BASE[cat] + _CAT_SPAN[cat] * ((v >> 16) & 15) / 12.0
    ra = a >> 2
    rb = b >> 2
    sc += 0.02 * (ra if ra > rb else rb) / 12.0
    if nboard < 5 and cat < 5:
        fl = nboard == 3
        sa = a & 3
        sb = b & 3
        if (sk >> (3 * sa)) & 7 == 4 or (sb != sa and (sk >> (3 * sb)) & 7 == 4):
            sc += 0.16 if fl else 0.09
        elif fl and sa == sb and (sk >> (3 * sa)) & 7 == 3:
            sc += 0.03
        if cat < 4:
            mask = BIT[a] | BIT[b]
            for c in board:
                mask |= BIT[c]
            o = SOUTS[mask]
            if o >= 2:
                sc += 0.12 if fl else 0.07
            elif o == 1:
                sc += 0.06 if fl else 0.035
    return sc


def best_keep(h3, flop, fk):
    """Heuristic keep for a 3-card hand on a flop: returns (score, discard_index)."""
    x, y, z = h3
    s0 = hscore(y, z, flop, fk, 3)
    s1 = hscore(x, z, flop, fk, 3)
    s2 = hscore(x, y, flop, fk, 3)
    if s0 >= s1 and s0 >= s2:
        return s0, 0
    if s1 >= s2:
        return s1, 1
    return s2, 2


def combo_index(a, b, c):
    # a,b,c distinct ints -> index into 22100 combos
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    return c * (c - 1) * (c - 2) // 6 + b * (b - 1) // 2 + a


# Preflop equity of every 3-card combo vs top 100/50/25/12% of 3-card hands,
# computed offline by Monte Carlo with heuristic discards (x10000, zlib+base64).
_PF_BLOB = (
    "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"
    "SmfbeM42oM8agxlyVuaVzoUunAuYHWdl9uiMG8gZd2OIkcecw2wrkhmnc67UYl7prOvIWTeQsw7oU2oxIzDmmBEY83Ec80kcc8wG"
    "1LfmzFvPmbeaM69JhN9uKTNP595/c+7lc+61lnmls681Z996zr7VnH1An9YRZoDMLI7tFY7tdo7tNY4tRr+LzBadgRj9VTJbdAZi"
    "9LfLbMHse1lmoM5BzD7MwYGcg5h9P8gMxLzSWbiDs7C2r7NwIGchZjRmIdBnhMyqyTKvjsqFWYULs+otmVcz5MKswnWOM/EoZ+JX"
    "nIlfcibO4Ex8izNxWNSGs/AIZ2IljuEzP6iclZh9ypLbZFbqTMSsUt7r7ynrrUuO59zcYfpwdo7n7MSsU/a8lmzE+YnZqTMRs0f5"
    "Lo1819dXtjvrdeQcHcg5emOYxlnakbN0IGcpZpsy3U6yapGnrNpcZqnORMwS5bnx5Lmp5Ll1ZLlfkOXGk+WmkuXWuTnUmLO1NWfr"
    "es7W1ZytmGX6DC3JtneTbf9Mts2T2aozUeYiZ8MVzobtnA3XOBuU3bqQ3T4nu60iu31Hdtvu5swqztodnLXKbavIbd+R27bL7FIW"
    "fpksfI4svJssXEEW/kFm7Qji22Ti21ux4tv1KcW3t4hvh2LFt5qpc5y7Rzl3v+Lc/ZJzdwbn7lucu8Oic5y9Rzl7v+Ls/ZKzdwZn"
    "71ucvcOiYdEGf4M/WK7Bfi3vVrEiLy/5H842B0XefyTz5K6qwT2+4b6qMV5CrMibLLbaO51s7uwzL8/bm9ybzLWrvebJ01KyN4my"
    "XLvToAyln0mbPA9lO802T891unA3aqRfIXbS+7lfxavidF07v5dbA3QK8vwZ3gyvJDzpSlHX3JWjrlMww0NdSXiL1KG2ufi081GH"
    "WtTdEl71xviwB/yzYm+78YF94k8PYPP8T7zfOzvv1/NLvVJvur/W/60zP1jod3I2KZrn/977RFqUemgz3dWulRad/IXSIpI2aHVe"
    "YtTzp0uMtdIGrdAmii74jwewGsEYsfSgUOz+YI/Yfwdvik33h/p1nd0aNAgecE8+Org1mOJP8fOD9cEB/4D/TnA0WOOv8efH012r"
    "odLyAR9tR7t2aJvv2qHtO64d2s6PF0tbtL5VYjYIRkvMW4N8ibk+eEdiHg3QdnHc2/0W7LXgsNjBYIKBFQTNDGxdoCu7wiDHwPoH"
    "cYDsKAwWBDh9GuXe74R/wl8aLAyqOVsWrA72+fv8xuZXpoF7r3Gp/q49fApde/iMcu3hs9S1Xige+3z4NHbt4TMu9an4wKtQfmNB"
    "MEp+Y0ywVH5jofjAq7H8xq8MfD5NVc54XHlJnde3ejr7Nwf/kdT5X9XoPeY87qua00mdv3mezt+Ep1kwmVmQa5snNQ/2Mg92Gi3D"
    "nN/LTEDZTvNzX+dpO87TPM7TE55mQwtfs6GC2VASVvE0H3oxH2YwH24JtQ5zvhczYgYzAnW3hJ94Oh/r+Tof13I+LuR8HONrVpxl"
    "VvzArJgeaFZMin7vaV6UMi9+62tedGJeRJG2wZwvZWb81tfM6MTMQJsoGurrvGsQ6Ly7lfNuPefdUc67xwPNjjHMjkJmxx5mx5vM"
    "jvlxXV/z4wHmxxTmxwHmxxrmx+JY22LOP8AMmcIMOcAMWcMMQdvFcRzo/FrA+TWG82sh59dqzq9fGZ1f6wPNksPMkglGs6QZs2RX"
    "oFmSYzRLxqVW+Zonh5knJ5gn1QLNk32+5kmDQPPk05T6YM4fZqacYKZUCzRT4IM53yDQTIHPp6mPnHnpXvqJ1InUDGc23abf7Kwi"
    "VZH62tl+Z79K/1V6prNazhalFqVapLdIR5YoO2zwkRWVLIDPqgYZU0uyBrmimQN22BwgCzDvkQWY98gAfN/L7NlpkDlGskfzJ9ci"
    "VzSD9jKDwA65FrP+pOSKzu88zm/MeJT3YhbNYBbdEp5gHiGDukgWaR6VSK5oJvViJs1gJoEdSkLMcsxjzHLM47Wcxws5jzHDUV/K"
    "bMIMny6ZoNkURWOYT2eZTz8wn5BJyKdJkiuaUaXMqN/6mlGdmFFgh0kRZvV0yRWdr7dyvq7nfD3K+YoZPV0yQLNqCrPqALNqDbNq"
    "cYyMQl6NYV4VMq/2MK/eZF7Nl1zRzHqAmTWFmXWAmbWGmQV2mB9jFmNeLuC8HMN5uZDzcjXnJWYx5iVmcH+Z+ZpdJ5hdmMFLZeZr"
    "dmEGN5aZv575dZj5hcxCfjVjfiGz1kl2aX6Nk1zRDDvMDDvBDKsWaIYhe5FhDQLNMLDDOMkUXMiUE5IrN0u2IFMqJFd+lY4LmZIp"
    "udJCLnjBD17wgxf84AU/eMGvFrOsBbPsg3gwM2wDs6ySd/BZ1VRmHDJLFVotyTjNMmSMaq7PqLhWU3F95vKnOTNvLzMPGaXKzVC5"
    "FUnmaZYhM9oxM/KYGaq0Tnq9mH8zmH+3hFWYgb2YgTOYgcgkVXTbPFV0XajoRkoGapYhA1RjnafGmu4vZAaowppHhXWeCmu634l5"
    "GEWqr+ZRX52nvprud2ImIoNU6V31zjIXf2AuqtL7xEcmapZhpjcgH93Kmb6eM/0oZ7oqq+n+A8zHKczHA8zHNczHxXFdZuQDzMgp"
    "zMgDzMg1zEhkjirAC/4Y5mQhc3IPc/JN5iQyUrMMM3oBZ/QYzuiFnNGrOaNVUzU2qqj6U1EVUlGNoqJaSkW1LFBF1djNf9VT/amn"
    "CqmnRlFPLaWeWhaonmosGaPKsDeV4WtUhgepDAuoDNdRGRYGyMxx5KOPyEcnUspHNl35qIJ8tD+lfJSZrny0iLn5EXPzREpz06Zr"
    "blYwN/enajE/WzA/P4g9Zqj+Hjw1Q/X3Kvh78NQMbcEM/SD+IG4ftA/e83G95yf8Ax5smbct+aKzhFnmvZjcJndHA9zjG+6PBm94"
    "n4rl+a3EEv5w7yZn8/wj3kJnt9iEf5M3XEoWekekpL6UoRRlKEVZfXvUqytzdbO/U+ze4Dqxev5at5Yf6+12c3+Es38LrvkdnK0I"
    "6/ljvbVSM0Lq/i3o4F+TmpelDrW7xaeT1KEWdS+H5f5/ia320wPYyKCD2Ean/GFv+639z73Pvb3++/7jzpoFX7i8W+iPDCYGx/xj"
    "/qrobVePNntd7fvSYqH/hbQ45qPNqmi6tEGrvRLjfWmDViMlxsQAbaZHJtCV9dXgJbHJQX0De8i0FqvHz97+//rPO5sdlLk+2uwP"
    "CWbJPE6ZoqDA2RPmfmGJRXFv1+p/peVmv0xaHvbRNuVma5G03BWg7aI4J9VbYv6vtEXrIRJzVoC2aP2ExLzfoG1OqpbJElsaHBPr"
    "bgrFvg5+Z2APmI/FxpgtYsuCxg4ZRgdfBi1c5rwW1HLvhbjtTHv5M0XF5kHzbfBtMNhMNFeCK0FZaplr3Vg8XhOfWq49fNq51u3F"
    "49sAPoNde/iUpb4TH3h9Kb/RQnzg1U5+o734wGuw/MZEA5/vUu+aCrFM84nYy2a/WI59Wew2q9/nyp/K+plN2VvFkiYyrzq719Qw"
    "PwQ/BCfNMJNytt2V9XS23JxyNTVMVft3U9fZI/aMGeKsSUbStYjE84cAvied1zDx7Cm+y53XKfGsa/4unkPMGfGsK77wvld+s4Y5"
    "Kb85THzhvVx+85T4whu+8IZv3YzK7Ma1Lak5fMDTTE+YF5Oa60cDvUd+4/5oMNzTXD3CXP3U04xv5WvG32Jv8jTnFzLn61stQ34v"
    "ZNajrL5d62lOdvI1J68xJ+v6mvk7mfnXBZr5K8Kxnub+CF9zvwNz/+VQ65DfI3zN/g7MftS9HLb2NffeZ+59wdybGGju/ZevCJAe"
    "KAJ0IAIcJAKsij73FAMe9xUDFhIDjhEDpkfaBvn9uK8osJAocIwogDbTo//1NcfKmGOzAs2xIubY/UZzLDNQJHiJSFDfKBK0/hck"
    "WBQ/7ysWbCYWHCYWFASKBbuIBTkpbYv83kw0OEw0KAgUDXYRDdA2J9U40FxqwVyqbzSX2jOXHmQuTWQuZRlFhGOBIkKhUUT4HRHh"
    "YyLCFiJCWWp0oJjwGjGhgJhQxygmfBsoJlwhJnyXUh/k92tEhQKiQh2jqAAf5PcVogJ8vktFRnOmBnNmGHPmVebMKebM35kzZ5gz"
    "FUaR4RMiw34iw8tWkWG/UWT4mVVkuJXI0CTjVaPY8EOg2JAyig09iQ01iA11iQ1DiA11M9QX+Q1f5HfKKDr0JDrUIDrUJToMITrA"
    "t25GV2f5zh501s1ZM2evOtvlbKqzN51Nc/ays+bORjub6ayLs1ecARmU/dsHQIJKlsfn0UBQwgdSAB8ULcD+CYPMR64fYa4j6/F9"
    "IRGjvv2UmAG0yBPEAD4oaiwkaoD9b7HIdOQ0Mn234IPmNLIc5cjy3YICihwvh3WJHTuJHUCNewU5gA+KHiN8RY8ORA+w/4oQmY3c"
    "fZ+5+wVzF5mN3EVWvy3ZrwiykAhyjAgyPfovYgjQAxjSgRhykBiySvBBUeRxX1FkIVHkGFEE7L8qQib3FnzQHEUmI0eLmKPIZOQo"
    "sri3ZL0iyWEiCbI4JTpUkSQnlUkseYlYAhQBlrT+FyxZFAMfFE02E00OE00KAkWTXUQTsP+iGJm7TPBBcxGZi1xsz1x8kLk4kbmI"
    "rF0m2a6IUkBEQda2k2xXRLlCRPkuBTSpJYiimAI06S6IopjyMTFlCzGlTPBBUeU1okoBUaWOUVQBYgFVrhBVwP5lKWRqUjJLc24Y"
    "c+5V5twp5tzfmXNnmHPI0qRkuSILsvSkZLkiSw0iS10iyxAiS92MCmLLJ8SW/cQWoEqOIItiC1AF2HIrsaWJ4IOiC5AM6JIyii49"
    "iS41iC51iS5DiC5g/yaCDl0FH7rJBXRoJvgwVS6gw5uCD83lAjqMFnx4Ra58Ikw3IsyrRJipRJhpRJjmRJiZRJhXiDA90oEtijKK"
    "MJU6A59Hg0q0Aaro6iPhA20UYYAWR4gWuppIcDUxz69vbyLqLCTqAE10VfKGp6uSPEEdRRiggq4jdvvXiAq6iqjHVcRuvwOx5+VQ"
    "1xD1uIbY7Xcg+gBFdLVy1NtJ/NHVyr2CPoowyP73mf1fMPt1/TAy0NXD21w97PUXEoOOEYOmR7p2eJtrh73+QqLQMaIQ0ENXMeVc"
    "xaz2OxCHDhKHgEKKMMjyMma5rhuGBEXMcl03PGF01dDb30wsOkws0lVDiquGJxwW6Jqht7+ZaHSYaKRrhhTXDE8IaujqxnB1c5Wr"
    "m8lB63/BI6CRIgyyuQWzWdcLtbheaMf1QjHXC4ONrhaWcbXwZVBATNLVQjuuFoq5Whjscn80Uek1olIBUUnXCu24VijmWmGwoIWu"
    "emoZXfUsDXTV0938jrj0MXFpC3EJqKQIg6ytwawdxqx9lVl7iln7d2btGWatrhKSXCXcy1XCSa4StnOVsNzUJTYNITbVzXiV6KRr"
    "hHu5RjjJNcJ2rhGWm7pEpyFEJ6CErobe5Wook6uhl42uhnK4GrrN6mporrmV+AR0akL90ZX640Hqj2bUH7uoP96k/niZ+mM09UcX"
    "4lNX4tODxKdmxKddxKc3iU8vE59GE5+6OITJJ0p1I0q9SpSaSpSaRpRqTpSaSZR6hSjVI71Hem6QG3zs4/rYz5Bdt9/6xf6r3j3O"
    "xppi/x7vVbn7OsA9vuH+6+B9v79Ysf/rAFbix35Xr6v3jV/T7+dslC1x31H2jftWU0r6SRlKv5E2NaWsn73qEAZWIn/rrk3QOdA/"
    "UVHqPyy/cVcw1b/H2eSgbdDa2bKw1JU+LDX3+FOlprXULQvXSx1qUYfayeLTVurWh+f8dgGsZtBM7LGgj9hNsrtzNPjIvX8NZ+nB"
    "TP8NZ5OCGoJyz5u/B7sdHk6OPnK1v5UWb/gzpcV10uZ5s9tHm8nRYGmDVmiDVpMkRg1pg1ZoMzj6bfCu2CqXm7Crwa8M7FMzX+xJ"
    "o38apFOw1J/mrDhYGNzjbIjJDUY5W2DOBJ2cbXfz+3hwPFgcdwqm+Uul5T3SdogZFeRKy07BGWl5PEDbxfHn0hatiyXmQmmL1miL"
    "1tsl5nKDtp/Hax26wR5yGAd7zPxDrLVRpqkwgy3sT3aI2DbXs7c7m2gWOFTeGDxsDsufM0izHU01Z7faPeacs2r2osl31jx9m2v9"
    "mHhsDBaIR1XxSbPVTEfxOGf2iEe+uSgenvjACz7wgg+84AMv+MALPvCCj5e+3txlYV+Y82LZ9hWx2+1isQfsO2I7bf0QdkD+dtLt"
    "4aumpdjHbmSGOUva2Wass9n2I3O9vd4+Yhvb2Nk0O0P+HESFi5FwVi3j/zyHiW/SOi/xvF58H3FejcVzp4FvhfN6Rzzr/cQX3vCF"
    "N3zh/Yj8ZmPxhXeF/OY74lsv4yb7jdg++xexz/hOpfI3X9qHt4bPiV2xXUJY9XCkWIewv9g+1//G2QI7zz7vbKRdZRs4e9FFGu3s"
    "O3vAFjobG56z7Zxlh1vteGejwrvD1Xa1rZe5zxibJhGelxgjnfcqiTBaYnznvA9IhHb2nEQYb7dKhNUWMeplDpUYiLJAnmGexECU"
    "F+UZPpMYiIIYiIIYiDJKnuHuEDGGZlYiHa5XPcWz3xL1xpp7PMW9rwO9B9bh/usg9hW3ahK3+vuKfr8OFP1G2a6e4l8/n/hntQxY"
    "188nAlqg3cO+4tNU4lPbQPHpmq8o2CZQFJxMFFwW9vcVB+8hDrYOFAfXh1oHrLuHSNg6UCRE3frwt77i0EziUI1AcejvxKF2gaJh"
    "M6JhH6LhUaLh5KiGr3j4BvHwukDxcLeveDg40jbAujeIiNcFiohoA6xDm8HRUl/xZmGgeJNLvDlDvFluFG/eDRQVlxEVf2UUFecT"
    "FdcQFRfH03zFxXsCxcVRxMVOxMXjxMXPY20LrLsnUGQcRWTsRGQ8TmRE28/jxwLFlQXElcPElY5GcWUPceUiceV+o+j4BNHxH0TH"
    "W6yi42Ci4xCiY/P02wPFx43Ex6pG8bEa8fEc8TGf+Oilqw+wbiMREj7AumpEyHNEyHwiJHy89JZG8eNJ4sds4sdHxI/GVvFjBvHj"
    "HeLHXVZR8rxRlHzFKkouJkq+Q5SsHypK3k6UrJZR+YvDiJNjiZPXW8XJmDi50yhOJqziZL2Mn/oC68YSKeELrIuJlPAF1iWsIiV8"
    "62WkWcWJecSJVcSJz4gTB4gT54gTW4kTd4eKE99YRcu/EC3rh4qW7YmWzxEtuxAtRxIt+xMt62Uaq3j5PPGyAfFyNPGykHjZjng5"
    "nni5mng5NFNjAOueJ2I2IGKOJmIWEjHbETHHEzFXEzERY2hmsbNXnG10ts1ZjrO+zuY4O+zsWWc7nf3N2Txn052dcPaus19m/TJr"
    "Q+aGzA+dAS1VHeYGQMdKFYjPrwMgZ4agJzBTERTqcKwBGgL/ahL/gIQlgpBEUdufOAoEBY6OEsxUJO3nE0ktUBPoVyqYqTgH9APO"
    "AflKBRkVTYF8kwUZgaRXBU0VTycTT5cJZiqi3kNEbR0ookIdLguBdh8JZiqeAe0mCWYqngHpPhJEVFQF0k0SRFRUHRwBUYGrzYir"
    "fYirR4mrkwUzFVnfILJeFyiyApWBrFCHkyOgG3AL6FYsmKm4dYa4BXQDbgHZ0A7IVixIqOjaieh6nOj6efwu8XUZ8RXICnydT3xd"
    "Q3xdLJipCHtPoAg7igjbiQh7nAgLdbg4BpptE8xUfDpMfOpoFJ/2EJ8uEp+AZNsEARVlgWRoX40oe44om0+U9dLvJ84+QZz9B3EW"
    "CAucHUycHUKcbS6YqUi7kUgLFH9YMFOR9hyRNp9IC3XYPB0I9KpgpuLQbOLQR8ShxlZxaAZx6B3iUKXfMKLtWKItkAt+MdEWyDVN"
    "kE/Rtl7GXcRbIO0XgraKt4uJt+8Qb4FGwNvbibfVBDMrf1URdywRF+g+WzBTERfoDsRNWEVcqMNqGUAr4M084s0q4s1nxJsDxJtz"
    "xJutxBugFfAGSAX/54m6DYi6o4m6hUTddkTd8UTd1UTdoZnfEHf/QtzFOwJ32xN3nyPudiHujiTu9ifu1hPMVOR9nsjbgMg7mshb"
    "SORtR+QdT+RdTeSFOqwniFksmLlNLiBmjmDmYbmAmM8KZs6TC4g5XTDzl1m4gJi4XiH6biP69iX6Hib67iT6ziP6niD6IgriIAri"
    "fJYB3FUEVvSt1KX4/DqoRGIgrq7cMwSJFX2BpDWJpLoSL+FK3OGv1XV4Cdfh3wjS6or+fa7oi30gsqIvEHMqEVPX4JMDXYGX+vcQ"
    "l3UFPtnhqK6/S/17iMy6/p4sCKsr/atc6Zf4k4nNQGZFXyDjTCKjrr0nBX8nMurK+yP/DeKzrrwnBbryft4MjmoQod8gQuu6e1Kg"
    "6260UIQGRjcjRvchRh8lRgOhFX2BgLrmLg5yiYBniIC65t5udMXdiSvu4mAUcboTcfo4cfrzWNfbnbjeLg5GEak7EamPE6mBqLoz"
    "8FvuDKzizsDVYD6xeg2xerH8aW5da28LFhDpDhPpdK2dxrX2rVxrV7O60t4WbCRe60r7YVONeH2OeJ1PvPbSdZ29LdhIxNZ19sOm"
    "GhH7HBE7n4gNJNUdg7XBE8TsfxCzdcegNXcMKswQYjYQW9EXiPYkEW02Ee0jIpqusR/hGnsa19gVttJvGHF7LHFbV9izucJ+hCvs"
    "aVxhVzgc/D9PRe6xRG5dX8/m+voRrq+ncX1dIQiqOwnruZPwhdGdhGzuJNzOnYQHuOreyZ2EAxbIregL5JpH5FpF5PqMyHWAyHWO"
    "yLWVyKVr61Ghrqz3meeJ3w2I36OJ34XE73bE7/HE79XE76GZhgj+PBG8ARF8NBG8kAjejgg+ngi+mggO5NQdhpu4w7CP7/oZdxhK"
    "7XPE8C7E8JHE8P7EcCB4PerVYurVjdSrOdSrc6hXn6Ve/Rv16nTq1XepVzcQw4uJ4RuJ4TnE8DnE8GeJ4X8jhk8nhr9LDN/g0PcV"
    "Ivk2InlfIvlhIvlOIvk8IvkJIrk+z7t8HkT6LGNnsDOoI1edoMBfK/aa/738fdUfg9fc/+P+x6CBeU3+ptT3Hu4bmF7yr4l0doix"
    "VaxG8KScj0wJ1shZxFVbI3jcf1JKNvtrpGSZlKF0ivz5izVStsw+EPzcwAqCDANLmbZiu4INfjtno4NT+i+LmZnBG87mhbuCdv4G"
    "qTnsn5KaN4KZUjNI6lA7Ws5CTkkdalE3KDzlMBvW3TQUe908LrbXvCNWEOT5l5z9GFz0pwXTHKLsCJY4m27OiTo7GxUEl/w8aTFN"
    "2kw0S4Id0qKatDkbFUkbtEIbtEIbtJouuHQuQJuiqKas4jc67VhTrKXDbthhs1HMt1lid5v3/K+dVTcdgmHOGpp35U9vPuV8Pgw+"
    "DB6180zorDS+23ztvycthwUdpGVh8K60/DBA20dtaOZJy6+kLVqjLVqjLVo/JTFrGrRF61L5+11DTW0Le8/sFrtkqlpYmp0g9oSt"
    "EPtfe10IO+b6bb+zdWZd8KRDr3JTx5QFZUF9l9OTnH1qD5kX7Au2ZTjPNnF2ffoxsz/YIR5Pik+5aw+f+naS2SceL4hPy7CJnSce"
    "LcQHXvCBV7n8Rh0DH3jBB17wgdf18mctq9scsc/MRAv7b7tC7BPbVeykfTOEnbXTQtjr4SqxTY5R8HcjB9p2ZpGz0faMucnhS50w"
    "y3Zxlgp32OHOxoVj7TVn88M7wuV2uV2Qscl5dRTPRaadeN4kvnXCLjZLPIfbHeJ5zY4Vz+UWvgsyGokvvOELb/jCG77whi+84Qvv"
    "+fKbd4TwbZTRPCwS88MmYnlEt6fC82Jzwxsi2O6weQSr6e5gX4XtItg8e9YstAvtEZtpWzr7j3CDhT0Z/ib8o/2jLQzXhNnOGkRz"
    "wgv2gj0THnW/kee+X3Zvf0c4J3Oe80aMI8478ycR/mgRo9B5r5EIFyxinHHeRyXCHeFlibBNYiDKEXmGzJ/EQJRCeYY1EgNRzsgz"
    "HJUYiDJHsHJY+KWFDQ3/JJYXbhE7GbaJYH8Nfx3BboryxOaHr0ewK+G/R7BGUVuxVnaRKHQTltnQ/a9X+GiY4Wxq2CNs7GxP+Imr"
    "KbM5UdXoofChMCM6JPPp59GZcIazMCp3LFM93JPVykVZJJFCidXLRXlUIjWWWHtcKWLluDiIlRG9GR6SSDPCMxKpelgukUokFqIh"
    "FqL1kmd6VGIh2h55pk/CHHmmqhFiIRpiIRpiIRpilWRVsgKu7z3F/rW+MoRjgEqOMHovLIF786SvGL+GGN85UKbYSqa4ah/3lSs2"
    "kyuWWS0DL2wmW6Bsmd3gK5afIpbPDBTLf26UMTLIGG3JGPPCdr5yxmFyxhuBcsagUOvAC4fJGm8EyhqoGxTm+YrZF4nZOwLF7HPE"
    "7L8HyhwNjTLH42SOd8gcZ6NLvnLHtEC5Ywm5o5pR7iiKtA14AW3AC0vIHmgDXkCboug9X7G5Q6DY/C6xuaZRbJ5HbN4YKIPUNMog"
    "y8kgG8kgWVYZpDT+2lcOGRYohxSSQz4kh4RGOeSrWNuCF4YFyiKFZJEPySKhURZB26/iHYFi8DpicB2jGLyPGHyIGDzPKgbXtsok"
    "u40ySVWrTDKBTFJBJrkuVCa5Pn1/oFzi8F14oSxQLplklEtesMolTcglLdLVB7wAH/BCWaBsMskom8AHvNCEbAKfFukdjWJtO2Lt"
    "GWJtllWs3UGsHUusvSNUrM2xyigTySgryChdyShvhsoo08goq8goCzJ2GOWUReSUm6xyShdyynByyjVyynJySqMM9QUvLCKrwBe8"
    "0IWsMpysco2sspysAt9GGWeNYmqmVUzdQEz9TaiYuoaYOoeYepSYepmYWhQqszQhs/Qns5wns9wQKbM0J7PsDpVZ2kXKLHMyF1rl"
    "lpbklson+CO5JTtUbrlglVvyQuWWO8gt2zI1BnihJdnl/48BXsgOlV0QA7yQFyq73EF2QYxtmYusYmcZsfPRULGzB7HzE2Jn1Uix"
    "81Co2HmG2FlO7PzSKsP8KVSG2UKGaRMpw/yaDJNHhnmdDPPvZJi2ZJg9WautckwYKsdkkGMak2PKrHLMQ6FyzJvkmBnkmOrkmJIs"
    "jQVeQCzwQgZZpjFZBrHACw+FyjJvkmVmkGWqk2UQqyRrlbMdzrY62+vsI2frnFXLrpZd1VkVZ76zG5ztztqd9bqzK86y3f8ecIZ/"
    "ngu++51NcwaG0dXHzgCMUrnKkE+zlpzzYwCeIesYMA0YBJyxhpwB9sD3zWSeZRas00uYR7nnqvCMss9msg9WH1ctGAPccIrcAMYA"
    "N4AtUH6YDAS2QPmgEOwDDsogB7UlB80TnlEWOkwWeiNQFsLqY14IhgAHXCQH7AiUA86RA8AOqAc7oH4JmQjsgPqiCCx0SphIuehx"
    "ctE75KKzwjPKRmCyH4VnlI3AZNOFZ8A0YARgPRgBWP8usR6MAKyfR6wHG6Ad2ADtCslIH5KRwAZo91WsbIRLOWk5OWkjOQls5Asj"
    "gWeUlYYFykqFZKUPyUqhUVbC6qM0BgMA09cR0+sYxfR9xPRDxPR5VjEd6I/2QH+0B/qXC2soMwH90b4JmalFem1yE1jpPWEm5aYJ"
    "5KYKchNYCdx0vfCMspOsPoRnlJ0mGWUnMN+nwjPKTlh9XJ8OxN8kPKPYfYbYnWUVu3cQu8cSu4H4wG6g/SZhC2UooD38upChhpOh"
    "rpGhlpOhGmXkkKMmkqNWkKO6kqOAZuCoaeSoVeSoBcIzylKLyFJgxNHCM8pSw8lS18hSy8lSWH0syADCzxOeUYzeQIwGwgOj1xCj"
    "5xCjjxKjLxOjge7wb0mmqvT/I5kK6F4oLKFMBXQ/IyyhTLUts4hc1YRc1Z9cdZ5cBZYCVzUnV4GlwFVgKXDVHOEZZauWZKv/ewpl"
    "q+xQ2QpMCbbKC5Wt7iBbYfUxJxOI3kp4RrH40VCxuAex+BNiMRAdWHwoVCw+QywuJxYDzVsJOyhjZZCxGpOxgOZ7hB2Usd4kY80g"
    "Y1UnY5Vkga3AWWCrocJYyllgK3DWr8lZeeSs18lZ/07OakvO2iM8o6wFBjXCM8pajclaYFCw1kOhstabZK0ZZK3qZC2sPvYIy6wS"
    "ntkrF1jmI+GZqsI0YJkqwjO7s3CBZV4XnnlAmAYsgwssg2sHmWsvmWsdmQvREM8ncyEa4l0hcyEa4iEa4k0jc23LrEPW2knm+uc6"
    "CJ/mnyxmlMXAY2CxH7lPVoP7ZFO4S1aDu2RTgmX2cbLZZrIZWEp323pxt+0eYTNlLrDNKbKN7o+1Mbo7tou7Y6MD3R1rYwaFuje2"
    "i3tjowPdG0ONshp4LYO81pa8BlZT5gKrXCSr6L7YRO6LTTe6K1bAXbEfgyXkNt0Vm+6YS/fECrgn9mOwhOyme2JooewGftPdue7c"
    "nXudu3N7hd2UucAeuh9WnfthDbkf9pSZR/bQ3bC7uRtWnbthDbkb9pTR3bBH7Vex7oXdzb2w6twLa8i9sKeM7oWhpbKccpzy3HLy"
    "3EbynO7a+cJyylxgiXVkCd0HKzf7yBKHyBK6D9Yy1F2wY9wFW8ddsHKju2D1uQv2qW1CrmuRrntgx7gHto57YOVG98Dqcw/sU9uE"
    "bAcW0t28odzNe4+7eZfMBPJdBflOd/P+14LtlLnABu3IBmfIBrr/VYf7Xynuf43j/tf8UHe/NplF5Dzd/Rptu5DzhpPzrpHzlpPz"
    "GmXo3tcms4isp3tfo20Xst5wst41st5ysh7YR3f5qtuJ5L0V5L2u5D3d5TvJXb6zdhV5D6ynzAXU132vI3YDUV/3vZ7kvldhOIeo"
    "f5Sof5mor7te87jr9X/+fyT36a5XYai7Xg0i3fU6E95B7tuWqXte87jn9f9HUPbTPa/CUPe8EEHZ7w6yH1hHd/+ac/fP5+5fHnf/"
    "nuLu31zu/uneH/hPd/++CufIiYLud7WyZUR33e/qxf2uqdzv2hPqflcO97syuN/1c+53hZHudrWyuttluNvVi7tdU7nbtSfU3a4c"
    "7nZlcLfr59ztCqOSLN3ramV1r8twr6sX97qmcq9rT6h7XTnc68rgXtfPudeFSMqC4EHdFRzKXcE87gqe5K7gX8M88uDr5MF/Jw+2"
    "JQ+CBfdwfbSK66OtXB99lKXro2pcH1Xh+uiGbF0fvZ6l66Nsro+S2bo+2k8eXEUe3Eoe/ChLebAaebAKefCG7CvkQuXBbPJgMnsa"
    "uXBb5g6y4V6y4TqyoT5fNT5fFT4fIiob6vNl8/kQUdkQ86wiqAjGyzXeoft8sbxgsP+Qsy0mL3jIHyx3lwPc4xvuLwdXg14Gdiro"
    "YmBfBMv9L5w1MC2F8crsF8EX/nIp6Ry0lJKZUoZSlKEUZTNtRzNY7HUzQGyRoNNN9vcmCG5xNsc8Fyx3NslEukcV/t7cEgRSszx4"
    "TmrKAtTtC6dLHWpRh9pJ4hMZ1E0PO5ojYlPNUbElpr+FfWf0Xy9cbO6V/76NaxFcc1bVHgwOOVthvzDHnTWOF5uJwb3S4lowVVoc"
    "Cg5Ki+PmC2nRTdqgFdqgFdqgFdqgFdp0i2vbh8UaCSp2sc1kZ2ehLbBlYs/KfzXg7nCae6MHzYPG2lom11l3O0j+xs9OF+2Ys+vC"
    "G+TUuENqmmuFttbmmlrSsrUZJC2PmePS8i57g7TMlLZojbZojbZojbZojbZojbaZqZeEhZrY780zFrbQbhXbwzOYVmGm2Dk7Q7Aj"
    "y24K8K5Juz142jxtCmyO+d7ZUrcy9KxnF4eD7dfma7M0jMMpdopbYWW5t4FP0j5t4FPgfitHPDzxWRx+beCz1LWHT4v0H1JZ0p+b"
    "AvjACz7wgg+84AOvpfIbcQifH9y7/DKEHbV9QthD4QCxnWFXsR0OR2ETw7Ni5eFzEazA/kHmzo1hVTdGBdYP75Nx+nXY3L7o7N8c"
    "euU5y45uDHFy3TUqlr74TUaBPWL+IJ4Ftqp4ltn7xPNF21w882yZeP7Fwrerw7ti8bxLfOENX3j7gpH3iS+84Qtv+MK7q2Blsfje"
    "lfFieG8E+334dAS7HJ4QuynS8qbRSLGiaKbYY9FasY+ijBh20Y35JmcDwkO2VljLYfFVx+/zw35h1XCus5ejHtKHj0drHXddDhPR"
    "s441ysP0eIh8Lsy86LybSIRaEuOQ80aMfs67qkToE/aQCJfDtRKhPESM9Lg8RIyFmVMkBqIgBqIgBqL0k2eoKjEQBTEQJSG//WyE"
    "GIiCGFMyb4m2hrBMngf1jB4Rez36XKwk+kxsZLRX7EBULYY9F42MYdXju8RO27dt07BpuNxpj9Xh6nBx2D4c62x3mApLnb0SXQgv"
    "OhsZfR5mR9kubp6w8q3xkmiBs6J4k/zLQvuzTrs4iLXcRUGsxS5Ke4lUKrFeiS6GFyRStsQqcVHyJNKCaIlEmhRtkkiTJBaiIRai"
    "LZZnai+xEA2xEA2xEK1EnilPYiEaYiEaYk3KOhv+LoL1imqL/UX+lfdZ0cPRVLEVUbsYdn18LIL9Ju4r1iv+RGxGrP+q6XNxsdhG"
    "t06Z7Oz2aHaY7+xKWDuq5+whNyf7OXsrahYVOvtrVBBNcNbQ9fvmaHO0I2oQt4paRa/EteJl0bJoSzwmLo/Ko73ZG120QRIxP5wt"
    "EetFtSViP4n5lovWTCJOkJgNXTzE3OGiIeYrLh5ibnHxEHNv9jiJiaiIiaiIiagPyTM+LTER9a/yjAUSE1ERE1FfkWesFW+RZxwT"
    "I+a47EqGxTXYVx6dT7bdYh7ylW8v8x4ci/vLYFThy5aB8mUvo6zbhazrONZX3u0cKO/OtFoGju0cKPOibKYNAuXF58iLkVFeHGyU"
    "fQeQfW+yyr77wlsC5d/l5N8y8u/0UOvAscvJwGVkYNRND+8NlP+mkv8Okv++MMp/R4yy8FGycH+rLFxKFm4cTwyUh6+Rhw+Rh48b"
    "5eFusbYBx14jEx8iEx83ysRo0y2+JVCeq2WU5waR546T526wynMPW2XjLmTjhWTjMrLx3aGycYfUg0b5OJd83Jp8fIx8fJdVPs5M"
    "aVtwbC4ZuTUZ+RgZ+S6rjIy2malNgfLZdvJZjlE+m0c+G2yVz+JQ+ayJVVZ+hqy8lay8gqycGSorzyArt0g/bpSXnyYvf09e9qzy"
    "8tdGeXmKVV7+IaU+4Ninyczfk5nhA4792igzT7HKzPD5IfUHo7xV1Spv3Ufeak7eKiNv3RgqbxWTt34ZKjv3ITsPIDt3JTvXjJSd"
    "z4bKzs9Fys6/yThilJ8LrPJzGfn5RfJzHvn5L+TnGaHy810Z6guOLbDK0GVk6BfJ0Hlk6L+QoWeEytDwvSujiVV+OkR+ukp+qhoq"
    "P/UgP60lPz0bKT8NIT/dGylLP02WPhEqS98bKUuPJEvPJEuvJUtnxMrSCzM3WeXpWqHy9Hzy9FzydB/y9GXydPm/8PSUTI0BjkUM"
    "cOx8MvVcMnUfMvVlMnX5vzA1YkzJfNsqD50lD7UPlYdS5KEL5KHPyUN5kfLQEvLQJvLQ1lDZum2kbP0I2fpzsvVnZOu9ZOtqsbL1"
    "SLL1XWTr/VlNQ+Xr1eTrseTrUvL1RfJ1dqR83Y58vYB8PamSr7M0Fjh2NRl7LBm7lIx9kYyNWODYdmTsBWTsSZWMnQWWHRQq38wm"
    "39SOlG+eJt80I98UkG+qxco3Dcg3tcg3Y8g3v4uUtWuTtWeRtaeStdvFytrHImXtvrGy9idk7RFk7WKy9t7syaHydj55u16kvN2P"
    "vF1I3p5A3t5M3m5F3l5G3i4nb4/L1pjg2Hwyd71ImbsfmbuQzD2BzL2ZzN2KzL2MzF1O5kbMcdlXnaWqp6p3c5brrIWzls7weZ+z"
    "Ac6ynLVx9gtnP3N2t7MHnP2Q/UN2TvWc6iOcHc4+nN2neh+pf9QZWFtXxxXC0pWrYHxeFgafJiwO7lYmx+p4i5HVcADuVh6WtXAA"
    "plY2n2nB5ODzLuTzMuVuYfTOgTI6VsdlFiwMvn2OfBsZ5VswMMqXk9XLyOrTQzB6R2F15XUwOnh9n3C3MvtyMnsZmR2r430hWBe8"
    "OpW8epC8CtYFr4JxUX+N7H6I7A7GRX23+Aj5/Sj5HcwOfi8lvzcW7laGv0aGP0SGP26U4bE6bhyDZacJdyt/DiJ/Hid/gmXBn2BY"
    "tMsly7cmyx8jy4Nh0S4z9TB5vgt5fiF5vow8D4YHz3cQ7lamzyXTtybTHyPT32WV6bE67pACq4Int5Mnc4zy5Dzy5GCrPAlWBU+C"
    "UdH+abL992R7MCrag1HRHoy6VJi4Cfn+GfL9VvL9CvI9mL6VsL3yPXaL0Z9g/KfJ+N+T8aEmlgp3K+NPscr4WB23SFcWBXcrH95H"
    "PmxOPiwjH4JFwYfF5ENlUPgq65eR9V8k6+eR9f9C1geDwu+ujF+S9/uQ9weQ97uS98H44H0w/kRhfeX93wh3K/MXWGX+MjL/i2T+"
    "PDL/X8j8M0JlfqyOf5MB1rwo3K28d5W8VzVU3utB3ltL3gPfgfeGkPfAmBeFeZX955P955L9+5D9L5P9y/+F/adkguHB/0+T/8H8"
    "l4X9lf9Hkv9nkv/Xkv/B/OD/hcLdqgCgPgYId6sCmEsF0IcK4DIVQPm/KACsjhdmgiXBb2fJb+1D5bcU+e0C+e1z8ltepPy2hPy2"
    "ifwGhjwtjKsqYCxVQClVwEWqADAk4rSjClhAFTCpUgVkQQFAB0ABZIoKUB3wOXXAZ9QBe6kDoACgA0ZSB9xFHbBfuFuVwGoqgbFU"
    "AqVUAhepBKBKRgp3qxJYQCUwqVIJZIG9wYobhbuVx8CK4LGnyWPNyGMF5DGwInisAXmsFnlsDHkMjLhRmFbVABgR8fpRDRRSDUyg"
    "GthMNdCKamAZ1UA51cC47N9RD9SmHphFPTCVegBKAHoASgB6AErgN6IGVA+MoB4oph7YK9ytiiCfiqBepIqgHxVBIRXBBCqCzVQE"
    "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"
    "rEMN/TNAUGWqyCr3qfB53lSqM6iw76nP9L9xpudvh42evzW3evp22OjpW3OntPTs7bDRszeUqEqDTnuVOg0qTRUZVNQZqig9d7vM"
    "U7eYp25/46nbZZsRTaBaK6ZaG0S1BtWlp3sDeLr3KE/3Xla1Ful52ymet5XbCqolPW/bGupp2ymetpVbPW27k6dtW8Pm8S+o2vSs"
    "DS1UtTWiasMM01O/Szz1e56nfjt56ldmodpUkUEV6TnbMJ6zHQr1nK0mz9m6RXrKNpSnbMN4ynYo1FO2mjxl6xZVSQ2neptF9XaS"
    "6k3P2GryjA0tVb1Bv02gfnuE+m0J9dto6jeoN1VkUD96vjaR52sHeb62JTxJ9aPna3MjPV3rwNO1iTxdO8jTtS1hM2q4Z6jh7kzX"
    "s7UOPFubyLO1gzxbg4equGeo4qCu9JTwEZ4SPsZTwq1hQ+q4XOq4+6njoOJUkUHl/I4qZxVVjp6rNeW52i95rlbCc7UtkZ6qfRDW"
    "oJbrTi13hFpuA7WcnqqV8FRti9NGeqYGT1Vz3anmjlDNbaCa0zO1Ep6pbRFVpaeHl3h6eIKnh1dDPT0sivpRzz1JPaenhx9HUHOq"
    "yKBm+lLNTKWa0fO0YzxPWxnpedp9PE/rx/O0KbGepnXhadocnqZV8DTtGE/TVkbbqOmKqOn0NG2K00J6ltaFZ2lzeJZWwbO0YzxL"
    "Wxlto6oroqrTs7Qpoqb0VLGEp4oXf6LqoOuyqOt2U9e1p65bSV0HVaeKDKpFz9GGRE9RtRRTtSyiaplA1aLnaGN5jnYw1nO0Rik9"
    "RXuWp2hDoheo7b6ltltPbbeO2k5P0cbyFO1grKdojZz20TO0Z3mGNiR6geruW6q79VR366ju9AxtLM/QDsZ6htZIVJSeNnaP9LSx"
    "Pk8bb+Jp48M8bbQ8bRzF08ZpPG38H542bhF1p4oM6kTPz05Hen5meX72Ds/PuvP8bD3PzzbEen7Wkudno3h+VpjS07O3eHp2OtLT"
    "M8vTs3d4etadp2freXq2Ib5EjTeCGk9Pzwqd1tGzs7d4dnY60rMzy7Ozd3h21p1nZ+t5drYhvkSVN4IqT8/OCkU96Snkyqgfdd5Q"
    "6ryr1Hl6cvZD1Jg6byJ1Xl/qvAbUecup86DyVJFBhXxIFaLnZn15blbAc7OVPDfbHg+gCvmaKkTPzcriOlQh+6lC9NTsnng7tZ6e"
    "mvXlqVkBT81W8tRse6ynZr15ajY3padmZXE3ar3bqPWyauiZGSKr2tMzs748MyvgmdlKnpltj/XMrDfPzOam9MwMkVXt3Ua1B9Wk"
    "p5MzeTpZyNPJRTyd7Mk+/j7Ood7T08mWPJ1cztPJLjydfJunk39NQe2lcf+qL/evXuL+VT73r8Zx/6qY+1fTuH81jvtXU7h/1ZD7"
    "V59y/2op96+Kqff6Uu+9RL2XT703jnqvmHpvGvXeOOq9KdR7Dan3PqXeW0q9VwxFRtW3i6pvG1XfXKq+HVR9s6n6Cqj6JlL1DaLq"
    "G03Vt5OqbxlVXwdnr96s9lryteQPcv3wzztco5zd7wzfKmtRoqVaa9N+YM1rrL3/n21GudrKljbNpm1Jbkn2lqt3clpaZgJWnvZE"
    "WtHNRTc39crdJ+6bekuT5VL2RBrul/7TC9cTaVqXmdAITb2imzXG0qTeww/3S5O+53v7kvuSnvtfLWeIoL++JQmPyl8pF29EQ8ym"
    "HuJoVPx6U4mCC1Fw7WNkfMP3P6f1ZtQtjFz5HPhcmqz8FUTTt56Whl9pyufz+Xwef8fn73gustb7rEfJn9O+SX6TbOzhauxdSzsg"
    "tiBtWtqimxfdPNxb4D5xP9zrLPf4hvvO3vm0RBL2bdrKBKxlYlKia1rXtChxNu0WZ239lu47yqLELWlnpWS+11LaTEqgDKUom+9V"
    "PgGuaWn6OwfS9GmGe4tu1ufp7Ok9ngH3nb1JCY13Nk3jJZL6VCsT+lRt/a5p+ly38Lnme1qGZ7iFT4ay+V4jv5Ffx9kF74KX6Wf6"
    "PZ39yfuTh6fS3vkmiaeo7IUF8kR4wmvylHg2fVL0znAPv4rfOZumv4NfbClPok8738OT4nnxpHjetj6eTZ/4Fj4xeqetPFkjebZM"
    "ufBkuFCK8kw+NUpR/nQCz6tPrk9d2W/47OxVvgGeVEf8mryBPjWeQMcw4gi25AhGifleV77JLXwTPKHOhPOcCd+m4U3asj8b+dqf"
    "Fzztz558l0Z+Jt9H36Wn/3RC26NG30jbo+bpxLnkueQquVYlGyTmihW5n52fNj/tE6/IfeL+E6+Z3OMb7pt59RIPie1Mm5+AzUj0"
    "SBxNO5o2MHEx0SmtU1ozf4b7jrKB7jvKmvn53gxp0yMxUNpcTKAs35ufOCL250RWEnYuMUfs8WSP5La0bWnliZrJyM3t8sSPiatp"
    "V9Ne8R935agrd6WoK3flqHvF/7PUoRZ1qC0Xnx8TqPuzX/m2uJIJfae5CX3zT7z5afruzTy9x/vivpnXI6HPfpHP/lBCe2A+e6CZ"
    "fzRN+6AT+yDf0zK8byf2AsryvR5JfcaafMYfE/qMRxLaE1lJ7Yk5Ce2JV/xtadoXeFO879U07Ys/+1qH90Ud3hd1eF/U/dk3gQl+"
    "7uyQf8j/k7PqQfXgAWdlfpn/ujP0hI7+uSTevHKUi6QX0CsNpGfQH9o7GP1PPLwp3u0i3w1vOUPeXnso30Pv1JMe0j5q5qM/tJc6"
    "sZcw+s18vBneoSbf4ceEvgPe6nF5a+0pvFW5vDV6ab70lPYVeumc9BT6Q3tLZwz6Q3sLo/+Kj94w0h9/kh5Bb1SX/nhdegS1qEct"
    "6lGLetSi/oj0lfaa9ljlPMFnM6+y99BLmk0NpPe0x/D2F/n2mh0zmB0DXV8cZS92Yi+idzTL6iXmsx/Ri9pjeMuf5kV5QrPi8eRP"
    "s6Lcvfs29uZPc6JcekWzbz6z78/MvnPSm9pj6BWdP4d8nT/VA50/Zb72pwm0Pw/52p/VA+3PMtdj6o8W2qvqjxbaq0eSR5LNveZe"
    "Qq6Ewxb5SJxMa5y4L+2+tBe8k+7/cf+Clyf3+Ib7PG9nYrdYReKk2GuJjxJT06amnU+sS7yf9n7aSv819x1l5913lK30a0kZSs9L"
    "m3UJlNXy05NPibVJ3irWJ3kxAftH4rHkhcQF129PJT9L+yxtZvKG5B8Sf0gc8P/hSlFX05WjbqYrR90Bf4nUobam+DyVnCk+NyRR"
    "t8S/lFiYhGUlXxLLTC4VeyfRIglrltyfaOvszeTWZHGiODE6OSnZNdHVjXRG8m9pf0tLmmbJton90qI4gTajXT3a/Jj4WxraJM2S"
    "AG3Q6k2JsTU5WmJMSqINWiXlv11R2fO4Gie0f72EjsIL3n1pOg55nt6j73Gf532U0H5cx37cndDROMnRWOlPTdPxeJ/jUcvXMvT9"
    "+xwRlNXyH0tqfz3F/rqB/fVUUkflVo7KxYSOygH/AsflszQdlz8kdFyW+FqHvkcd+h516HvULfH3J7Rftia1XyaxXzLYLwuTOjov"
    "cXSWcnRacHSSpm1Cx6eY49OV4/O3NB2fJYG2Qd8Xc4S6coTQBn2PNm6ETDP513PTnCWdtXGGrUj8C7v4j6vh37w9E5wJMDqaHc09"
    "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"
    "+Fz0ur+Ptlv9fbS1sEtCtD0XlZE2tN4qfQ4atKEVbWWiD70ah9Uwb4hNMt+KLTNtLexV81440FsLe6PJD/O9Nu9hWnnb7n0betsS"
    "vWoGhu+JR34In2m+tYd4NBSfLdE/xAde8IHXNInRQ3zgBZ9/RBup43abn8SmmdstbIm518JaWCc21ewyh8PD4Xpf10vDpeE5n2ND"
    "va0wQ20D08DUt8tsPVPPdE+mej/4rvd+8D3nvS6LZwPxre/94Ns96Su+8F4vMU+ZcxLzsvjCu77EXGbh2zfZZh60sI62lFh5+7HY"
    "edPMwk762LAmdqZYdbtPKriWx73EeLNWUHywvdWONWNNK5uyP4c/+9V9PzvCjDClU9VtI7NPeiRG+nh/9Bns/dGnlfdHn3HeH31K"
    "p4oT9EGvWvIevpe8h7WD5T1uta3kPVJ2nLxHP1ta/nZIScbj6hlqXj8ZavbPsBsCzf+/Wn2NnMfrv9qvQ81fazR/x4VaBaNZBe+6"
    "kYHWQTHroJ7Te8j5YlYC7tXza1DN04NG83Qb8zTTaDWkWQ1bWQ3PRRcDrYcHQ62HJayHMpG2IecfDLUilrAi0FYmei/UfLzRaD72"
    "YD6+wXz0q0+pijdYFd+yKtparYot0cBQ6yKfddHKaF00ZF38I1If5Hw+K6OV0cpoyMqAzz+iXUbz7hTz7jLzbqjVvFvGvDthtDp+"
    "YnXcbrU67mV1OFZH9+RwqPWxlPUx1Gh9NGB91GN99E3UFzm/lBUy1GiFNGCF1GOFwLdvss9ofj1lNb8s8+tW5leK+dWP+fWg1Sop"
    "xSr5mFXSjFVSn1Uyk1VSOtXIaJ0krJORrJOxrJOfQ62TEUbrpDjRPsj5hJUykpUylpWCPsh59EHOo09xMio1KrXJ29+9/cPbO942"
    "emvvDX8h5w1vRd5e9jbJG35Sb4U3VImyQweLqihhAXz9q0XF5ErVoFa0csAOMyyqAHmPKkDeowLwfTGrp55D5Tws1aP1867UilZQ"
    "MSsI7PCuQ9Yjv5H1yO9tzG9k/OtSEVpFS1hFZSJU0FVSRVpHW1lHz0mtaCU9GGolLWElgR2ei5DlyGNkOfK4B/P4DeYxMhzt+awm"
    "ZPg0qQStpn9EE1hPb7CevmU9oZJQT1ukVrSi8llRrYxWVENWFNhhS4SsRr6eYr5eZr4iq1dIrWi+IqOnSgVoVSGjz0kFaFXVY1X1"
    "TU6wrn5iXaGiUFf3sq4c66q71IpW1lJW1lCjldWAlVWPlQV26J4gi5GXyGLkpWVe3sq8TDEv+zEvkcHwT1hdI1ldY1ldyOBWkvla"
    "XcUJKgv1VYr19THrqxnrqz7raybrq3QKtaIVlrDCRrLCxrLCUL2oMFQvKgzsUFoqZZTUyj/kQqW8I7Vyr1yolDekVibJhUrBhV7o"
    "9w9W2UZW2b2ssiJW2SRW2QpWWdME9aWVplVWwjv4+ldbUnGoLFVouVJxWmWoGNVc11BxtaTiusbXz0hWXjErDxWlyu1hKrfNUnla"
    "ZagM1Vq3mm2sDFVar4eqtG41S1h/ZSLVWa+HqrNuNUtYgagkVXRXUdHlU9GNkQrUKkMFqMZqYXuwAt5gBajCetXksw5VYU2jwtru"
    "60P11asmn5Wo+moa9dV2qSBVeh+Gb7AWv2UtqtJbZlCJWmXI9FPM9MvMdNVWK8wyZroqq6lUVuuprM5RWa0w9ViPfRPVVVOpq9ZT"
    "V52jroKnViQqRxXgRirA3VSA06gAlxjHmkRFapUho1VT1aKmSqipBlNTtaKmGkdFVZ2KqhYVVUJFNZiKqhUV1Tif/41YmQkrcyQr"
    "cywrU/VUK+qpcVIxqgy3mVKszY9Zm81Ym/VZmzNZm6jM0uSjUeSjv5OP3iEftScfvUE+epl81Iq1OYq1+XfW5juszfaszTdYmy+z"
    "Nlv5KtvECv0HK3QjK/ReVmgRK3QSK3QFK7Rp0jQp78q7kRbXSNvQtBH7LOwZvhi8GNzsPvNf8fpmV1pe4zu8Lu26h+q71NQROxZG"
    "pmZYM2xvRpsOYYfwgjvmv8O99v573Lvg3pF7uNtefEYb3HvHXQq3iy2QamllDpvbxd4La5nsMDtca5b7lXJfU8fWtTgBn+DrLFva"
    "1vq7y6Ulx6BtQnS/tKF1rfRZLm1oRdv9UU9zi9hq84DY52aI2M32SbFHzHQzKZwU3m+2mObeHvPr4IvhxXC7aWzzTJ65MX7Et8Pn"
    "ft+6RTwuhvDZ7lvhc2PcSXzgdb/E2CI+8IIPvODTKR7v8QO22s4xsB9MJQu7yb4gdo9tKFbNPCo/QbHXfyZ8lsC33WZuM8dNZTvY"
    "DDYTbVefjZ4rkmreD757vdda8bzNwPe494LvRO8H32ZJ2VQ1ifmo2Ssx14ovvOEL74kSs6uFb9nU3Xa+WG37nNi1tpyD9bJ5Yp3t"
    "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"
    "Q4YuIkO/FylDo++wzF1O+emmSPlpJvmpCvlpH/npEPlpZaz8tJf8VCZRll4UK0tPJ0vfSZaeT5a+KVGW/iNZ+mKsLN00e0CkPG3J"
    "03vJ0+Vj5em5kfL0HbHydBPy9CzydMNsjQGOtWTqvWRqxADHzo2UqREDHNuETD2LTI0YDbP7RcpD7clDBeShFeSh+rHyUE3y0FLy"
    "UOVEeehorDzUKFa2zidbLyFb5ybK1tPI1vXJ1jalbF2YKFv3TSlbl6rya6R8PSBWvn45Ur7uFStf55CvJ5Cv15Cvf5soX1ckX2O9"
    "9WukjI1Y4NiXI2VsxALH5pCxJ5Cx15Cxf5soY1ckYyPWB5Wrxco3Pcg3g8g3b5Fvisg3O8g3HRPlm6Xkm4yU8k1r8k2ZRFk7Jmv3"
    "JGvfQtZ+lay9lqydlVLWfpKsPZCs/TxZu0vOxlh5exJ5u26ivF2evF2WvD2MvD2UvJ1H3m5M3q6TUt7ulKMxwbGTyNyICY4tT+Yu"
    "S+YeRuYeSubOI3M3JnPXSSlzI2annIveGlRtULWdt3re7vFWxttYbxneWnv7NefXnNyquVW35mzNOe/toaoPVZ3srbO3xd4GeYNv"
    "2htiDffWwhVzdVzswNIlq+DJwthg8NXC4uBuZXKg3vsOrAwe7koeBiPPEqZWNh8TgcnB52DyUcLm4G5l9E/J6Fgdt4vAwuDbv5Jv"
    "15NvwcA/CEMrq4OBhYcjMDp4vSp5vQ953cXgbmX2S2T2dlaZHatjF4N1watgXfDqYvIqWBe8CsZFOxgX7Vlk92Nk944JmB38DmY/"
    "Juyu/F5Mfh8l3K0M39Aqw2eR4Y+R4bE6HpWAZVcJdyt/fkb+BMuCP3eRP8Gw8BtMlm9Jll9GlgfDwm9d8k/y/IPk+R3k+fPkeTA8"
    "eH5OCtytTD+YTN+STL+MTN/NKdNjdTwnBVYFTy4nT24gTzZ1ypO/kCfBquBJMCr8R5DtwajwzyDbg1HhD0ZdJkw8iXz/H7ZXvpef"
    "dBC2V75/mHzfJg3uVsYfQcZHbq4W7lbGh5oA429wyvhYHbdJg0V/Ee5WPhxDPhxFPgSLgg83kQ8vkA/BoL8IAyvrg0GPCAMr688h"
    "6xeR9cGg6Dcs8wB5fy55/2nyfmfyPhgfvA/G3yKsr7zfVrhbmb8+mR8MDOaf7JT555D5i8j870XK/Fgdt80Ea84W7lbem0neq0Le"
    "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"
    "WiFP0y7yNO0LnqZ9mdSjpiugptPTtM1ekelZ2hM8SyvkWdpFnqV9wbO0L/X9JYKqOj1LQwRVddB1eqpYNqWnij8mjanrelLXbaeu"
    "m0ld9yV1HVSdKjKoFj1HO5C0oWqZR9VynKplP1XLC1Qt66la9BztTFpP0c7xFO1AoqdonXiKdiNP0e7mKdqh1FFqu57Udh9R23Wt"
    "omdo53iGdiDRM7ROPEO7kWdod/MM7VDqKNVdT6q7j6juoKL0tPFMUp76Lpf67r9PG//J08ZzPG38LlWR+m4Y9R3UnSoyqJPBVCdv"
    "U52sozrZRHXyOdXJeaoTPT87kdbzswE8P/PqJFGNp6dnEU/P2vP07Cuenh1I6elZD56e7eXp2Yn0Xmq8c9R4+3P07Mzy7Czi2Vl7"
    "np19xbMzRFSVN5YqbwdV3l6qvHNUeVBPegq5KKWnkNXSh6nzVN/9R+etpM7TU8gm6SHUef2p81ZR50HlqSKDCompQvpQhQynCvmY"
    "KmQSVYiem43mudlknpvVyWxLFaLnZgOz9NRsK0/N+vDU7KtUHrXeHmq9Q9R6emo2mqdmk3lqVidTT80a89RsoNc2ema2lWdmfXhm"
    "9lUqj2pvD9XeIao9PTMbzTOzyTwzq5OpZ2aNeWY2UFSTnk4e5OnkPJ5O7ubpZF+eTr6R/iP13jnqPT2d3MnTyV2ZB6j3nqPeg9pT"
    "RQbVpPtXBdy/Wsz9q0+q6/5VqRq6f/Ud96/yq+v+la2h+1d9q+v+VWfuXwXcvwprqN6bTr1XQL23mHrvk+qLqflU731HvZdfXfWe"
    "raF6r2911XudqfcC6r2wRj8+/3Q+fwGffzGf/5Pqi6n69Pm/4/PjHVT16fPjHVT1DaTq0+fHO/SrvqKi2ongRFAvxKX/L7nKe5sS"
    "TJHvSlpxR+9q64xKuK8eJ/5fS8k1o1KJ54xKMyo9Gz4b7gpw7QrqBBczYC2CBzK+q/RdpSthi+C7Snh9JbTyGt/htf1/vXA9kKFt"
    "FzM0wpXwu0oaw4b6Gv3w2oaF3nK99fHW3xsi6Ls/G6JHybvgqw0RDTGvSByNine/IlEKJU5/uXIZuT8jN6iEmBpdI5c8B77asORd"
    "EE0/dZ0A73KFz1fI5+vD9ynk+/TxkXP5bv35bg0qNajUMGwYdpKrU/hTRqUA5oJqwT8r/bPSnaHzX/H6zvCovMZ3eI3vjok1DtqK"
    "DQm2BBcqXai0MDiS0cHbR+EQ/z3uLQw6ZByRO+vlHu7iHu7i3vr/9wS4qgX6PpUCfZo7w39W0uc5GuprPANeHw23BBrvSIbGOxbo"
    "U7XlU30UXqikz9UhQ59rfaj38AwdMvTJcG99ONzba96KvbX3hr+Rjb//jafS0WkY4ilKRsHJE+EJ8Zx3yrPpk2J07gzxrngfvCve"
    "B+84RJ5En3Z9iCd18rT6vB/Js+kTd8jQJ8bofCRPNlyerb1ceDJcr/Gp2/Opa/Kpd1fqxCduyKcuGTd8PRqWfAI8qc74Txn4BPrU"
    "eAKdw4WcwSGcwYXB+lDnbwjnD3f0k+CztOVnwSf5iOM5nONZzPGcxM8ynJ+lmJ9lkn/q1/iJ2vMT1eQn2l1pd6XWYetwXYBrXRAH"
    "ncQmBUHQPqN9xqfhJP8Vrz8Ne8hrfIfXPcJRwUtirYINYr2DO4KTGSczXguWBj9X+rnS/rC3/x73XvPf497+8Cm5h7uvic/SAPee"
    "Ch8JzovtDnaIRWFuAGsbXgq+yPgi45Hg2WCEt/rhtmBOxpyMlqZt+EUG2h7xd5+VljkZaGtpBkobWh+RPs9KG1rRNtCUfFpcQaCf"
    "qRM/+adh+wz97D1CfY3Pi9c9wjsCffalfPaXAh2BDRyB/eHJDB2DnyvpGDwV6j18XtzD58W9p/yn0md8ls+4jc94PtCR2MGRyOVI"
    "tDRfZOhYjAh0LOZk6FgMNNqGzzsi0NFAGz4v2gYa/Puem71183YqPBXeYG4wnbzd5+2b8BsZCZ391iE+ecksT5JRwKjEMjIYDx0d"
    "zP6nIT4pPttSfjZ8yt7y6XWEngoxOqNkhHSM9st46ChhBDFKmP398snwGZ7lZ9jGz4BPhfv4VI/Ip9aRGmgwSo/ISOlY5XKsWsp4"
    "6GiNCHS0MLIYLcx+SxmNQTIep0JcGI0bZDy+CXHdzBFDK9rRina0on1+BsZKR01HrCRP8LVHWDJ6GCWtplhGT0cMn34pP71WR29W"
    "x2t+LLQ2erM2XpPR0SobxSprJaO4n3XRNnyWn3IbP6VWRdtQq+KRQKuifjjQfMHR1Jp4JNCaQIuOJsZzB8czl+OJ0WzJ/BnE/OnG"
    "/LmB+XMfx3MQx7Mbx/MGjud9Zn6G9h/E/t3Y/wb2h4e3cH54OsB1OqgSDBD7zI9oz4yeGefDz/z/8fp8OF1e4zu8nh42DJ4Tmx/0"
    "CmGvBRMDxOsZLA5OZZzKqGxe89/jXk//Pe5VNo/KPdztKT6LA9x71PwYfCl2LOgSwhYFm8WqeX5PgiSoEFYNj2Ucy0iHZ4Lm3laa"
    "av4+2ir4+2hLh82DM9LSR9rQWkH6VA3Rhla09TGvB6tD2OeBCWHHAvw1vHJ+ft4UWxAcD45mHM1oEXYMD2UcymgVtgurBlWDk8HB"
    "oF9Gv4zr7QLfDp8Wvh0+rXw7fE76dvhcb7eaBRLjeNBCYnQMW0mMduFJiXEwgM9WUzLyuCYFOr4DOAvnw54ZOg/TQ32Nscfr6eHE"
    "QMdxMcfxuUBno1eos1HZz6zOxynOx6NG72HsT3FGcO9RkxvqeFXleJ0JdLy+DHRWuoQ6K5sDnZWVJuG8HMvQeWke6Lz0MdqGsUcb"
    "xr55oDODtj7meKDj0jHUcWnHcTkY6LisDnV2DGenHGfnzUBn53p7NEPn5xDnp2qg89MvQ+dnq1EfjP0hzhB8MPbwwdjDZ6v53ttl"
    "b/gbboG3ct7wbxbx23sbeKvqrcgUGcyOVsf8ELNRUgX4Oj3ETFWR2cIc6YyhOs6HGH2M92KON0b+NZkRnbVHDWasocyazltlmSOd"
    "uVOcOVRHZYPRxrhW5bhitDGuGOlqMhM6exhp3O9jvuT8YeYwf5i5RTJ7mCOdQcw2ZrB5oDOI6lhpMLoYP4wuxq8dx+9goOOHkV0g"
    "M6CziJFtJTOgs7jVYAYxj4bzWI7ziBkcIbOIOdKZPMSZRBZgJpEFmElUx/UWM/S9zFEgs4QZKidz1EAuzBCuy5zNwOpsNuRsNuBs"
    "wgt+Q4PTgc4kTDGuBOE+87NZMrOYQUXCKjKzOpuYmcWcGUW214hsPf08zecMn+IMY+YUIRsSIecHmGGdTcxAVc6AYlqaiFaNiFaB"
    "iJYO+5iEM614VoF4hhadacy1IucxIucimWmdTYy0YlkLYlmrULHsJJFsQXCI861I1ipUJDvp5+EoZ/wQZ1xxrFWoOHZSZkoR9XUi"
    "6udE1GNE1BEy4zqbmCmtt6ast3Kst2zWW1Wrc/690TlvyjkvxznP5pxXtUODy5x5jdeU8coxXjbjwXNoUMVUMdNDXNPDwqCWWIcg"
    "P2iR0SIjMh38V7yOzJWwg9zLD/D6StidI/pe8I1Yv+D14HTG6Ywng7+EuRm5GbeZfv573HsyyM3AvdtMebmHu7iHu7hX3vw9uD+E"
    "lQt/DWD3hplirX02gvnvDl8N8ayZ4Tvh2GBscNqr5twAbXf7+2jL9PfRdtq8KG1ovVv6vBpmSp93QrS9aC74WYKdCvBXVY+EqVAz"
    "ZHaos/RtcJPkUenwgXBaMC3oHvYO84I8X3u3hY2CRsHX5lvfCp/Svh0+3X07fA76Vvh8bRaKD7xKS4wHwu4So3cIH3jBZ6E5EL4i"
    "1incJrY63Cf2WBgY2E1hA7GO4dXhM8EzwfDw8fDO4M4gDsd41fFZxu/D73wlVgka+7XUlYwrGYdtR+8H3+HeD76x94Pv78MqwXfi"
    "eSUDvoftFPGF93CJ+XgYS8wxIXzh3VhiDg/hO8WWZAmu/EBzoVagGROZFhmaM1dCfY08wesrPvt1zv8S6pxvDjRzvmHm3GZOZ2ju"
    "5DJ3yhu9hzzJZfbgXnmvC3RuX+XcvsO5vT/UDPo10AzKDDWDTpvcQHNoKHNoLHPoRaNtyJOhzKKxzCK0vejHXufwAc5hb87hbZzD"
    "dqFm0hFmUhdmkmEmfW2aB5pL05hLecylRsylhUZ9kCfTmE15zKZGzCb4LDRXhzpXj3OuxnCuvgt0roaHOlevhJpR25hR+5hRgdGM"
    "ahBqRh22zwSaU3cypz7L0JyqEmhOXcnQnJpi1Rd5ciezCr7IkyqBZhV8kSfwnWLzXJ77ozf8DnD8+/X23vCbFOp5w29mn+XtA/uB"
    "fdY96zp7Q0Yp+lQxyKASlOkg2YTsKpQMQ15plgF9IoOMQY4gY5AjyJZ+kkWaaeUNsqy7ZJrm2m2SV5ptucw2oM9tBhmCXHiVufAO"
    "cwHZ0VqyRzNuLDPuRYNsQ84h25BzyLZ7JeOQV5p1Q5l1Y5l1QJ/TBhnxreSVznlvzvltnHNkA9qnMfPymHmNmHkLDbIOuXeEudeF"
    "uWeYe19LXmn2TWP25TH7GjH7gD5fG2QA5vZxzu0Yzi0yAHOLDMDcYvY7SrZoBmL2Y8kWzUDMfmPJFmTfAclAzcF9zEFkH3IQ2XeT"
    "ZCDySrPwTmYhMhpZWCXQLERGIwuBPoctsipP8upquZBV7SWvXpILWYULWYUL3vC/mpnYh5n4EjMR3vCHN/zzJAc1GzUTS3AMX6+E"
    "JVmJ7FOWLJSs1ExEVinvPUnW60fWezIob04zO3OZncg6Zc/uZM/3Av1bG8p3rcl3d5PvMsl2rcl2d5PtMn1O5TJLhzJLxzJLkW3K"
    "qn8PlFXLkVXvlSzVTESWPMAs6c0suY1Zoiz3bTCNuZrHXG3EXF1omjNbpzFb85itjZityDJl2wtk21PUYymy7WzJVs1EZMPjzIYx"
    "zAblt9+T3xqT3TqS3YaT3WKy2+/Jbo19zii3dSS3DSe3xeS235PbGkt2KQsfIAt3IguvJgs/Rha+KdTfy6L4lkd8m0p8a098q0d8"
    "m0V8e5Z5m8e8ncq8bc+8rce8ncW8fdZn4h+ZvVcze/swe19i9mr8WYyPHnnm5/DncJlcy8KTGR0D2KFgSFDK28fhof+Pqy+B16la"
    "/08yvHu/a639Duc4jlCZyhAhbsk/qW6iZMzM/YkkQ9JwCKGDihQqruToZmwgYzeEMhy54WfmyFSUTJdfEUr1X9/n+Z57b7fnszvv"
    "2ftZz7vfvZ7n+/2uZ53j+P8/Jq+OyWt8h9fHvJ7vJ3Yi/oJYq/izZnQ4OjzvtcOOcEf4hW0VHx3i3Pn4jhDnvrBj5RzO4hzO4txY"
    "29XcLzbAdBGrY8aInY/vlmq+zfzD1I/X95Uyx6yPr48Xc+f9eVy7zZ/HtVf8eVwr5qrZ8zJmd/w2GfMP84qMmWOKyd+AaWgmidXz"
    "eQerYv4i1pya6k3zulTpy2asmRqfGs/2eb8gviB+wccp4+1G96av49fFY2ocPtn+Onwu+KuviMcd4gOvlyXGWJMtMR41FyTGK+Jz"
    "h7vVVLWw3eY5sXtNgdh0M0LsvNHszzPDZa+rpc+tivGK8ZfM9fYmc5PPuKPmkfgjvnZ2m4HxgfFvXZ73g29L7wffl7wXfJt5P/i2"
    "837w/db9Jr7wbikx5xv4wruZxDxq2knM3Qa+v7kvzGGx381nYkfMBrGuRjt51eSv5Rww/8/ME7tsuvsn+Xp8mjlpPox/GP/SfGpW"
    "xFfEt5lT0nH81JzxK4P743P9p9vnUaFZ4rL3x5hp3h9jvvT+GLPNe5+SEffHMWau98eYZomfo8vyHt3NNHmPk+ZLeY9PzTZ5j1My"
    "BqPmynvkGYz5OSrMeByPxTWvm8Y1+/eaq5j/x4y+Rs7j9THzrNH8TTJ/+8W1Cl5gFXxhR4daBztYB2OtnkPO72Al4NxYuzuuefoP"
    "o3k6h3l6v9Fq6MJqGMNqKOZ+D7Ue6se1HtazHqpZvYacrx/XiljPisC1avb1uObjWKP5+Cjz8RXm4ySjVVGEVfEXVsX7rIob3fVx"
    "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"
    "XSj32CbKl3v8UGIiqt7jJImJqKvlHn+VmIhaIPfYKbFY7vFDibm1dCHD4virUR7tZ5Rtd/rvlG/PWH0NjsXrMx6PlS9jVvny/5F1"
    "G5J1K7lTceXdPxvl3becngPH/tko8+LcW+5Gq7z4GHmxH3mxtFX2/ZDs+z3Zt4lraZR/c8i/J8m/tZxeA8fmkIFPkoFxrZZLGuW/"
    "M+S/xlb5L8Mp/xW1ysK/GGXhzlZZ+ChZeFZUGKEXeTiPPFzUKg93iv7oA47NIxMXtcrE8OkU3WyU59pY5bkW5LmD5LmqTnluhVE2"
    "HmmVjeeSje9yysbtyMaDE84oHy8hHw8nH28kH79glY9zIvUFxy4hIw8nI28kI79glZHhmxONNcpnf7LKZ2ny2UTyWV2nfNaafFbH"
    "KitXIyt3JCsHZOWhZOWkU1YemexslJebkJcfIC9/QF4ub5WXB5KXKyR1DDi2CZn5ATLzB2Tm8laZeSCZGWMqJOta5a1V5K1byFsd"
    "yVspp7y1jLwVi5S37nbKzmmyc1+ycymy8xqycz7ZOYqUnc+kMq3y8zXk58nk5z1G+bmbVX7OIj8fJz/vS+lYcOw1ZOjJZGiMBcd2"
    "s8rQWWTo42RojN2XWmaVn8Y45adHyU9brPJTHaf8VDZSfqpFflpDfro+UpY+6ZSl55Klt5KlG0TK0hXI0hFZ+hRZ+kTGXVZ5eiF5"
    "eiZ5GgwJjv3GKk+3c8rTjcnTL5KnR2ZoDHDsQjL1TDK1suwW+41Vpm7nlKkbk6lfJFMjxsiMak55aDp56AXyUJlIeWiaUx56OFIe"
    "6kAe+oU8lEooD513ytZBpGydSbaeS7Z+jmw9hWy9mmz9Gdn69YSydbGsGk75eh35ugH5OoN8/TD5egb5+jfydTqhfF0iUr7eWEpj"
    "gWPXkbEbkLEzyNgPk7FnkLF/I2MjFji2RKSMjVgbS5WOlG/akG8+JN+0IN8UcmJ78s2v5JswoXzTiXzzIfmmIFLWvkDWfpusXSKh"
    "rF2FrL09UtbOSyhrHyVr1yNrzyJrzy9dO1LePu+Ut1+NlLffJG9v4z2WJm8PJG/3J2+PJm8XSyhvby2tMcGxiAmOfTVS5n6TzL2N"
    "zF2azD2QzN2fzD2azF0socyNmFtLj8sel50okyjzZfaX2UXKFCmT4e1S9qXsamWqldmVvSvblVHbkb0jO7tMdpmD2Qezq5SpUqYg"
    "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"
    "2ezXSieoOIpQcej9ZvB+/zOyKo4qVBwDqDj+eL/neb+I/FppG9nodofjdndafnv3or3Z3W9fNi+bd93N7mWD1++6TfIa3+H1JtfP"
    "tnewb+zVDjbZTrBbzVazx2sL/MxN0Wiy/x7n9vjv2sqZ8m6y+Eywe8SnrZwr707aH8VKuUsWdrtHMlieLWv3m/2mnXtTfpf5YZft"
    "8NvK3aI8fx7X2rnK9k25Ms/iWrfofZcnY8paXMPVh2VMtsO1990ouftNzvL34G70OA+b6dpFsCv2Vs/8LWxnz7r4axKp6B63wq6w"
    "GdG7brldbsPEFX/1VvHYYOCTilZY+GREyy18wkSj6IrEuFV84AUfeMEHXvBpFHV0av/n9opVjYqIfeJ6RLAKTn+67Yq7yuEvIH7h"
    "nzd+Uuk5d72bYqfYKlGW6+UtjOa5w15PHE1c8X7w/cL7wfc57wffKlEvlyWehy18jyZWR1ck5lXuC4nZ3j0nMa938IU3fOENX3jX"
    "iWA/uTERbJHrL7W7SCr8TV+dT4mN9yse2FEXk9+yLnAfeQZt6Nkhx5+JudrRAMH9f0QF8neeX4x6RXvcHnc5edR7x2REQxlz2H+X"
    "IyNquAEyoqYrkBF7HMZcTm6QMRhVIO/xkYzBKIzBKIzBqBflPXpFGLMh2czjLqx1pMixgUy1PDon9nDUR6xk4mQEq5pYJFYh0twv"
    "8M9njLfbo1zhrC7R39xObw2iYVEz18wVSfTReImiifei96J56QrRJveujBwjY2/33JIrI3e6v8nIZg5ji/j36CMj34swdl66hozF"
    "6AJ5z14yFqMxFqMbyHsOizAWo5fLexZNYGyN9NOJQWIFieZicxIjxT5MzBVblLggtijxrdiQZCexe1LTkrCV0dfRNYlrElUTS6Jv"
    "vcUSA6KD3qokPo0ueTsWdY+Geuuf6JzAv/BQNflUYoS3ocn18vdUGmau9OMRo2ri22iJRDgYDZAIl6JPJcJQidHfj0eMqskRiack"
    "QlGJ0TDzMYmBKIiBKIiBKIiBKMfkHrpLDERBDEQZmtS/MYYYj2Xem3xbrCAxLAm7P9lV7NmkftZ2yW1iM5NDxN5IThWrl7pFbG/q"
    "VbHhnu1fSryUGJC4GK1MrEwcSjyfmO2td+IN/3znJF5NrI+2JrYmLifWJXp4W5jcncD79E2VTY7zdjxVP3U4eTj5cdZwHwWxBvgo"
    "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"
    "ahpuZiWOYiWigkayFiuyFruyFnuxFhuJFWemv85M78lMr81Mz2Kmq7KaHaiy2h6qsvqZympToMqqla8G1VWzA9VV20PVVT9TV8FT"
    "KxKV8xZrsgVr8nXWZC/WZAPW5FAx1VRHqanOBa2Z0dnM6OrM6A7MaFVUR6mozgX3sS7zWJcp1qUqqh4+/1VPHaWeOhfcx8rMY2Wm"
    "WJmqp3pIxagy7Epl2J7KsAeV4Vgqw46hKsPTUplaZagY5aNR5KPl5KPF5KMHyUfdyEf/w9p8iLU5irW5nLW5mLX5IGuzG2vzf8xf"
    "A32/h/h+o/h+y/l+i/l+D/L9uvH9MPKvAT7h9ADH9KBCWFSsRTguKOGtp39VIhgnr2rKa3zXU6p5edAwhL0T3i52OagcZgfZwZ/D"
    "NWG5oFwwzJ/JlnN/9t/j3LBwbYBzOPtn8VkT4txav1Y+EcAmhv8j9nXYW6x4eDLYEtsSmxu2jg8KBgVLw9fidcO6YTJePNwSw7W5"
    "4aAA15b6s7iWjPcIcQ1XcQ1XcQ1Xca1HWNXnGuy58E2xh8MrYivCy2LvBakQ+wXt/Jmxwdjgs/BkWNLbnfF6cexS/OI9rhWfdv46"
    "fD7zV0+Kx6wAPr+Es8QHXu0kxorwM4lxUnzgBZ9ZftU6WuxAeKvYXZ61YadD+VHFeMX46hCWDkuHx2LHYtt87T8bPBs8Eh8f4ueD"
    "3g0fjCMjN4fn5DPu8Z7HYvDd5v3g+0j8/wL4vuu94LvZe537/12dCXiVxdmGNRUMmZnvO+eQBMIaKRLwkkVAFASBsgipIFABE9Ma"
    "CiiCIPgTiI0iS1hFoRYKVAHFILEKlX0LiEDRUlBsIRWaEHYkIhBSfxcW/3ne9+HX6nudi+Scb97zCZnnvmcmAbnylFyLqw9IzzEG"
    "1+LqZfJV3svuk56X5NpTZrwplMoxSRb1J1PNojqZg1LlZq3Uh+Ztqd3mVj1hMktk12WN/xW7Lml2l0+3NPNvs9DP7lqmmdlq3kl4"
    "J2G33e2vvlVGNJUxa/z1GJNm08wuGVFLxjTz12PMbjtAxmDUBnmPJTIGozAGo/4t77FQxmAUxgyw35unLKrEvGRRZ00fi9pht0lV"
    "sculJpqbLOp5Ju5o+wuL78eoYq+Z23xNtd+Z9r7a2v4WOyXptrH9m0/M06a57WF6mK9NHYtzptRwtB+HsVXsbeaajGxvvpORvQ3G"
    "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"
    "TGhdd+dJsj7AjC5iRusOwGHTgRmdLKVr7t5cc3/ENfcco2vuB+0sJqCuuHtzxf0RV9xzTFfmdCpzOt7qers319sfcb09x3RlUqcy"
    "qfHVrDsDFxJWMKuPMKt1Z6AjdwaaWd3v1LX2dq61/8y1dq65jUl3F5NO19oxqyvt7Vxp/5kr7VyjK+0+XGnP5ko75lNR19nbuc7+"
    "M9fZuUbX2X24zp7NdXZM1tm6Y5DNHYOXjO4YdLK/YmYPYGY/xsw+IKVr7Ju5xs7hGvsK19j7ucbuyzX2vVxjr+UK+2ausHO4wr7C"
    "FfZ+rrD7coV9L1fYa30OjmJy92By6/r6CtfX+7m+7sv19b1cX6+VBNWdhE7cSRjKnYSnuJMwjTsJM7iTsJQ7CWMluTV9kVyNmVzd"
    "mVy6tu7GtXUbrq1rOF1bn+XaeovVlXV9rqzzubLuwpV1N66s23BlXcM1YX6nMr8XReOZ4Lquzue6ugvX1d24rm7DdXUN14QJnsoE"
    "R3LqDkMmdxhu5w5DU+4wdOcOw++4w5DtkpnhOcxwJLimL5JTffUEffV6TH3V0FeP01fz6KsN6Kutq6uvroxphk+srhl+ghl+PaYZ"
    "bpjhx5nheczwBszw1tUrmeNvRb9jkuv9nOD9oJMmeQqT/B4mud5PA94POmmS4+9ijLNx9gWDxwtmruloUUfNlwY/m33SHPW/4uOT"
    "pkQ+xmf4uMQM93mAGmCHSbWyd1rk/ghb1+aYHPMJvl8vAc+N8J/juU/scdNKrrnTjpBr6lo8d9x0kZ+WaGVft4ulSu0DDlXpv+rb"
    "mrZmnc++uf7ufuP/hDeZTeYDW+mfxWvr/PN47Tf+ebz2gW0kr+HVdTLmfvsbGbPE4rVGtovZLfW16WpRi6yeJPza1nSor80+g598"
    "+8y/VplQmbDc5to5Pgub2s020dcC+7V/Hdd85l/HNcv967imqX91s1zxgVyDqz6THl3tcumRa5tKj81yzQe2hZkhdYQnM783oyyq"
    "kT0utdYOlKow9xn8PTQlnn4vJrzo/zTeMw/6Svb/byBid1vb4vSpu78ykoBrS/x1uPakv+o9ubLA4Nru/jpc293PrgrpeZ8pkZ4F"
    "5qT0fE+uxdXdpWdt291q7n8u9YL/nUSN8H96qEFMkt22ukO9Z/tJHfPUx/dyTjQLTIqv53xX/DzYWP91A3dY4XML/TJsgcXXYA13"
    "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"
)

# ===================================================================== params
PARAMS = {"allin_spr": 1.6, "d0": 0.012, "d1": 0.012, "dstreet": 0.01, "r_ip": 1.03, "r_oop": 0.92,
          "budget": 0.12, "allin_eq": 0.80}

# ===================================================================== tables

def _load_pf():
    raw = zlib.decompress(base64.b64decode(_PF_BLOB))
    arr = array.array("H")
    arr.frombytes(raw)
    if sys.byteorder != "little":
        arr.byteswap()
    n = 22100
    cols = [[arr[j * n + i] / 10000.0 for i in range(n)] for j in range(4)]
    order = sorted(range(n), key=cols[0].__getitem__)
    pct = [0.0] * n
    for r, i in enumerate(order):
        pct[i] = r / (n - 1.0)
    return cols, pct


PF_COLS, PF_PCT = _load_pf()
PF_GRID = (1.0, 0.5, 0.25, 0.12)
_LG = [math.log(c) for c in PF_GRID]


def pf_eq(idx, c):
    """Preflop equity of combo idx against the top-c fraction of 3-card hands."""
    if c >= 1.0:
        return PF_COLS[0][idx]
    lc = math.log(max(c, 0.04))
    for j in range(3):
        if lc >= _LG[j + 1]:
            t = (_LG[j] - lc) / (_LG[j] - _LG[j + 1])
            return PF_COLS[j][idx] * (1 - t) + PF_COLS[j + 1][idx] * t
    # extrapolate below 12%
    t = (_LG[3] - lc) / (_LG[2] - _LG[3])
    v = PF_COLS[3][idx] - t * (PF_COLS[2][idx] - PF_COLS[3][idx])
    return max(0.02, v)


try:
    _fill_rank_table((7, 6, 5), deadline=_T0 + 3.0)
except Exception:
    pass


def _sig(x):
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def log(*a):
    try:
        print(*a, file=sys.stderr)
    except Exception:
        pass


# ============================================================= opponent model

class Freq:
    __slots__ = ("c", "n")

    def __init__(self, prior, strength):
        self.c = {k: v * strength for k, v in prior.items()}
        self.n = 0

    def add(self, k):
        if k in self.c:
            self.c[k] += 1
            self.n += 1

    def p(self, k):
        return self.c[k] / sum(self.c.values())


class MeanStat:
    __slots__ = ("s", "w", "n")

    def __init__(self, prior, strength):
        self.s = prior * strength
        self.w = strength
        self.n = 0

    def add(self, x):
        self.s += x
        self.w += 1
        self.n += 1

    def mean(self):
        return self.s / self.w


FOLD_ADJ = (0.85, 1.0, 1.12, 1.25)


def size_bucket(frac):
    if frac < 0.45:
        return 0
    if frac < 0.8:
        return 1
    if frac < 1.2:
        return 2
    return 3


class OppModel:
    def __init__(self):
        self.pf_open = Freq({"raise": 0.45, "call": 0.35, "fold": 0.20}, 6)
        self.pf_vs_raise = Freq({"fold": 0.35, "call": 0.45, "raise": 0.20}, 6)
        self.pf_vs_limp = Freq({"check": 0.6, "raise": 0.4}, 5)
        self.post_open = Freq({"check": 0.6, "raise": 0.4}, 6)
        self.post_face = Freq({"fold": 0.33, "call": 0.52, "raise": 0.15}, 6)
        self.post_face_rr = Freq({"fold": 0.22, "call": 0.53, "raise": 0.25}, 6)
        self.pf_vs_3bet = Freq({"fold": 0.25, "call": 0.50, "raise": 0.25}, 6)
        self.fold_b = [[0.0, 0] for _ in range(4)]  # folds, n per size bucket
        self.open_sz = [0, 0, 0, 0]   # opp bets by size bucket (when not facing a bet)
        self.face_sz = [0, 0, 0, 0]   # opp raises by size bucket (facing our bet)
        self.pf_honest = MeanStat(0.74, 3)
        self.post_honest = MeanStat(0.80, 3)
        self.disc_match = MeanStat(0.75, 4)
        self.sd_bet_river = []   # opp hand pct at showdown when they bet/raised the river
        self.sd_call_river = []  # opp hand pct at showdown when they called our river bet
        self.hands = 0
        self.vpip = 0
        self.pfr = 0
        self.showdowns = 0

    # -- derived
    def alpha_pf(self):
        return min(1.0, max(0.05, (self.pf_honest.mean() - 0.5) / 0.24))

    def alpha_post(self):
        return min(1.0, max(0.05, (self.post_honest.mean() - 0.52) / 0.28))

    def rational(self):
        return min(1.0, max(0.0, (self.disc_match.mean() - 1 / 3.0) / (2 / 3.0)))

    def fold_prob(self, frac):
        base = self.post_face.p("fold")
        b = size_bucket(frac)
        f, n = self.fold_b[b]
        prior = min(0.95, base * FOLD_ADJ[b])
        m = 4.0
        return min(0.95, max(0.0, (f + m * prior) / (n + m)))

    def sized_freq(self, open_ctx, rsize):
        """Frequency with which the opponent makes a bet/raise of this size bucket."""
        b = size_bucket(rsize)
        if open_ctx:
            st, cnt, prior = self.post_open, self.open_sz, (0.30, 0.40, 0.22, 0.08)
        else:
            st, cnt, prior = self.post_face, self.face_sz, (0.20, 0.40, 0.25, 0.15)
        tot = st.p("raise")
        nr = sum(cnt)
        share = (cnt[b] + 4.0 * prior[b]) / (nr + 4.0)
        return max(0.02, min(tot, tot * share))

    def raise_prob(self):
        return self.post_face.p("raise")

    def opp_raised_street(self, ev, street):
        for e in ev:
            if e["actor"] == "opp" and e["street"] == street and e["action"] == "raise":
                return True
        return False


def replay(history, my_seat):
    """Annotate history entries with the betting context the actor faced."""
    me_btn = my_seat == "button"
    bets = {"me": 1 if me_btn else 2, "opp": 2 if me_btn else 1}
    committed = dict(bets)
    street = "preflop"
    out = []
    first = {"me": True, "opp": True}
    raised = {"me": False, "opp": False}
    for h in history:
        st = h.get("street")
        if st != street:
            street = st
            bets = {"me": 0, "opp": 0}
            raised = {"me": False, "opp": False}
        a = h.get("actor")
        if a not in ("me", "opp"):
            continue
        o = "opp" if a == "me" else "me"
        to_call = bets[o] - bets[a]
        pot = committed["me"] + committed["opp"]
        act = h.get("action")
        e = {"street": st, "actor": a, "action": act, "to_call": to_call, "pot": pot,
             "reraise": raised[a]}
        if st == "preflop":
            if first[a] and to_call == 1 and bets[o] == 2:
                e["ctx"] = "open"
            elif to_call == 0:
                e["ctx"] = "vs_limp"
            else:
                e["ctx"] = "vs_raise"
            first[a] = False
        else:
            e["ctx"] = "open" if to_call == 0 else "face"
            if to_call > 0:
                e["frac"] = to_call / max(1.0, pot - to_call)
        if act == "call":
            add = to_call
        elif act == "raise":
            amt = h.get("amount") or bets[a]
            add = amt - bets[a]
            e["rsize"] = (amt - bets[o]) / max(1.0, pot + to_call)
        else:
            add = 0
        bets[a] += add
        committed[a] += add
        if act == "raise":
            raised[a] = True
        out.append(e)
    return out


# ======================================================================= bot

SIG_T = 0.10


class Player:
    def __init__(self):
        self.model = OppModel()
        self.total_hands = 200
        self.hand_id = 0
        self.seat = "button"
        self.hole3 = []
        self.hole2 = None
        self.my_disc = None
        self.opp_disc = None
        self.rng = random.Random()
        self.net = 0
        self.mc_time = 0.0
        self.n_dec = 0
        self.n_mc = 0

    # ---------------------------------------------------------------- messages
    def on_init(self, msg):
        self.total_hands = int(msg.get("total_hands", 200) or 200)

    def on_new_hand(self, msg):
        self.hand_id = int(msg.get("hand_id", self.hand_id + 1))
        self.seat = msg.get("seat", "button")
        self.hole3 = [c2i(c) for c in msg.get("hole_cards", [])]
        self.hole2 = None
        self.my_disc = None
        self.opp_disc = None

    def on_discard_result(self, msg):
        try:
            self.my_disc = c2i(msg["my_discard"])
            self.opp_disc = c2i(msg["opp_discard"])
            self.hole2 = [c2i(c) for c in msg["hole_cards"]]
        except Exception:
            pass

    def on_hand_result(self, msg):
        try:
            self.net += int(msg.get("net", 0))
        except Exception:
            pass
        try:
            self._learn(msg)
        except Exception as ex:
            log("learn error", repr(ex))

    def _learn(self, msg):
        m = self.model
        m.hands += 1
        ev = replay(msg.get("history") or [], self.seat)
        pf_raised = False
        pf_vol = False
        post_aggr = False
        for e in ev:
            if e["actor"] != "opp":
                continue
            act = e["action"]
            if e["street"] == "preflop":
                ctx = e["ctx"]
                if ctx == "open":
                    m.pf_open.add(act)
                elif ctx == "vs_limp":
                    m.pf_vs_limp.add(act)
                elif e["reraise"]:
                    m.pf_vs_3bet.add(act)
                else:
                    m.pf_vs_raise.add(act)
                if act in ("call", "raise"):
                    pf_vol = True
                if act == "raise":
                    pf_raised = True
            else:
                if act == "raise" and not e["reraise"]:
                    sb = size_bucket(e.get("rsize", 0.66))
                    if e["ctx"] == "open":
                        m.open_sz[sb] += 1
                    else:
                        m.face_sz[sb] += 1
                if e["ctx"] == "open":
                    m.post_open.add(act)
                elif e["reraise"]:
                    m.post_face_rr.add(act)
                else:
                    m.post_face.add(act)
                    b = size_bucket(e.get("frac", 0.66))
                    m.fold_b[b][1] += 1
                    if act == "fold":
                        m.fold_b[b][0] += 1
                if act == "raise":
                    post_aggr = True
        if pf_vol:
            m.vpip += 1
        if pf_raised:
            m.pfr += 1
        oh = msg.get("opp_hole_cards")
        od = msg.get("opp_discard")
        board = [c2i(c) for c in (msg.get("board") or [])]
        if oh and od and len(board) == 5:
            m.showdowns += 1
            o2 = [c2i(c) for c in oh]
            odc = c2i(od)
            o3 = o2 + [odc]
            p = PF_PCT[combo_index(*o3)]
            if pf_raised:
                m.pf_honest.add(p)
            flop = board[:3]
            fk = KEY[flop[0]] + KEY[flop[1]] + KEY[flop[2]]
            _, d = best_keep(o3, flop, fk)
            m.disc_match.add(1.0 if d == 2 else 0.0)
            riv = [e for e in ev if e["street"] == "river"]
            r_bet = any(e["actor"] == "opp" and e["action"] == "raise" for e in riv)
            r_call = (not r_bet) and any(e["actor"] == "opp" and e["action"] == "call" for e in riv)
            if post_aggr or r_bet or r_call:
                known = set(o3 + board + self.hole3)
                pv = self.hand_pct(o2, board, known, 200)
                if post_aggr:
                    m.post_honest.add(pv)
                if r_bet:
                    m.sd_bet_river.append(pv)
                if r_call:
                    m.sd_call_river.append(pv)

    def hand_pct(self, h2, board, known, n):
        """Strength of a 2-card hand on a full board vs random 2-card hands."""
        rest = [c for c in range(52) if c not in known]
        bk = 0
        for c in board:
            bk += KEY[c]
        cards = list(board)
        ov = ev_key(bk + KEY[h2[0]] + KEY[h2[1]], cards + list(h2))
        win = 0.0
        rs = self.rng.sample
        for _ in range(n):
            a, b = rs(rest, 2)
            v = ev_key(bk + KEY[a] + KEY[b], cards + [a, b])
            if ov > v:
                win += 1
            elif ov == v:
                win += 0.5
        return win / n

    def emp_eq(self, lst, my_pct):
        """(equity, weight) of my_pct against an empirical list of opponent pcts."""
        if not lst:
            return 0.5, 0.0
        recent = lst[-40:]
        t = 0.0
        for s in recent:
            t += _sig((my_pct - s) / 0.015)
        n = len(recent)
        return t / n, n / (n + 6.0)

    # ------------------------------------------------------------- budgeting
    def budget(self, time_left_ms):
        hands_left = max(1, self.total_hands - self.hand_id + 1)
        per_hand = time_left_ms / hands_left
        # expected Monte Carlo decisions per hand, learned online (prior 4)
        dph = (self.n_mc + 20.0) / (max(0, self.hand_id - 1) + 5.0)
        b = per_hand * PARAMS["budget"] / max(1.0, dph)
        if PARAMS.get("debug"):
            log("BUD tl=%d hl=%d dph=%.2f b=%.1f" % (time_left_ms, hands_left, dph, b))
        if time_left_ms < 8000:
            b = min(b, per_hand * 0.15, 6.0)
        if time_left_ms < 2500:
            b = min(b, 1.5)
        return max(0.5, min(b, 220.0)) / 1000.0

    # ------------------------------------------------------- range factors
    def _pf_factor_fn(self, ev):
        """Return f(pct)->weight for opponent preflop actions this hand."""
        m = self.model
        alpha = m.alpha_pf()
        steps = []
        rem = 1.0
        for e in ev:
            if e["actor"] != "opp" or e["street"] != "preflop":
                continue
            ctx, act = e["ctx"], e["action"]
            if ctx == "open":
                fr = m.pf_open.p("raise")
                fc = fr + m.pf_open.p("call")
                if act == "raise":
                    rem2 = rem * fr
                    steps.append(("aggr", 1 - rem2, 0))
                    rem = rem2
                elif act == "call":
                    steps.append(("cont", 1 - rem * fc, 1 - rem * fr))
                    rem = rem * fc
            elif ctx == "vs_limp":
                fr = m.pf_vs_limp.p("raise")
                if act == "raise":
                    rem2 = rem * fr
                    steps.append(("aggr", 1 - rem2, 0))
                    rem = rem2
                else:
                    steps.append(("check", 1 - rem * fr, 0))
            else:
                st = m.pf_vs_3bet if e["reraise"] else m.pf_vs_raise
                fr = st.p("raise")
                fc = fr + st.p("call")
                if act == "raise":
                    rem2 = rem * max(fr, 0.03)
                    steps.append(("aggr", 1 - rem2, 0))
                    rem = rem2
                elif act == "call":
                    steps.append(("cont", 1 - rem * fc, 1 - rem * max(fr, 0.03)))
                    rem = rem * fc
        rem = max(rem, 0.04)
        means = [_gmean(kind, t1, t2) for kind, t1, t2 in steps]
        tab = []
        for i in range(101):
            p = i / 100.0
            f = 1.0
            for (kind, t1, t2), mu in zip(steps, means):
                f *= (1 - alpha) * mu + alpha * max(_gfun(kind, p, t1, t2), 0.02)
            tab.append(f)
        return tab, rem, alpha

    def _post_steps(self, ev):
        m = self.model
        fb = m.post_open.p("raise")
        steps = []
        rem = 1.0
        for e in ev:
            if e["actor"] != "opp" or e["street"] == "preflop":
                continue
            act = e["action"]
            if e["ctx"] == "open":
                if act == "raise":
                    rem2 = rem * m.sized_freq(True, e.get("rsize", 0.66))
                    steps.append(("aggr", 1 - rem2, 0))
                    rem = rem2
                elif act == "check":
                    steps.append(("check", 1 - rem * fb, 0))
            else:
                st = m.post_face_rr if e["reraise"] else m.post_face
                ff = st.p("fold")
                fr = max(st.p("raise"), 0.03)
                if not e["reraise"]:
                    fr = max(0.02, m.sized_freq(False, e.get("rsize", 0.66)))
                if act == "raise":
                    rem2 = rem * fr
                    steps.append(("aggr", 1 - rem2, 0))
                    rem = rem2
                elif act == "call":
                    fc = 1 - ff
                    steps.append(("cont", 1 - rem * fc, 1 - rem * fr))
                    rem = rem * fc
            rem = max(rem, 0.06)
        return steps

    # ------------------------------------------------------------ Monte Carlo
    def monte_carlo(self, board, ev, budget_s):
        t_start = time.perf_counter()
        deadline = t_start + budget_s
        self.n_mc += 1
        pft, _, _ = self._pf_factor_fn(ev)
        r = self.model.rational()
        nb = len(board)
        flop = board[:3]
        fk = KEY[flop[0]] + KEY[flop[1]] + KEY[flop[2]]
        bk = 0
        for c in board:
            bk += KEY[c]
        shuf = self.rng.shuffle
        rnd = self.rng.random
        scores = []
        pw = []
        res = []
        if self.hole2 is None:
            # pre-discard: opponent holds 3 cards, both will discard
            h = self.hole3
            keeps = ((h[1], h[2]), (h[0], h[2]), (h[0], h[1]))
            kk = [KEY[x] + KEY[y] for x, y in keeps]
            dead = set(h + board)
            deck = [c for c in range(52) if c not in dead]
            res3 = ([], [], [])
            need = 5 - nb
            kdraw = 3 + need
            nd = len(deck)
            pos = nd
            i = 0
            while True:
                if (i & 15) == 0 and i >= 64 and time.perf_counter() > deadline:
                    break
                if i >= 4000:
                    break
                i += 1
                if pos + kdraw > nd:
                    shuf(deck)
                    pos = 0
                s = deck[pos:pos + kdraw]
                pos += kdraw
                o = s[:3]
                sc, d = best_keep(o, flop, fk)
                if r < 1.0 and rnd() > r:
                    d = int(rnd() * 3) % 3
                oa = o[1] if d == 0 else o[0]
                ob = o[1] if d == 2 else o[2]
                b5 = board + s[3:]
                k5 = bk
                for c in s[3:]:
                    k5 += KEY[c]
                ov = ev_key(k5 + KEY[oa] + KEY[ob], b5 + [oa, ob])
                for j in range(3):
                    x, y = keeps[j]
                    mv = ev_key(k5 + kk[j], b5 + [x, y])
                    res3[j].append(1.0 if mv > ov else (0.5 if mv == ov else 0.0))
                scores.append(sc)
                pw.append(pft[int(PF_PCT[combo_index(o[0], o[1], o[2])] * 100)])
            result = EqResult(scores, pw, None, self._post_steps(ev), self.model.alpha_post())
            eqs = [result.eq_with(res3[j]) for j in range(3)]
            best = max(range(3), key=lambda j: (eqs[j] + (0.0005 * (12 - (h[j] >> 2)) / 12.0)))
            result.set_res(res3[best])
            result.keep_eqs = eqs
            result.best_discard = best
        else:
            h2 = self.hole2
            od = self.opp_disc
            dead = set(self.hole3 + board)
            if self.my_disc is not None:
                dead.add(self.my_disc)
            if od is not None:
                dead.add(od)
            deck = [c for c in range(52) if c not in dead]
            need = 5 - nb
            mk = KEY[h2[0]] + KEY[h2[1]]
            myv = ev_key(bk + mk, board + h2) if need == 0 else 0
            match_w = r + (1 - r) / 3.0
            miss_w = (1 - r) / 3.0 + 0.02
            kdraw = 2 + need
            nd = len(deck)
            pos = nd
            i = 0
            while True:
                if (i & 15) == 0 and i >= 64 and time.perf_counter() > deadline:
                    break
                if i >= 4000:
                    break
                i += 1
                if pos + kdraw > nd:
                    shuf(deck)
                    pos = 0
                s = deck[pos:pos + kdraw]
                pos += kdraw
                a = s[0]
                b = s[1]
                w = 1.0
                if od is not None:
                    _, d = best_keep((a, b, od), flop, fk)
                    w = match_w if d == 2 else miss_w
                    w *= pft[int(PF_PCT[combo_index(a, b, od)] * 100)]
                sc = hscore(a, b, board, bk, nb)
                if need:
                    b5 = board + s[2:]
                    k5 = bk
                    for c in s[2:]:
                        k5 += KEY[c]
                    mv = ev_key(k5 + mk, b5 + h2)
                    ov = ev_key(k5 + KEY[a] + KEY[b], b5 + [a, b])
                else:
                    mv = myv
                    ov = ev_key(bk + KEY[a] + KEY[b], board + [a, b])
                res.append(1.0 if mv > ov else (0.5 if mv == ov else 0.0))
                scores.append(sc)
                pw.append(w)
            result = EqResult(scores, pw, res, self._post_steps(ev), self.model.alpha_post())
        el = time.perf_counter() - t_start
        self.mc_time += el
        if PARAMS.get("debug"):
            log("MC budget %.1fms took %.1fms n=%d" % (budget_s * 1000, el * 1000, len(scores)))
        return result

    # --------------------------------------------------------------- actions
    def act(self, req):
        legal = req.get("legal_actions") or []
        street = req.get("street", "preflop")
        ev = replay(req.get("history") or [], self.seat)
        if street == "preflop":
            return self.act_preflop(req, legal, ev)
        board = [c2i(c) for c in req.get("board") or []]
        tl = float(req.get("time_left_ms", 30000) or 0)
        if tl < 600:
            return self.panic_action(req, legal, board)
        budget = self.budget(tl)
        eqr = self.monte_carlo(board, ev, budget)
        if street == "river" and self.hole2 is not None and len(board) == 5:
            try:
                known = set(self.hole3 + board)
                if self.opp_disc is not None:
                    known.add(self.opp_disc)
                my_pct = self.hand_pct(self.hole2, board, known, 200)
                m = self.model
                if req.get("to_call", 0) > 0 and m.opp_raised_street(ev, "river"):
                    e, w = self.emp_eq(m.sd_bet_river, my_pct)
                    if w > 0:
                        eqr.eq = (1 - w) * eqr.eq + w * e
                        eqr.emp_call = None
                else:
                    e, w = self.emp_eq(m.sd_call_river, my_pct)
                    if w > 0:
                        eqr.emp_call = (e, w)
            except Exception as ex:
                log("emp error", repr(ex))
        return self.act_post(req, legal, street, eqr, ev)

    def _sizes_post(self, req, fracs, allow_allin=True):
        pot = req["pot"]
        to_call = req["to_call"]
        mn = req["min_raise"]
        mx = req["max_raise"]
        out = set()
        for f in fracs:
            x = int(round(req["opp_bet"] + f * (pot + to_call)))
            x = max(mn, min(mx, x))
            if mx - x < 0.35 * (pot + (x - req["my_bet"]) + (x - req["opp_bet"])):
                x = mx
            out.add(x)
        if allow_allin or mx - req["my_bet"] <= PARAMS["allin_spr"] * (pot + to_call):
            out.add(mx)
        return sorted(out)

    def act_post(self, req, legal, street, E, ev):
        m = self.model
        pot = req["pot"]
        to_call = req["to_call"]
        my_bet = req["my_bet"]
        opp_bet = req["opp_bet"]
        my_stack = req["my_stack"]
        opp_stack = req["opp_stack"]
        ip = self.seat == "button"
        eq = E.eq
        alpha = E.alpha
        if street == "river":
            R = 1.0
        elif street == "turn":
            R = 0.99 if ip else 0.95
        else:
            R = 0.98 if ip else 0.93
        cands = []
        if to_call > 0:
            if "fold" in legal:
                cands.append((0.0, "fold", None))
            if "call" in legal:
                cands.append((eq * R * (pot + to_call) - to_call, "call", None))
        else:
            if "check" in legal:
                cands.append((eq * R * pot, "check", None))
        rr = m.opp_raised_street(ev, street)
        if "raise" in legal and "min_raise" in req:
            fracs = (0.33, 0.5, 0.75, 1.0, 1.5) if to_call == 0 else (0.6, 0.85, 1.2)
            for x in self._sizes_post(req, fracs, E.eq >= PARAMS["allin_eq"]):
                add = x - my_bet
                oadd = x - opp_bet
                if add > my_stack:
                    continue
                frac = oadd / max(1.0, pot + to_call)
                allin = (x == req["max_raise"]) or oadd >= opp_stack
                if rr:
                    pf = m.post_face_rr.p("fold") * (0.85 + 0.15 * min(frac, 2.0))
                    pr = m.post_face_rr.p("raise")
                else:
                    pf = m.fold_prob(frac)
                    pr = m.raise_prob()
                if allin:
                    pr = 0.0
                pc = max(0.0, 1.0 - pf - pr)
                eq_c = alpha * E.top(1.0 - pf) + (1 - alpha) * eq
                if E.emp_call is not None and not rr:
                    ee, ww = E.emp_call
                    eq_c = (1 - ww) * eq_c + ww * ee
                delta = PARAMS["d0"] + PARAMS["d1"] * min(frac, 3.0)
                if street != "river":
                    delta += PARAMS["dstreet"]
                fpot = pot + add + oadd
                ev_c = (eq_c - delta) * (R if not ip else 1.0) * fpot - add
                evv = pf * pot + pc * ev_c
                if pr > 0:
                    eq_r = alpha * E.top(max(0.04, pr)) + (1 - alpha) * eq
                    e = min(fpot, my_stack - add, opp_stack - oadd)
                    e = max(0, e)
                    ev_r = max(-add, (eq_r - delta) * (fpot + 2 * e) - add - e)
                    evv += pr * ev_r
                cands.append((evv, "raise", x))
        if not cands:
            return safe_default(legal)
        cands.sort(key=lambda t: -t[0])
        best = cands[0]
        if PARAMS.get("debug"):
            log("H%d %s %s eq=%.3f t50=%.3f t25=%.3f n=%d pot=%d tc=%d a=%.2f | %s" % (
                self.hand_id, street, " ".join(I2C[c] for c in (self.hole2 or self.hole3)), E.eq,
                E.top(0.5), E.top(0.25), E.n, pot, to_call, alpha,
                " ".join("%s%s:%.1f" % (c[1][0], c[2] or "", c[0]) for c in cands[:5])))
        # mild randomisation between near-equal options to stay unpredictable
        if len(cands) > 1 and cands[1][1] != best[1]:
            gap = best[0] - cands[1][0]
            if gap < 0.015 * pot and self.rng.random() < 0.25:
                best = cands[1]
        self.n_dec += 1
        return make_action(best[1], best[2], req)

    def act_preflop(self, req, legal, ev):
        m = self.model
        pot = req["pot"]
        to_call = req["to_call"]
        my_bet = req["my_bet"]
        opp_bet = req["opp_bet"]
        my_stack = req["my_stack"]
        opp_stack = req["opp_stack"]
        ip = self.seat == "button"
        idx = combo_index(*self.hole3)
        _, rem, alpha = self._pf_factor_fn(ev)
        def eqv(c):
            return alpha * pf_eq(idx, c) + (1 - alpha) * PF_COLS[0][idx]
        eq = eqv(rem)
        R = PARAMS["r_ip"] if ip else PARAMS["r_oop"]
        cands = []
        if to_call > 0:
            if "fold" in legal:
                cands.append((0.0, "fold", None))
            if "call" in legal:
                cands.append((eq * R * (pot + to_call) - to_call, "call", None))
        elif "check" in legal:
            cands.append((eq * R * pot, "check", None))
        if "raise" in legal and "min_raise" in req:
            mn, mx = req["min_raise"], req["max_raise"]
            if opp_bet <= 2:
                sizes = (5, 6, 8) if to_call > 0 else (6, 8)
            else:
                sizes = (int(opp_bet * 2.6), int(opp_bet * 3.2), mx)
            xs = set()
            for x in sizes:
                x = max(mn, min(mx, x))
                if x > 0.42 * (my_bet + my_stack):
                    x = mx
                xs.add(x)
            if m.opp_raised_street(ev, "preflop"):
                fold_base = m.pf_vs_3bet.p("fold")
                pr = m.pf_vs_3bet.p("raise")
            else:
                fold_base = m.pf_vs_raise.p("fold")
                pr = m.pf_vs_raise.p("raise")
            for x in sorted(xs):
                add = x - my_bet
                oadd = x - opp_bet
                if add > my_stack:
                    continue
                allin = x == mx
                frac = oadd / max(1.0, pot + to_call)
                pf = min(0.95, fold_base * (0.85 + 0.15 * min(frac, 3.0)))
                p_r = 0.0 if allin else pr
                pc = max(0.0, 1.0 - pf - p_r)
                eq_c = eqv(rem * (1.0 - pf))
                fpot = pot + add + oadd
                delta = 0.0 if fpot < 60 else 0.02 + 0.03 * min(1.0, fpot / 400.0)
                rr = R if fpot < 100 else 1.0
                ev_c = (eq_c - delta) * rr * fpot - add
                evv = pf * pot + pc * ev_c
                if p_r > 0:
                    eq_r = eqv(rem * max(p_r, 0.04))
                    e = min(int(2.5 * oadd) + 2, my_stack - add, opp_stack - oadd)
                    e = max(0, e)
                    ev_r = max(-add, (eq_r - 0.02) * rr * (fpot + 2 * e) - add - e)
                    evv += p_r * ev_r
                cands.append((evv, "raise", x))
        if not cands:
            return safe_default(legal)
        cands.sort(key=lambda t: -t[0])
        best = cands[0]
        if len(cands) > 1 and cands[1][1] != best[1] and best[0] - cands[1][0] < 0.04 * pot \
                and self.rng.random() < 0.2:
            best = cands[1]
        return make_action(best[1], best[2], req)

    def panic_action(self, req, legal, board):
        """Near-zero-cost decision used only when the time bank is almost empty."""
        h = self.hole2 or self.hole3
        bk = 0
        for c in board:
            bk += KEY[c]
        if self.hole2:
            sc = hscore(h[0], h[1], board, bk, len(board))
        else:
            sc, _ = best_keep(h, board[:3], KEY[board[0]] + KEY[board[1]] + KEY[board[2]])
        to_call = req.get("to_call", 0)
        pot = req.get("pot", 1)
        if sc > 0.8 and "raise" in legal:
            return make_action("raise", req.get("opp_bet", 0) + 0.7 * (pot + to_call), req)
        if to_call == 0:
            return safe_default(legal)
        if sc > 0.45 + 0.6 * to_call / max(1.0, pot + to_call) and "call" in legal:
            return {"type": "action", "action": "call"}
        return safe_default(legal)

    def discard(self, req):
        hole = [c2i(c) for c in req.get("hole_cards") or []]
        if len(hole) != 3:
            return None
        self.hole3 = hole
        self.hole2 = None
        board = [c2i(c) for c in req.get("board") or []][:3]
        if float(req.get("time_left_ms", 30000) or 0) < 600:
            _, d = best_keep(hole, board, KEY[board[0]] + KEY[board[1]] + KEY[board[2]])
            return I2C[hole[d]]
        ev = replay(self._last_history, self.seat) if self._last_history else []
        budget = self.budget(float(req.get("time_left_ms", 30000))) * 1.3
        E = self.monte_carlo(board, ev, budget)
        return I2C[hole[E.best_discard]]

    _last_history = None


def _gmean(kind, t1, t2):
    t = 0.0
    for i in range(41):
        t += _gfun(kind, i / 40.0, t1, t2)
    return t / 41.0


def _gfun(kind, p, t1, t2):
    if kind == "aggr":
        return _sig((p - t1) / SIG_T)
    if kind == "cont":
        return _sig((p - t1) / SIG_T) * (1.0 - 0.45 * _sig((p - t2) / SIG_T))
    # check: strongest hands would usually have bet
    return 1.0 - 0.55 * _sig((p - t1) / SIG_T)


class EqResult:
    """Weighted Monte Carlo samples with action-conditioned opponent range."""

    def __init__(self, scores, pw, res, steps, alpha):
        n = len(scores)
        self.alpha = alpha
        order = sorted(range(n), key=scores.__getitem__)
        tot = 0.0
        for w in pw:
            tot += w
        tot = tot or 1.0
        pct = [0.0] * n
        acc = 0.0
        for i in order:
            w = pw[i]
            pct[i] = (acc + 0.5 * w) / tot
            acc += w
        wt = list(pw)
        if steps:
            a = alpha
            # combined action factor tabulated on a 200-point percentile grid
            tab = [1.0] * 201
            for kind, t1, t2 in steps:
                mu = (1 - a) * _gmean(kind, t1, t2)
                for j in range(201):
                    g = _gfun(kind, j / 200.0, t1, t2)
                    tab[j] *= mu + a * (g if g > 0.02 else 0.02)
            for i in range(n):
                wt[i] *= tab[int(pct[i] * 200.0 + 0.5)]
        self.w = wt
        self.pct = pct
        self.order_desc = order[::-1]
        self.n = n
        self.res = None
        self.eq = 0.5
        self.emp_call = None
        if res is not None:
            self.set_res(res)

    def eq_with(self, res):
        s = 0.0
        t = 0.0
        w = self.w
        for i in range(self.n):
            s += w[i] * res[i]
            t += w[i]
        return s / t if t > 0 else 0.5

    def set_res(self, res):
        self.res = res
        self.eq = self.eq_with(res)

    def top(self, c):
        """Equity against the strongest c fraction (by weight) of the range."""
        if c >= 0.999 or self.res is None:
            return self.eq
        w = self.w
        res = self.res
        tot = 0.0
        for x in w:
            tot += x
        lim = c * tot
        s = 0.0
        t = 0.0
        for i in self.order_desc:
            s += w[i] * res[i]
            t += w[i]
            if t >= lim:
                break
        return s / t if t > 0 else self.eq


# ================================================================ protocol

def safe_default(legal):
    if "check" in legal:
        return {"type": "action", "action": "check"}
    if "fold" in legal:
        return {"type": "action", "action": "fold"}
    if "call" in legal:
        return {"type": "action", "action": "call"}
    return {"type": "action", "action": legal[0] if legal else "check"}


def make_action(action, amount, req):
    legal = req.get("legal_actions") or []
    if action not in legal:
        return safe_default(legal)
    if action == "raise":
        try:
            mn = int(req["min_raise"])
            mx = int(req["max_raise"])
            amt = int(amount)
            amt = max(mn, min(mx, amt))
            return {"type": "action", "action": "raise", "amount": amt}
        except Exception:
            if "call" in legal:
                return {"type": "action", "action": "call"}
            return safe_default(legal)
    return {"type": "action", "action": action}


def main():
    player = Player()
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        if not isinstance(msg, dict):
            continue
        t = msg.get("type")
        reply = None
        if t == "action_request":
            try:
                player._last_history = msg.get("history")
                reply = player.act(msg)
            except Exception as ex:
                log("act error", repr(ex))
                reply = safe_default(msg.get("legal_actions") or [])
        elif t == "discard_request":
            hole = msg.get("hole_cards") or []
            card = None
            try:
                card = player.discard(msg)
            except Exception as ex:
                log("discard error", repr(ex))
            if card not in hole:
                card = hole[0] if hole else "2c"
                try:
                    card = min(hole, key=lambda c: RANKS.index(c[0]))
                except Exception:
                    pass
            reply = {"type": "discard", "card": card}
        elif t == "match_end":
            break
        else:
            try:
                if t == "new_hand":
                    player.on_new_hand(msg)
                    player._last_history = None
                elif t == "discard_result":
                    player.on_discard_result(msg)
                elif t == "hand_result":
                    player.on_hand_result(msg)
                elif t == "init":
                    player.on_init(msg)
            except Exception as ex:
                log("msg error", t, repr(ex))
            continue
        try:
            out.write(json.dumps(reply) + "\n")
            out.flush()
        except Exception:
            break
    m = player.model
    log("hands %d net %d mc %.2fs | opp vpip %.2f pfr %.2f fold %.2f aggr %.2f "
        "a_pf %.2f a_post %.2f rat %.2f sd %d" % (
            m.hands, player.net, player.mc_time, m.vpip / max(1, m.hands), m.pfr / max(1, m.hands),
            m.post_face.p("fold"), m.post_open.p("raise"), m.alpha_pf(), m.alpha_post(),
            m.rational(), m.showdowns))


if __name__ == "__main__":
    main()
