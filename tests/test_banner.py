"""Banner tests.

The requirement that matters is negative: at no width, in no encoding, may the banner
emit a line the terminal cannot hold or print. Everything else about it is decoration.
"""

from __future__ import annotations

import io

import pytest

from setpoint import banner
from setpoint.render import Style, enable_unicode_output, encodable

PLAIN = Style(False)
PAINTED = Style(True)


class TestFit:
    @pytest.mark.parametrize(
        "width", [200, 120, 100, 80, 66, 65, 50, 40, 32, 31, 30, 20, 14, 13, 8]
    )
    def test_no_line_is_wider_than_the_terminal(self, width):
        for line in banner.lines(width, unicode_ok=True):
            assert len(line) <= width

    @pytest.mark.parametrize("width", [200, 66, 40, 20, 8])
    def test_the_ascii_form_fits_too(self, width):
        for line in banner.lines(width, unicode_ok=False):
            assert len(line) <= width

    def test_a_width_of_one_still_produces_something_printable(self):
        # Not a real terminal, but the arithmetic must not produce a negative slice.
        assert all(len(line) <= 1 for line in banner.lines(1, unicode_ok=True))

    @pytest.mark.parametrize("width", [63, 40, 30, 29, 28, 13, 12, 11, 9, 8])
    def test_no_form_is_ever_cut_through_a_letter(self, width):
        # The failure this prevents: a terminal one column too narrow used to print
        # half a letterform rather than stepping down to the next size.
        body = [
            line for line in banner.lines(width, tagline=False, unicode_ok=True) if line.strip()
        ]
        drawn = {
            "wordmark": banner.WORDMARK_WIDTH,
            "mark": banner.MARK_WIDTH,
            "name": len(banner.NAME),
        }
        chosen = banner.tier(width)
        assert max(len(line) for line in body) == min(drawn[chosen], width)

    def test_a_terminal_exactly_as_wide_as_the_mark_still_shows_it(self):
        # The mark is centred, so it may use the full width. Dropping to a smaller form
        # one column early is what made a wide terminal show the compact one.
        assert banner.tier(banner.WORDMARK_WIDTH) == "wordmark"
        assert banner.tier(banner.WORDMARK_WIDTH - 1) != "wordmark"


class TestTiers:
    def test_a_wide_terminal_gets_the_wordmark(self):
        assert banner.tier(100) == "wordmark"

    def test_a_middling_terminal_gets_the_mark(self):
        # There is no half-height wordmark: it read as a cut-off one every time it was
        # shown, so below the full mark the small lockup takes over.
        assert banner.tier(50) == "mark"

    def test_no_form_is_a_shortened_wordmark(self):
        assert "compact" not in banner.FORMS

    def test_a_narrow_terminal_gets_the_mark(self):
        assert banner.tier(20) == "mark"

    def test_a_terminal_too_narrow_for_any_drawing_gets_the_name(self):
        assert banner.tier(10) == "name"

    def test_the_tier_steps_down_one_column_at_a_time(self):
        assert banner.tier(banner.WORDMARK_WIDTH) == "wordmark"
        assert banner.tier(banner.WORDMARK_WIDTH - 1) == "mark"
        assert banner.tier(banner.MARK_WIDTH) == "mark"
        assert banner.tier(banner.MARK_WIDTH - 1) == "name"

    def test_a_caller_can_refuse_the_largest_form(self):
        assert banner.tier(100, limit="mark") == "mark"

    def test_the_rules_frame_the_letters(self):
        # They come from the design the mark was drawn for; without them it reads as a
        # fragment rather than a lockup.
        body = banner.lines(100, tagline=False, unicode_ok=True)
        assert set(body[0]) == {banner.RULE}
        assert set(body[-1]) == {banner.RULE}
        assert len(body[0]) == banner.WORDMARK_WIDTH

    def test_the_rules_match_the_form_they_frame(self):
        body = banner.lines(40, tagline=False, unicode_ok=True)
        assert len(body[0]) == banner.MARK_WIDTH

    def test_the_tagline_shortens_before_it_would_wrap(self):
        body = banner.lines(40, unicode_ok=True)
        assert banner.TAGLINE_SHORT in body
        assert banner.TAGLINE not in body


class TestEncoding:
    def test_an_encoding_without_box_drawing_falls_back_to_ascii(self):
        # Measured on the development machine: a Windows console on code page 857
        # reports cp1254, which cannot encode the wordmark. Printing it raised and took
        # the whole command down.
        assert banner.tier(100, unicode_ok=False) == "ascii"
        body = banner.lines(100, unicode_ok=False)
        assert all(line.isascii() for line in body)

    def test_the_check_is_what_the_stream_can_actually_encode(self):
        latin = io.TextIOWrapper(io.BytesIO(), encoding="cp1254")
        wide = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        assert not encodable(banner.WORDMARK[0], latin)
        assert encodable(banner.WORDMARK[0], wide)

    def test_a_stream_that_cannot_be_reconfigured_is_left_alone(self):
        assert enable_unicode_output(io.StringIO()) is False

    def test_a_utf8_stream_needs_no_reconfiguring(self):
        assert enable_unicode_output(io.TextIOWrapper(io.BytesIO(), encoding="utf-8")) is True


class TestPaint:
    def test_without_colour_the_text_is_untouched(self):
        assert banner.render(PLAIN, 100) == banner.lines(100)

    def test_with_colour_the_visible_text_is_unchanged(self):
        import re

        for painted, plain in zip(
            banner.render(PAINTED, 100, unicode_ok=True),
            banner.lines(100, unicode_ok=True),
            strict=True,
        ):
            assert re.sub(r"\x1b\[[0-9;]*m", "", painted) == plain

    def test_every_colour_is_closed(self):
        # Not "ends with a reset": the graphic is padded, so a line can end in spaces.
        # What must hold is that no colour leaks into whatever is printed next.
        for line in banner.render(PAINTED, 100, unicode_ok=True):
            resets = line.count("\x1b[0m")
            assert line.count("\x1b[") - resets == resets

    def test_the_face_and_the_bevel_are_different_colours(self):
        # Both are stated. Leaving the bevel to the terminal made it come out too close
        # to the accent, and the strokes inside the letters stopped reading.
        painted = banner.paint("███╗", PAINTED)
        assert f"[38;5;{banner.FACE_COLOUR}m" in painted
        assert f"[38;5;{banner.BEVEL_COLOUR}m" in painted
        assert banner.FACE_COLOUR != banner.BEVEL_COLOUR

    def test_the_bevel_is_white(self):
        # White against a coloured face is the pair that survives both themes.
        assert banner.HEX[banner.BEVEL_COLOUR] == "#ffffff"

    def test_the_face_is_light_enough_to_read_as_a_solid_block(self):
        # Measured, not chosen by eye: blue carries little luminance, so a navy face
        # vanishes on a dark terminal and the mark reads as a hollow outline.
        hexes = banner.HEX[banner.FACE_COLOUR]
        rgb = [int(hexes[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        chan = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        luminance = 0.2126 * chan[0] + 0.7152 * chan[1] + 0.0722 * chan[2]
        assert (luminance + 0.05) / 0.05 >= 3.0

    def test_both_colours_have_a_hex_form_for_the_panel(self):
        assert set(banner.HEX) == {banner.FACE_COLOUR, banner.BEVEL_COLOUR}


class TestCentring:
    def test_a_printed_banner_is_not_centred(self):
        # Padding makes each row longer than the mark, so narrowing the window wraps a
        # block that would otherwise still have fitted. Only a redrawing surface centres.
        assert not banner.render(PLAIN, 100)[0].startswith(" ")
        assert banner.render(PLAIN, 100, centred=True)[0].startswith(" ")

    def test_the_block_moves_as_one_piece(self):
        # Centring each row on its own shears the letterforms, because the rows of a
        # block letter are not all the same length.
        body = banner.centre(banner.lines(100, tagline=False, unicode_ok=True), 100)
        leading = {len(line) - len(line.lstrip(" ")) for line in body}
        assert len(leading) == 1

    def test_it_is_centred_on_the_terminal(self):
        body = banner.centre(banner.lines(100, tagline=False, unicode_ok=True), 100)
        pad = len(body[0]) - len(body[0].lstrip(" "))
        assert pad == (100 - banner.WORDMARK_WIDTH) // 2

    def test_a_terminal_no_wider_than_the_mark_gets_no_padding(self):
        body = banner.centre(banner.lines(64, tagline=False, unicode_ok=True), 64)
        assert not body[0].startswith(" ")


class TestMarkup:
    def test_the_panel_form_closes_every_tag(self):
        # An unclosed tag is not a cosmetic problem: the markup parser refuses the whole
        # string and the panel fails to mount.
        text = banner.markup(100, limit="mark")
        closing = text.count("[/]")
        opening = text.count("[") - closing
        assert opening == closing

    def test_it_honours_the_limit(self):
        assert banner.markup(100, limit="mark").count("\n") < banner.markup(100).count("\n")
