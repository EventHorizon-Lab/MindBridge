"""Pure temporal parsing and evidence spans, independent of memory and search adapters."""

from datetime import datetime, timedelta, timezone

import pytest

import mindbridge.kernel.temporal as temporal_module
from mindbridge import SearchHit


def test_rolling_spans_resolve_in_every_calendar_unit() -> None:
    """ "Past two years" is a window that ends at the clock, in either language and any unit."""
    reference = datetime(2026, 8, 31, 12, tzinfo=timezone.utc)
    parse = temporal_module.parse_temporal_range

    assert parse("conferences I attended in the past two years", reference) == (
        datetime(2024, 8, 31, 12, tzinfo=timezone.utc),
        reference,
    )
    assert parse("What did I spend in the last 3 months?", reference) == (
        datetime(2026, 5, 31, 12, tzinfo=timezone.utc),
        reference,
    )
    # A calendar shift, not a day count, and clamped when the target month is shorter.
    assert parse("trips in the past 6 months", reference) == (
        datetime(2026, 2, 28, 12, tzinfo=timezone.utc),
        reference,
    )
    assert parse("the recent two weeks", reference) == (
        reference - timedelta(days=14),
        reference,
    )
    assert parse("过去两年我参加了哪些会议", reference) == (
        datetime(2024, 8, 31, 12, tzinfo=timezone.utc),
        reference,
    )
    assert parse("最近三个月去过哪里", reference) == (
        datetime(2026, 5, 31, 12, tzinfo=timezone.utc),
        reference,
    )
    # A bare "past year" is one rolling year; a bare "last year" stays the calendar year.
    assert parse("travel in the past year", reference) == (
        datetime(2025, 8, 31, 12, tzinfo=timezone.utc),
        reference,
    )
    assert parse("What happened last year?", reference) == (
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    # "the last lecture" is an ordinal, not a span, and an unbounded count is refused.
    assert parse("the last lecture I attended", reference) is None
    assert parse("the past 999999 days", reference) is None
    assert parse("the past 500 years", reference) is None


@pytest.mark.parametrize(
    ("question", "expected"),
    (
        ("my recent day trips", None),
        ("conferences I attended in recent years", None),
        ("what did I do the previous year?", None),
        ("the last few years were busy", None),
        ("最近天气怎么样", None),
        ("最近天天加班吗", None),
        # A vague count earlier in the question does not hide the precise one after it.
        (
            "in the past few days, especially the last 3 days, what did I photograph?",
            (
                datetime(2026, 8, 28, 12, tzinfo=timezone.utc),
                datetime(2026, 8, 31, 12, tzinfo=timezone.utc),
            ),
        ),
    ),
)
def test_a_rolling_span_needs_its_count(
    question: str, expected: tuple[datetime, datetime] | None
) -> None:
    """ "Recent" and "previous" beside a unit noun are adjectives, not spans.

    Defaulting a missing count to one turned "my recent day trips" into a one-day window and
    "in recent years" into one rolling year, narrowing questions that used to run unbounded.
    """
    reference = datetime(2026, 8, 31, 12, tzinfo=timezone.utc)

    assert temporal_module.parse_temporal_range(question, reference) == expected


@pytest.mark.parametrize(
    "question",
    (
        # A year with no day is not a day: the old grammar read the 20th out of "2025".
        "Today is July 2025, what did I do?",
        # A month abbreviation that merely prefixes a word names no month.
        "Today is 2 decades since we moved, what happened?",
        "Today is 3 junior league games in, what happened?",
    ),
)
def test_text_that_only_looks_like_a_stated_today_leaves_the_clock_alone(question: str) -> None:
    wall_clock = datetime(2026, 9, 11, 8, tzinfo=timezone.utc)

    anchored, temporal_text = temporal_module.temporal_context(
        question, wall_clock, infer_reference=True
    )

    assert anchored == wall_clock
    assert temporal_text == question


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("Booking confirmed: Z Hotel Glasgow, June 23-24, 2022.", ("2022-06-23", "2022-06-25")),
        # The first half of a cross-month range borrows the year of the second.
        ("Stay: June 23 - July 2, 2022", ("2022-06-23", "2022-07-03")),
        ("Nights of 1-2 June 2022 at the inn", ("2022-06-01", "2022-06-03")),
        ("23 June to 25 June 2022", ("2022-06-23", "2022-06-26")),
        ("Checkout 2022-06-25; check-in 2022-06-23", ("2022-06-23", "2022-06-26")),
        ("photos from July 2024", ("2024-07-01", "2024-08-01")),
        ("2024年8月的照片", ("2024-08-01", "2024-09-01")),
        # A stated day already bounds its month; "October 2024" beside it adds nothing.
        ("checked in on 1 October 2024", ("2024-10-01", "2024-10-02")),
        # A year-less day with nothing dated joined to it is prose, not a date.
        ("you may 5 people at the table", None),
        # Bare years, relative phrases, sentinels and digit-free text state no span.
        ("ECCV 2024 in Milan", None),
        ("what happened 3 days ago", None),
        ("membership valid until 9999-12-31", None),
        ("a sheep on a hillside", None),
    ),
)
def test_stated_dates_are_read_from_record_text(
    text: str, expected: tuple[str, str] | None
) -> None:
    reference = datetime(2026, 9, 12, tzinfo=timezone.utc)

    span = temporal_module.stated_date_span(text, reference)

    assert span == (
        None
        if expected is None
        else tuple(datetime.fromisoformat(value).replace(tzinfo=timezone.utc) for value in expected)
    )


def _row(
    content: str, occurred_at: datetime | None, occurred_end: datetime | None = None
) -> SearchHit:
    return SearchHit(
        id=content,
        content=content,
        score=0.5,
        created_at=datetime(2026, 9, 12, tzinfo=timezone.utc),
        occurred_at=occurred_at,
        occurred_end=occurred_end,
    )


def test_a_row_span_prefers_stated_dates_and_falls_back_to_event_time() -> None:
    """The stay an email states bounds the read; the day the email arrived does not."""
    reference = datetime(2026, 9, 12, tzinfo=timezone.utc)
    utc = timezone.utc
    mail = _row("Booking: Z Hotel, June 23-24, 2022.", datetime(2022, 4, 1, 9, tzinfo=utc))
    photo = _row("a green pallet", datetime(2022, 6, 23, 14, tzinfo=utc))

    assert temporal_module.rows_time_span((mail, photo), reference) == (
        datetime(2022, 6, 23, tzinfo=utc),
        datetime(2022, 6, 25, tzinfo=utc),
    )
    # Dates about several things -- the stay beside a terms revision years earlier -- are not a
    # span of one event, so that row is bounded by its event time instead.
    newsletter = _row(
        "Booking: June 23-24, 2022. Terms updated January 2019.",
        datetime(2022, 6, 12, 22, tzinfo=utc),
    )
    assert temporal_module.rows_time_span((newsletter,), reference) == (
        datetime(2022, 6, 12, 22, tzinfo=utc),
        datetime(2022, 6, 13, tzinfo=utc),
    )
    # One instant widens to the end of its day; no time at all is no span.
    assert temporal_module.rows_time_span((photo,), reference) == (
        datetime(2022, 6, 23, 14, tzinfo=utc),
        datetime(2022, 6, 24, tzinfo=utc),
    )
    morning = _row("morning", datetime(2022, 6, 22, 9, tzinfo=utc))
    evening = _row("evening", datetime(2022, 6, 23, 20, tzinfo=utc))
    assert temporal_module.rows_time_span((morning, evening), reference) == (
        datetime(2022, 6, 22, 9, tzinfo=utc),
        datetime(2022, 6, 24, tzinfo=utc),
    )
    assert temporal_module.rows_time_span((_row("undated", None),), reference) is None
    # A sentinel event time cannot overflow the widening into an exception.
    assert (
        temporal_module.rows_time_span(
            (_row("forever", datetime(9999, 12, 31, tzinfo=utc)),), reference
        )
        is None
    )


@pytest.mark.parametrize(
    "spelling",
    ("July 1, 2025", "July, 1 2025", "July 1st 2025", "1 July 2025", "the 1st of July, 2025"),
)
def test_a_stated_today_anchors_the_clock_in_every_common_spelling(spelling: str) -> None:
    """The question's own clock wins however its date is punctuated.

    "Today is July, 1 2025" was not recognized at all, so the question was read against the real
    wall clock and its "past two years" landed on the wrong two years.
    """
    wall_clock = datetime(2026, 9, 11, 8, tzinfo=timezone.utc)
    question = f"Today is {spelling}, which conferences did I attend in the past two years?"

    anchored, temporal_text = temporal_module.temporal_context(
        question, wall_clock, infer_reference=True
    )

    assert anchored == datetime(2025, 7, 1, tzinfo=timezone.utc)
    assert "today is" not in temporal_text.casefold()
    assert temporal_module.query_temporal_range(temporal_text, anchored) == (
        datetime(2023, 7, 1, tzinfo=timezone.utc),
        anchored,
    )
