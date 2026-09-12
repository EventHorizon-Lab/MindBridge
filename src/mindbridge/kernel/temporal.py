"""Temporal language and event-time arithmetic shared by retrieval and answering.

Nothing here reads storage or calls a model: a query's temporal window is a deterministic
function of its text and the reference clock, and the same parser serves the ranking signal,
the lexical deepening pass, and the recall planner's notes.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta, timezone

from mindbridge.exceptions import ValidationError
from mindbridge.types import SearchHit

_ISO_DATE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")


_TODAY_ISO_DATE = re.compile(r"\btoday\s+is\s+(\d{4}-\d{2}-\d{2})", re.IGNORECASE)


_MONTH_NAME = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)


_ORDINAL = r"(?:st|nd|rd|th)?"


_TODAY_NAMED_DATE = re.compile(
    r"\btoday\s+is\s+(?:the\s+)?(?:"
    rf"(?P<month>{_MONTH_NAME})\b\.?\s*,?\s*(?P<day>\d{{1,2}})(?!\d){_ORDINAL}"
    rf"|(?P<day_first>\d{{1,2}}){_ORDINAL}\s+(?:of\s+)?(?P<month_second>{_MONTH_NAME})\b\.?"
    r")(?:\s*,?\s*(?P<year>\d{4})\b)?",
    re.IGNORECASE,
)


_NAMED_MONTH_YEAR = re.compile(rf"\b({_MONTH_NAME})\s+((?:19|20|21)\d{{2}})\b", re.IGNORECASE)


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "一": 1,
    "两": 2,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
}


_ROLLING_SPAN_EN = re.compile(
    r"\b(?:(?:past|recent|previous|last)\s+(?P<count>\d{1,5}|[a-z]+)\s+"
    r"(?P<unit>day|week|month|year)s?"
    r"|past\s+(?P<bare_unit>day|week|month|year))\b"
)


_ROLLING_SPAN_CJK = re.compile(
    r"(?:过去|最近)\s*(?P<count>\d{1,5}|[一两二三四五六七八九十]{1,2})\s*(?:个)?(?P<unit>天|周|星期|月|年)"
)


_ROLLING_UNIT_MONTHS = {"month": 1, "year": 12, "月": 1, "年": 12}
_ROLLING_UNIT_DAYS = {"day": 1, "week": 7, "天": 1, "周": 7, "星期": 7}
_MAX_ROLLING_DAYS = 36_500


_STATED_DAY = re.compile(
    r"\b(?:"
    rf"(?P<month>{_MONTH_NAME})\b\.?\s+(?:(?P<day_start>\d{{1,2}}){_ORDINAL}\s*[-\u2013]\s*)?"
    rf"(?P<day>\d{{1,2}})(?!\d){_ORDINAL}"
    rf"|(?:(?P<day_first_start>\d{{1,2}}){_ORDINAL}\s*[-\u2013]\s*)?"
    rf"(?P<day_first>\d{{1,2}}){_ORDINAL}\s+(?:of\s+)?(?P<month_second>{_MONTH_NAME})\b\.?"
    r")(?:\s*,?\s*(?P<year>(?:19|20|21)\d{2})\b)?"
)


_DATE_JOINER = re.compile(r"^\s*(?:[-\u2013]|to|and|through|until|till)\s*$")
_ANY_DIGIT = re.compile(r"\d")
_MAX_STATED_SPAN = timedelta(days=366)


_CJK_YEAR_MONTH = re.compile(
    r"(?<![A-Za-z0-9_$-])((?:19|20|21)\d{2})年\s*(1[0-2]|0?[1-9])月"
    r"(?![A-Za-z0-9_$-])"
)


_CJK_CALENDAR_YEAR = re.compile(r"(?<![A-Za-z0-9_$-])((?:19|20|21)\d{2})年(?![A-Za-z0-9_$-])")


_CALENDAR_YEAR = re.compile(r"(?<![\w$-])((?:19|20|21)\d{2})(?![\w-])")


_MONTHS = {
    name: index
    for index, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
        start=1,
    )
}


def validated_occurred_at(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError("occurred_at must include a timezone")
    return value.astimezone(timezone.utc)


def search_occurrence_range(
    occurred_from: datetime | None,
    occurred_until: datetime | None,
) -> tuple[datetime | None, datetime | None]:
    bounds = []
    for value, name in (
        (occurred_from, "occurred_from"),
        (occurred_until, "occurred_until"),
    ):
        if value is not None and (
            not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None
        ):
            raise ValidationError(f"{name} must include a timezone")
        bounds.append(None if value is None else value.astimezone(timezone.utc))
    start, end = bounds
    if start is not None and end is not None and end <= start:
        raise ValidationError("occurred_until must be later than occurred_from")
    return start, end


def intersect_occurrence_range(
    temporal_range: tuple[datetime, datetime],
    occurred_from: datetime | None,
    occurred_until: datetime | None,
) -> tuple[datetime, datetime] | None:
    start = (
        max(temporal_range[0], occurred_from) if occurred_from is not None else temporal_range[0]
    )
    end = (
        min(temporal_range[1], occurred_until) if occurred_until is not None else temporal_range[1]
    )
    return None if end <= start else (start, end)


def occurrence_overlaps(
    occurred_at: datetime | None,
    occurred_end: datetime | None,
    occurred_from: datetime | None,
    occurred_until: datetime | None,
) -> bool:
    if occurred_from is None and occurred_until is None:
        return True
    if occurred_at is None:
        return False
    return (
        occurred_from is None
        or (
            occurred_end > occurred_from
            if occurred_end is not None
            else occurred_at >= occurred_from
        )
    ) and (occurred_until is None or occurred_at < occurred_until)


def validated_occurred_end(start: datetime | None, value: datetime | None) -> datetime | None:
    end = validated_occurred_at(value)
    if end is not None and (start is None or end <= start):
        raise ValidationError("occurred_end must be later than occurred_at")
    return end


def resolved_reference_at(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError("reference_at must include a timezone")
    return value


def overlaps_temporal_range(
    occurred_at: datetime | None,
    occurred_end: datetime | None,
    temporal_range: tuple[datetime, datetime],
) -> bool:
    if occurred_at is None:
        return False
    start, until = temporal_range
    event_until = occurred_end or occurred_at + timedelta(microseconds=1)
    return occurred_at < until and event_until > start


def query_temporal_range(text: str, reference_at: datetime) -> tuple[datetime, datetime] | None:
    try:
        return parse_temporal_range(text, reference_at)
    except (OverflowError, ValueError):
        raise ValidationError("temporal expression exceeds the supported date range") from None


def temporal_context(
    text: str,
    reference_at: datetime,
    *,
    infer_reference: bool,
) -> tuple[datetime, str]:
    match = _TODAY_ISO_DATE.search(text)
    try:
        if match is not None:
            anchored = date.fromisoformat(match.group(1))
        else:
            match = _TODAY_NAMED_DATE.search(text)
            if match is None:
                return reference_at, text
            month_name = match.group("month") or match.group("month_second")
            day = match.group("day") or match.group("day_first")
            month = _MONTHS[month_name[:3].casefold()]
            anchored = date(
                int(match.group("year")) if match.group("year") is not None else reference_at.year,
                month,
                int(day),
            )
    except ValueError:
        raise ValidationError("today reference date is invalid") from None
    temporal_text = f"{text[: match.start()]} {text[match.end() :]}".strip()
    if infer_reference:
        reference_at = _day_start(anchored, reference_at)
    return reference_at, temporal_text


def parse_temporal_range(  # noqa: C901 - ordered phrases are clearer as one parser
    text: str,
    reference_at: datetime,
) -> tuple[datetime, datetime] | None:
    if not text:
        return None
    dates = stated_iso_days(text)
    if dates:
        return _date_range(min(dates), max(dates), reference_at)

    normalized = text.casefold()
    months = stated_months(normalized)
    if months:
        first = min(date(year, month, 1) for year, month in months)
        last = max(date(year, month, 1) for year, month in months)
        start = _day_start(first, reference_at)
        return start, _shift_month(_day_start(last, reference_at), 1)

    relative_day = re.search(r"(?<!\d)(\d{1,5})\s*天前", normalized) or re.search(
        r"\b(\d{1,5})\s+days?\s+ago\b", normalized
    )
    if relative_day is not None:
        days = int(relative_day.group(1))
        if days <= _MAX_ROLLING_DAYS:
            target = reference_at.date() - timedelta(days=days)
            return _date_range(target, target, reference_at)

    rolling = rolling_span(normalized, reference_at)
    if rolling is not None:
        return rolling

    if _contains(normalized, "day before yesterday", "前天"):
        target = reference_at.date() - timedelta(days=2)
        return _date_range(target, target, reference_at)
    if _contains(normalized, "yesterday", "昨天"):
        target = reference_at.date() - timedelta(days=1)
        return _date_range(target, target, reference_at)
    if _contains(normalized, "day after tomorrow", "后天"):
        target = reference_at.date() + timedelta(days=2)
        return _date_range(target, target, reference_at)
    if _contains(normalized, "tomorrow", "明天"):
        target = reference_at.date() + timedelta(days=1)
        return _date_range(target, target, reference_at)
    if _contains(normalized, "today", "今天"):
        target = reference_at.date()
        return _date_range(target, target, reference_at)

    day_start = _day_start(reference_at.date(), reference_at)
    week_start = day_start - timedelta(days=reference_at.weekday())
    if _contains(normalized, "last week", "上周", "上星期"):
        return week_start - timedelta(days=7), week_start
    if _contains(normalized, "this week", "本周", "这周", "本星期"):
        return week_start, week_start + timedelta(days=7)
    if _contains(normalized, "next week", "下周", "下星期"):
        return week_start + timedelta(days=7), week_start + timedelta(days=14)
    if _contains(normalized, "past week", "过去一周", "最近一周"):
        return reference_at - timedelta(days=7), reference_at

    month_start = day_start.replace(day=1)
    if _contains(normalized, "last month", "上个月", "上月"):
        return _shift_month(month_start, -1), month_start
    if _contains(normalized, "this month", "这个月", "本月"):
        return month_start, _shift_month(month_start, 1)
    if _contains(normalized, "next month", "下个月", "下月"):
        return _shift_month(month_start, 1), _shift_month(month_start, 2)

    year_start = day_start.replace(month=1, day=1)
    if _contains(normalized, "last year", "去年"):
        return year_start.replace(year=year_start.year - 1), year_start
    if _contains(normalized, "this year", "今年"):
        return year_start, year_start.replace(year=year_start.year + 1)
    if _contains(normalized, "next year", "明年"):
        return (
            year_start.replace(year=year_start.year + 1),
            year_start.replace(year=year_start.year + 2),
        )

    years = [int(value) for value in _CALENDAR_YEAR.findall(normalized)]
    years.extend(int(value) for value in _CJK_CALENDAR_YEAR.findall(normalized))
    if years:
        return (
            _day_start(date(min(years), 1, 1), reference_at),
            _day_start(date(max(years) + 1, 1, 1), reference_at),
        )
    return None


def _contains(text: str, *phrases: str) -> bool:
    return any(phrase in text for phrase in phrases)


def stated_date_span(text: str, reference_at: datetime) -> tuple[datetime, datetime] | None:
    """Return the half-open span of absolute dates one record's text states."""
    if not text or _ANY_DIGIT.search(text) is None:
        return None
    normalized = text.casefold()
    days = [*stated_iso_days(text), *stated_named_days(normalized)]
    spans = [_date_range(day, day, reference_at) for day in days]
    dated_months = {(day.year, day.month) for day in days}
    for year, month in stated_months(normalized):
        if (year, month) in dated_months:
            continue
        first = _day_start(date(year, month, 1), reference_at)
        spans.append((first, _shift_month(first, 1)))
    if not spans:
        return None
    return min(start for start, _until in spans), max(until for _start, until in spans)


def rows_time_span(
    rows: Sequence[SearchHit], reference_at: datetime
) -> tuple[datetime, datetime] | None:
    """Return the half-open span covered by rows' stated dates or event times.

    A point event contributes through midnight after its event day. This preserves the existing
    single-point behavior and ensures the latest of several point anchors is inside the exclusive
    upper bound.
    """
    starts: list[datetime] = []
    ends: list[datetime] = []
    for hit in rows:
        stated = stated_date_span(hit.content, reference_at)
        if stated is not None and stated[1] - stated[0] <= _MAX_STATED_SPAN:
            starts.append(stated[0])
            ends.append(stated[1])
        elif hit.occurred_at is not None:
            starts.append(hit.occurred_at)
            if hit.occurred_end is not None:
                ends.append(hit.occurred_end)
            else:
                try:
                    ends.append(
                        _day_start(hit.occurred_at.date() + timedelta(days=1), hit.occurred_at)
                    )
                except OverflowError:
                    ends.append(hit.occurred_at)
    if not starts:
        return None
    start, until = min(starts), max(ends)
    if until <= start:
        try:
            until = _day_start(start.date() + timedelta(days=1), start)
        except OverflowError:
            return None
    return start, until


def stated_iso_days(text: str) -> list[date]:
    days = []
    for value in _ISO_DATE.findall(text):
        try:
            day = date.fromisoformat(value)
        except ValueError:
            continue
        if 1900 <= day.year <= 2199:
            days.append(day)
    return days


def stated_named_days(normalized: str) -> list[date]:
    """Return fully stated named days, borrowing a joined following date's year."""
    tokens: list[tuple[int, tuple[str, ...], int | None, int, int]] = []
    for match in _STATED_DAY.finditer(normalized):
        month_name = match.group("month") or match.group("month_second")
        day_texts = tuple(
            day_text
            for day_text in (
                match.group("day_start"),
                match.group("day_first_start"),
                match.group("day") or match.group("day_first"),
            )
            if day_text is not None
        )
        year_text = match.group("year")
        tokens.append(
            (
                _MONTHS[month_name[:3]],
                day_texts,
                None if year_text is None else int(year_text),
                match.start(),
                match.end(),
            )
        )
    days = []
    for index, (month, day_texts, year, _start, end) in enumerate(tokens):
        if year is None:
            following = tokens[index + 1] if index + 1 < len(tokens) else None
            if (
                following is None
                or following[2] is None
                or _DATE_JOINER.match(normalized[end : following[3]]) is None
            ):
                continue
            year = following[2]
        for day_text in day_texts:
            try:
                days.append(date(year, month, int(day_text)))
            except ValueError:
                continue
    return days


def stated_months(normalized: str) -> list[tuple[int, int]]:
    months = [
        (int(match.group(2)), _MONTHS[match.group(1)[:3]])
        for match in _NAMED_MONTH_YEAR.finditer(normalized)
    ]
    months.extend(
        (int(match.group(1)), int(match.group(2))) for match in _CJK_YEAR_MONTH.finditer(normalized)
    )
    return months


def rolling_span(normalized: str, reference_at: datetime) -> tuple[datetime, datetime] | None:
    """Resolve the first usable rolling calendar span in the text."""
    matches = (*_ROLLING_SPAN_EN.finditer(normalized), *_ROLLING_SPAN_CJK.finditer(normalized))
    for match in matches:
        groups = match.groupdict()
        if groups.get("bare_unit") is not None:
            count, unit = 1, groups["bare_unit"]
        else:
            raw_count = groups["count"]
            unit = groups["unit"]
            if raw_count.isdigit():
                count = int(raw_count)
            else:
                found = _NUMBER_WORDS.get(raw_count)
                if found is None:
                    continue
                count = found
        if unit in _ROLLING_UNIT_DAYS:
            days = count * _ROLLING_UNIT_DAYS[unit]
            if 0 < days <= _MAX_ROLLING_DAYS:
                return reference_at - timedelta(days=days), reference_at
            continue
        months = count * _ROLLING_UNIT_MONTHS[unit]
        if 0 < months <= _MAX_ROLLING_DAYS // 30:
            return _shift_month(reference_at, -months), reference_at
    return None


def _day_start(value: date, reference_at: datetime) -> datetime:
    return datetime.combine(value, time.min, tzinfo=reference_at.tzinfo)


def _date_range(first: date, last: date, reference_at: datetime) -> tuple[datetime, datetime]:
    return _day_start(first, reference_at), _day_start(last + timedelta(days=1), reference_at)


def _shift_month(value: datetime, months: int) -> datetime:
    """Shift by whole months, clamping the day to the target month's length."""
    year, month = divmod(value.year * 12 + value.month - 1 + months, 12)
    last_day = calendar.monthrange(year, month + 1)[1]
    return value.replace(year=year, month=month + 1, day=min(value.day, last_day))
