"""Temporal language and event-time arithmetic shared by retrieval and answering.

Nothing here reads storage or calls a model: a query's temporal window is a deterministic
function of its text and the reference clock, and the same parser serves the ranking signal,
the lexical deepening pass, and the recall planner's notes.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone

from mindbridge.exceptions import ValidationError

_ISO_DATE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")


_TODAY_ISO_DATE = re.compile(r"\btoday\s+is\s+(\d{4}-\d{2}-\d{2})", re.IGNORECASE)


_MONTH_NAME = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)


_TODAY_NAMED_DATE = re.compile(
    rf"\btoday\s+is\s+({_MONTH_NAME})"
    r"\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{4}))?",
    re.IGNORECASE,
)


_NAMED_MONTH_YEAR = re.compile(rf"\b({_MONTH_NAME})\s+((?:19|20|21)\d{{2}})\b", re.IGNORECASE)


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
            month = _MONTHS[match.group(1)[:3].casefold()]
            anchored = date(
                int(match.group(3)) if match.group(3) is not None else reference_at.year,
                month,
                int(match.group(2)),
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
    dates = []
    for value in _ISO_DATE.findall(text):
        try:
            dates.append(date.fromisoformat(value))
        except ValueError:
            continue
    if dates:
        return _date_range(min(dates), max(dates), reference_at)

    normalized = text.casefold()
    months = [
        (int(match.group(2)), _MONTHS[match.group(1)[:3]])
        for match in _NAMED_MONTH_YEAR.finditer(normalized)
    ]
    months.extend(
        (int(match.group(1)), int(match.group(2))) for match in _CJK_YEAR_MONTH.finditer(normalized)
    )
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
        if days <= 36_500:
            target = reference_at.date() - timedelta(days=days)
            return _date_range(target, target, reference_at)

    rolling = re.search(r"(?:过去|最近)\s*(\d{1,5})\s*天", normalized) or re.search(
        r"\b(?:past|last)\s+(\d{1,5})\s+days?\b", normalized
    )
    if rolling is not None:
        days = int(rolling.group(1))
        if 0 < days <= 36_500:
            return reference_at - timedelta(days=days), reference_at

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


def _day_start(value: date, reference_at: datetime) -> datetime:
    return datetime.combine(value, time.min, tzinfo=reference_at.tzinfo)


def _date_range(first: date, last: date, reference_at: datetime) -> tuple[datetime, datetime]:
    return _day_start(first, reference_at), _day_start(last + timedelta(days=1), reference_at)


def _shift_month(value: datetime, months: int) -> datetime:
    year, month = divmod(value.year * 12 + value.month - 1 + months, 12)
    return value.replace(year=year, month=month + 1)
