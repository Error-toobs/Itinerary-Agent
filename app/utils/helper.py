from datetime import date, datetime, timedelta
from typing import Union

def ensure_date(val):
    """
    Normalize input to a datetime.date object.

    Accepts:
    - date
    - datetime
    - string in 'YYYY-MM-DD' or 'YYYY/MM/DD'
    - None raises TypeError (explicit message)
    """
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, str):
        # Try common formats
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(val, fmt).date()
            except ValueError:
                pass
        raise TypeError(f"String date is not in a recognized format (expected YYYY-MM-DD or YYYY/MM/DD): {val!r}")
    if val is None:
        raise TypeError("Input to ensure_date must be a date, datetime, or a non-empty date string; got None")
    raise TypeError(f"Input to ensure_date must be a date, datetime, or a date-like string; got {type(val).__name__}: {val!r}")



def next_date(d: Union[date, datetime]) -> Union[date, datetime]:
    """
    Given a date or datetime object, return the next calendar date.
    - If input is date, return date.
    - If input is datetime, return datetime at same time on the next day.

    Examples:
        next_date(date(2024, 2, 28)) -> date(2024, 2, 29)  (leap year)
        next_date(datetime(2024, 2, 28, 13, 45)) -> datetime(2024, 2, 29, 13, 45)
    """
    d=ensure_date(d)  # Validate input type
    if isinstance(d, datetime):
        # Move to next day, keep same time
        return d + timedelta(days=1)
    elif isinstance(d, date):
        # Move to next day, keep date type
        return d + timedelta(days=1)
    else:
        raise TypeError("Input must be a datetime or date object")

# Optional: overload for purely date input that returns date (type hints already cover it)
def next_date_only(d: date) -> date:
    return d + timedelta(days=1)


def parse_date_from_text(text: str) -> date:
    """
    Parse a date mentioned in natural text like "24 feb 2025" and return a date object.

    Supported:
    - Day month year with month as full or short name (case-insensitive)
    - Delimiter can be spaces; accepts single spaces or multiple spaces

    Examples:
    - "24 feb 2025" -> date(2025, 2, 24)
    - "3 August 1999" -> date(1999, 8, 3)
    - "01 jan 2020" -> date(2020, 1, 1)

    Raises:
    - ValueError if the text cannot be parsed as a valid date
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string containing a date.")

    s = text.strip()

    # Accept ISO-style dates as well (e.g., 2025-02-24 or 2025/02/24)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass

    # Normalize multiple spaces to single space and lowercase for month names
    s_lower = s.lower()
    parts = [p for p in s_lower.split() if p]
    if len(parts) != 3:
        raise ValueError(f"Unable to parse date from: '{text}'")

    day_str, month_str, year_str = parts

    # Validate day
    try:
        day = int(day_str)
    except ValueError:
        raise ValueError(f"Invalid day in date: '{day_str}'")

    # Map month names (full and short) to month numbers
    MONTHS = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'sept': 9, 'september': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12,
    }

    month_num = None
    # month_str could be numeric as well (e.g., "2" for February)
    if month_str.isdigit():
        month_num = int(month_str)
    else:
        key = month_str[:3]  # use first 3 letters as a heuristic
        month_num = MONTHS.get(key)
    if not month_num or not (1 <= month_num <= 12):
        raise ValueError(f"Invalid month in date: '{month_str}'")

    # Year
    try:
        year = int(year_str)
    except ValueError:
        raise ValueError(f"Invalid year in date: '{year_str}'")

    # Construct and validate date (handles leap years, etc.)
    try:
        dt = date(year, month_num, day)
    except ValueError as e:
        raise ValueError(f"Invalid date: {e}")

    return dt


# # print(parse_date_from_text("24 feb 2025"))
# if __name__ == "__main__":
#     start_input = "2026-02-24"
#     end_input = "2026-03-02"

#     start_date = ensure_date(start_input)
#     end_date = ensure_date(end_input)

#     print("Start date:", start_date, type(start_date))
#     print("End date:", end_date, type(end_date))

