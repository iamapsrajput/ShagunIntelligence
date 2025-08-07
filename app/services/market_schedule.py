"""
Market Schedule and Trading Hours Management
Handles Indian stock market trading hours, holidays, and special sessions
"""

from datetime import datetime, time, timedelta
from typing import Optional

import pytz
import requests
from loguru import logger

from app.core.config import get_settings


class MarketScheduleManager:
    """Manages market schedule, trading hours, and holiday detection"""

    def __init__(self):
        self.settings = get_settings()
        self.timezone = pytz.timezone(self.settings.TRADING_TIMEZONE)

        # Standard trading hours
        self.market_open_time = time.fromisoformat(self.settings.TRADING_START_TIME)
        self.market_close_time = time.fromisoformat(self.settings.TRADING_END_TIME)

        # Pre-market and after-hours
        self.pre_market_start = time(9, 0)  # 9:00 AM
        self.pre_market_end = time(9, 15)  # 9:15 AM
        self.after_hours_start = time(15, 30)  # 3:30 PM
        self.after_hours_end = time(16, 0)  # 4:00 PM

        # Enhanced cache for holidays with metadata
        self._holidays_cache: dict[int, dict[str, any]] = {}
        self._cache_expiry: datetime | None = None
        self._last_api_attempt: datetime | None = None
        self._api_retry_delay = timedelta(hours=1)  # Retry API after 1 hour if failed

    def get_current_market_time(self) -> datetime:
        """Get current time in market timezone"""
        return datetime.now(self.timezone)

    def is_market_open(self, check_time: datetime | None = None) -> bool:
        """Check if market is currently open"""
        if check_time is None:
            check_time = self.get_current_market_time()

        # Convert to market timezone if needed
        if check_time.tzinfo is None:
            check_time = self.timezone.localize(check_time)
        elif check_time.tzinfo != self.timezone:
            check_time = check_time.astimezone(self.timezone)

        # Check if it's a weekend
        if check_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check if it's a holiday
        if self.is_market_holiday(check_time.date()):
            return False

        # Check trading hours
        current_time = check_time.time()
        return self.market_open_time <= current_time <= self.market_close_time

    def is_pre_market(self, check_time: datetime | None = None) -> bool:
        """Check if it's pre-market hours"""
        if check_time is None:
            check_time = self.get_current_market_time()

        if check_time.tzinfo is None:
            check_time = self.timezone.localize(check_time)
        elif check_time.tzinfo != self.timezone:
            check_time = check_time.astimezone(self.timezone)

        if check_time.weekday() >= 5 or self.is_market_holiday(check_time.date()):
            return False

        current_time = check_time.time()
        return self.pre_market_start <= current_time < self.pre_market_end

    def is_after_hours(self, check_time: datetime | None = None) -> bool:
        """Check if it's after-hours trading"""
        if check_time is None:
            check_time = self.get_current_market_time()

        if check_time.tzinfo is None:
            check_time = self.timezone.localize(check_time)
        elif check_time.tzinfo != self.timezone:
            check_time = check_time.astimezone(self.timezone)

        if check_time.weekday() >= 5 or self.is_market_holiday(check_time.date()):
            return False

        current_time = check_time.time()
        return self.after_hours_start <= current_time <= self.after_hours_end

    def get_market_status(
        self, check_time: datetime | None = None
    ) -> dict[str, any]:
        """Get comprehensive market status"""
        if check_time is None:
            check_time = self.get_current_market_time()

        is_open = self.is_market_open(check_time)
        is_pre = self.is_pre_market(check_time)
        is_after = self.is_after_hours(check_time)
        is_holiday = self.is_market_holiday(check_time.date())
        is_weekend = check_time.weekday() >= 5

        # Determine status
        if is_open:
            status = "OPEN"
            message = "Market is open for trading"
        elif is_pre:
            status = "PRE_MARKET"
            message = "Pre-market session"
        elif is_after:
            status = "AFTER_HOURS"
            message = "After-hours trading"
        elif is_holiday:
            status = "HOLIDAY"
            message = "Market closed - Holiday"
        elif is_weekend:
            status = "WEEKEND"
            message = "Market closed - Weekend"
        else:
            status = "CLOSED"
            message = "Market closed"

        # Get next market open time
        next_open = self.get_next_market_open(check_time)

        return {
            "status": status,
            "message": message,
            "is_open": is_open,
            "is_pre_market": is_pre,
            "is_after_hours": is_after,
            "is_holiday": is_holiday,
            "is_weekend": is_weekend,
            "current_time": check_time.isoformat(),
            "market_open_time": self.market_open_time.isoformat(),
            "market_close_time": self.market_close_time.isoformat(),
            "next_market_open": next_open.isoformat() if next_open else None,
            "timezone": str(self.timezone),
        }

    def get_next_market_open(
        self, from_time: datetime | None = None
    ) -> datetime | None:
        """Get the next market open time"""
        if from_time is None:
            from_time = self.get_current_market_time()

        # Start checking from the next day if market is already closed today
        check_date = from_time.date()
        if from_time.time() > self.market_close_time:
            check_date += timedelta(days=1)

        # Look for next trading day (max 10 days ahead)
        for i in range(10):
            check_datetime = datetime.combine(check_date, self.market_open_time)
            check_datetime = self.timezone.localize(check_datetime)

            # Skip weekends and holidays
            if check_datetime.weekday() < 5 and not self.is_market_holiday(check_date):
                return check_datetime

            check_date += timedelta(days=1)

        return None

    def get_next_market_close(
        self, from_time: datetime | None = None
    ) -> datetime | None:
        """Get the next market close time"""
        if from_time is None:
            from_time = self.get_current_market_time()

        # If market is open today, return today's close
        if self.is_market_open(from_time):
            close_datetime = datetime.combine(from_time.date(), self.market_close_time)
            return self.timezone.localize(close_datetime)

        # Otherwise, find next trading day's close
        next_open = self.get_next_market_open(from_time)
        if next_open:
            close_datetime = datetime.combine(next_open.date(), self.market_close_time)
            return self.timezone.localize(close_datetime)

        return None

    def is_market_holiday(self, check_date: datetime.date) -> bool:
        """Check if a given date is a market holiday"""
        year = check_date.year

        # Get holidays for the year
        holidays = self.get_market_holidays(year)

        return check_date in [h.date() for h in holidays]

    def get_market_holidays(self, year: int) -> list[datetime]:
        """Get list of market holidays for a given year with enhanced caching"""
        now = datetime.now()

        # Check cache first
        if year in self._holidays_cache:
            cache_entry = self._holidays_cache[year]
            if (
                isinstance(cache_entry, dict)
                and cache_entry.get("expiry")
                and now < cache_entry["expiry"]
            ):
                logger.debug(
                    f"Using cached holidays for {year} (source: {cache_entry.get('source', 'unknown')})"
                )
                return cache_entry["holidays"]
            elif (
                isinstance(cache_entry, list)
                and self._cache_expiry
                and now < self._cache_expiry
            ):
                # Handle old cache format
                logger.debug(f"Using cached holidays for {year} (legacy cache)")
                return cache_entry

        # Check if we should retry API (avoid hammering failed APIs)
        should_try_api = (
            self._last_api_attempt is None
            or now - self._last_api_attempt > self._api_retry_delay
        )

        holidays = []
        source = "predefined"

        if should_try_api:
            # Try to fetch from APIs
            self._last_api_attempt = now
            holidays = self._fetch_holidays_from_api(year)
            if holidays:
                source = "api"
                logger.info(
                    f"Successfully fetched {len(holidays)} holidays from API for {year}"
                )
            else:
                logger.warning(
                    f"API fetch failed for {year}, using predefined holidays"
                )

        if not holidays:
            # Fallback to predefined holidays
            holidays = self._get_predefined_holidays(year)
            source = "predefined"

        # Enhanced cache with metadata
        cache_duration = timedelta(
            days=7 if source == "api" else 30
        )  # Shorter cache for API data
        self._holidays_cache[year] = {
            "holidays": holidays,
            "source": source,
            "fetched_at": now,
            "expiry": now + cache_duration,
        }
        self._cache_expiry = now + cache_duration  # Maintain backward compatibility

        logger.info(f"Cached {len(holidays)} holidays for {year} from {source} source")
        return holidays

    def _fetch_holidays_from_api(self, year: int) -> list[datetime]:
        """Try to fetch holidays from multiple API sources with robust error handling"""
        holidays = []

        # Try multiple API sources in order of preference
        api_sources = [
            self._fetch_from_nse_api,
            self._fetch_from_bse_api,
            self._fetch_from_alternative_api,
        ]

        for api_source in api_sources:
            try:
                source_holidays = api_source(year)
                if source_holidays:
                    logger.info(
                        f"Successfully fetched {len(source_holidays)} holidays from {api_source.__name__}"
                    )
                    return source_holidays
            except Exception as e:
                logger.warning(
                    f"Failed to fetch holidays from {api_source.__name__}: {e}"
                )
                continue

        logger.warning(
            f"All API sources failed for year {year}, using predefined holidays"
        )
        return []

    def _fetch_from_nse_api(self, year: int) -> list[datetime]:
        """Fetch holidays from NSE API with improved error handling"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        # Try multiple NSE endpoints
        nse_urls = [
            f"https://www.nseindia.com/api/holiday-master?type=trading&year={year}",
            f"https://www.nseindia.com/api/holiday-master?type=clearing&year={year}",
            "https://www.nseindia.com/api/holiday-master",
        ]

        for url in nse_urls:
            try:
                response = requests.get(url, headers=headers, timeout=15, verify=True)
                if response.status_code == 200:
                    data = response.json()
                    holidays = []

                    # Try different data structures
                    holiday_lists = [
                        data.get("CBM", []),
                        data.get("CM", []),
                        data.get("FO", []),
                        data.get("CD", []),
                    ]

                    for holiday_list in holiday_lists:
                        if holiday_list:
                            for holiday in holiday_list:
                                try:
                                    # Try different date formats
                                    date_formats = [
                                        "%d-%b-%Y",
                                        "%Y-%m-%d",
                                        "%d/%m/%Y",
                                        "%d-%m-%Y",
                                    ]
                                    date_fields = [
                                        "tradingDate",
                                        "date",
                                        "holidayDate",
                                        "Date",
                                    ]

                                    for date_field in date_fields:
                                        if date_field in holiday:
                                            for date_format in date_formats:
                                                try:
                                                    holiday_date = datetime.strptime(
                                                        holiday[date_field], date_format
                                                    )
                                                    if holiday_date.year == year:
                                                        holidays.append(holiday_date)
                                                    break
                                                except (ValueError, TypeError):
                                                    continue
                                            break
                                except (KeyError, ValueError, TypeError):
                                    continue

                            if holidays:
                                return sorted(
                                    list(set(holidays))
                                )  # Remove duplicates and sort

            except requests.exceptions.RequestException as e:
                logger.debug(f"NSE API request failed for {url}: {e}")
                continue

        return []

    def _fetch_from_bse_api(self, year: int) -> list[datetime]:
        """Fetch holidays from BSE API"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        }

        try:
            # BSE holiday calendar endpoint
            url = f"https://api.bseindia.com/BseIndiaAPI/api/ListHolidays/w?segment=Equity&year={year}"

            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                holidays = []

                for holiday in data.get("Table", []):
                    try:
                        # BSE typically uses different date format
                        holiday_date = datetime.strptime(
                            holiday.get("Date", ""), "%Y-%m-%dT%H:%M:%S"
                        )
                        holidays.append(holiday_date)
                    except (KeyError, ValueError, TypeError):
                        continue

                return sorted(holidays)

        except Exception as e:
            logger.debug(f"BSE API request failed: {e}")

        return []

    def _fetch_from_alternative_api(self, year: int) -> list[datetime]:
        """Fetch holidays from alternative API sources"""
        # Alternative sources for Indian market holidays
        alternative_sources = [
            f"https://date.nager.at/api/v3/PublicHolidays/{year}/IN",  # Public holidays API
            f"https://calendarific.com/api/v2/holidays?api_key=your_key&country=IN&year={year}",  # Calendarific
        ]

        for url in alternative_sources:
            try:
                if "calendarific" in url and "your_key" in url:
                    continue  # Skip if no API key configured

                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    holidays = []

                    # Handle different response formats
                    if isinstance(data, list):  # Nager.at format
                        for holiday in data:
                            try:
                                holiday_date = datetime.strptime(
                                    holiday.get("date", ""), "%Y-%m-%d"
                                )
                                # Filter for market-relevant holidays
                                if self._is_market_relevant_holiday(
                                    holiday.get("name", "")
                                ):
                                    holidays.append(holiday_date)
                            except (KeyError, ValueError):
                                continue

                    elif "response" in data:  # Calendarific format
                        for holiday in data.get("response", {}).get("holidays", []):
                            try:
                                holiday_date = datetime.strptime(
                                    holiday.get("date", {}).get("iso", ""), "%Y-%m-%d"
                                )
                                if self._is_market_relevant_holiday(
                                    holiday.get("name", "")
                                ):
                                    holidays.append(holiday_date)
                            except (KeyError, ValueError):
                                continue

                    if holidays:
                        return sorted(holidays)

            except Exception as e:
                logger.debug(f"Alternative API request failed for {url}: {e}")
                continue

        return []

    def _is_market_relevant_holiday(self, holiday_name: str) -> bool:
        """Check if a holiday is relevant for stock market closure"""
        market_relevant_keywords = [
            "republic",
            "independence",
            "gandhi",
            "diwali",
            "holi",
            "dussehra",
            "eid",
            "christmas",
            "good friday",
            "janmashtami",
            "ram navami",
            "mahavir jayanti",
            "buddha purnima",
            "guru nanak",
            "karva chauth",
        ]

        holiday_lower = holiday_name.lower()
        return any(keyword in holiday_lower for keyword in market_relevant_keywords)

    def _get_predefined_holidays(self, year: int) -> list[datetime]:
        """Get comprehensive predefined holidays for the year with NSE/BSE official holidays"""

        # NSE/BSE Official Trading Holidays (Updated for 2024-2026)
        holidays_data = {
            2024: [
                (datetime(2024, 1, 26), "Republic Day"),
                (datetime(2024, 3, 8), "Holi"),
                (datetime(2024, 3, 29), "Good Friday"),
                (datetime(2024, 4, 11), "Id-Ul-Fitr (Ramzan Id)"),
                (datetime(2024, 4, 17), "Ram Navami"),
                (datetime(2024, 5, 1), "Maharashtra Day"),
                (datetime(2024, 6, 17), "Bakri Id"),
                (datetime(2024, 8, 15), "Independence Day"),
                (datetime(2024, 8, 26), "Janmashtami"),
                (datetime(2024, 10, 2), "Gandhi Jayanti"),
                (datetime(2024, 11, 1), "Diwali Laxmi Pujan"),
                (datetime(2024, 11, 15), "Guru Nanak Jayanti"),
                (datetime(2024, 12, 25), "Christmas"),
            ],
            2025: [
                (datetime(2025, 1, 26), "Republic Day"),
                (datetime(2025, 2, 26), "Holi"),
                (datetime(2025, 3, 31), "Id-Ul-Fitr (Ramzan Id)"),
                (datetime(2025, 4, 18), "Good Friday"),
                (datetime(2025, 5, 1), "Maharashtra Day"),
                (datetime(2025, 6, 6), "Bakri Id"),
                (datetime(2025, 8, 15), "Independence Day"),
                (datetime(2025, 8, 16), "Janmashtami"),
                (datetime(2025, 10, 2), "Gandhi Jayanti"),
                (datetime(2025, 10, 20), "Diwali Laxmi Pujan"),
                (datetime(2025, 11, 5), "Guru Nanak Jayanti"),
                (datetime(2025, 12, 25), "Christmas"),
            ],
            2026: [
                (datetime(2026, 1, 26), "Republic Day"),
                (datetime(2026, 3, 14), "Holi"),
                (datetime(2026, 3, 21), "Id-Ul-Fitr (Ramzan Id)"),
                (datetime(2026, 4, 3), "Good Friday"),
                (datetime(2026, 5, 1), "Maharashtra Day"),
                (datetime(2026, 5, 27), "Bakri Id"),
                (datetime(2026, 8, 15), "Independence Day"),
                (datetime(2026, 9, 5), "Janmashtami"),
                (datetime(2026, 10, 2), "Gandhi Jayanti"),
                (datetime(2026, 11, 8), "Diwali Laxmi Pujan"),
                (datetime(2026, 11, 24), "Guru Nanak Jayanti"),
                (datetime(2026, 12, 25), "Christmas"),
            ],
        }

        if year in holidays_data:
            holiday_dates = [holiday[0] for holiday in holidays_data[year]]
            logger.info(
                f"Using predefined holidays for {year}: {len(holiday_dates)} holidays"
            )
            return sorted(holiday_dates)
        else:
            # For years not in predefined data, try to extrapolate or return common fixed holidays
            logger.warning(
                f"No predefined holidays for year {year}, using common fixed holidays"
            )
            return self._get_common_fixed_holidays(year)

    def _get_common_fixed_holidays(self, year: int) -> list[datetime]:
        """Get common fixed holidays that occur every year"""
        fixed_holidays = [
            datetime(year, 1, 26),  # Republic Day
            datetime(year, 8, 15),  # Independence Day
            datetime(year, 10, 2),  # Gandhi Jayanti
            datetime(year, 12, 25),  # Christmas
        ]

        # Add May 1st if it falls on a weekday (Maharashtra Day)
        may_first = datetime(year, 5, 1)
        if may_first.weekday() < 5:  # Monday to Friday
            fixed_holidays.append(may_first)

        return sorted(fixed_holidays)

    def get_trading_calendar(
        self, start_date: datetime.date, end_date: datetime.date
    ) -> dict[str, any]:
        """Get trading calendar for a date range"""
        trading_days = []
        holidays = []
        weekends = []

        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() >= 5:  # Weekend
                weekends.append(current_date.isoformat())
            elif self.is_market_holiday(current_date):  # Holiday
                holidays.append(current_date.isoformat())
            else:  # Trading day
                trading_days.append(current_date.isoformat())

            current_date += timedelta(days=1)

        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "trading_days": trading_days,
            "holidays": holidays,
            "weekends": weekends,
            "total_trading_days": len(trading_days),
            "total_days": (end_date - start_date).days + 1,
        }


# Global instance
market_schedule = MarketScheduleManager()
