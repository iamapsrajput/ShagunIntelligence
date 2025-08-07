"""
Market Schedule API endpoints
Provides market hours, holidays, and trading calendar information
"""

from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel

from app.services.market_schedule import market_schedule

router = APIRouter()


class MarketStatusResponse(BaseModel):
    """Market status response model"""

    status: str
    message: str
    is_open: bool
    is_pre_market: bool
    is_after_hours: bool
    is_holiday: bool
    is_weekend: bool
    current_time: str
    market_open_time: str
    market_close_time: str
    next_market_open: str = None
    timezone: str


class TradingCalendarResponse(BaseModel):
    """Trading calendar response model"""

    start_date: str
    end_date: str
    trading_days: list[str]
    holidays: list[str]
    weekends: list[str]
    total_trading_days: int
    total_days: int


@router.get("/status", response_model=MarketStatusResponse)
async def get_market_status() -> MarketStatusResponse:
    """Get current market status"""
    try:
        status = market_schedule.get_market_status()
        return MarketStatusResponse(**status)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get market status: {str(e)}"
        )


@router.get("/is-open")
async def is_market_open() -> dict[str, Any]:
    """Check if market is currently open"""
    try:
        is_open = market_schedule.is_market_open()
        current_time = market_schedule.get_current_market_time()

        return {
            "is_open": is_open,
            "current_time": current_time.isoformat(),
            "timezone": str(market_schedule.timezone),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to check market status: {str(e)}"
        )


@router.get("/next-open")
async def get_next_market_open() -> dict[str, Any]:
    """Get next market open time"""
    try:
        next_open = market_schedule.get_next_market_open()
        next_close = market_schedule.get_next_market_close()

        return {
            "next_market_open": next_open.isoformat() if next_open else None,
            "next_market_close": next_close.isoformat() if next_close else None,
            "timezone": str(market_schedule.timezone),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get next market times: {str(e)}"
        )


@router.get("/holidays/{year}")
async def get_market_holidays(year: int) -> dict[str, Any]:
    """Get market holidays for a specific year"""
    try:
        if year < 2020 or year > 2030:
            raise HTTPException(
                status_code=400, detail="Year must be between 2020 and 2030"
            )

        holidays = market_schedule.get_market_holidays(year)

        return {
            "year": year,
            "holidays": [h.date().isoformat() for h in holidays],
            "total_holidays": len(holidays),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get holidays: {str(e)}")


@router.get("/calendar", response_model=TradingCalendarResponse)
async def get_trading_calendar(
    start_date: date = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: date = Query(..., description="End date (YYYY-MM-DD)"),
) -> TradingCalendarResponse:
    """Get trading calendar for a date range"""
    try:
        if start_date > end_date:
            raise HTTPException(
                status_code=400, detail="Start date must be before end date"
            )

        if (end_date - start_date).days > 365:
            raise HTTPException(
                status_code=400, detail="Date range cannot exceed 365 days"
            )

        calendar = market_schedule.get_trading_calendar(start_date, end_date)
        return TradingCalendarResponse(**calendar)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get trading calendar: {str(e)}"
        )


@router.get("/today")
async def get_today_schedule() -> dict[str, Any]:
    """Get today's market schedule"""
    try:
        today = market_schedule.get_current_market_time().date()
        status = market_schedule.get_market_status()

        # Get today's trading info
        is_trading_day = not (status["is_weekend"] or status["is_holiday"])

        return {
            "date": today.isoformat(),
            "is_trading_day": is_trading_day,
            "is_weekend": status["is_weekend"],
            "is_holiday": status["is_holiday"],
            "market_status": status["status"],
            "market_open_time": status["market_open_time"],
            "market_close_time": status["market_close_time"],
            "current_time": status["current_time"],
            "next_market_open": status["next_market_open"],
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get today's schedule: {str(e)}"
        )


@router.get("/this-week")
async def get_this_week_schedule() -> dict[str, Any]:
    """Get this week's trading schedule"""
    try:
        today = market_schedule.get_current_market_time().date()

        # Get start of week (Monday)
        start_of_week = today - timedelta(days=today.weekday())
        end_of_week = start_of_week + timedelta(days=6)

        calendar = market_schedule.get_trading_calendar(start_of_week, end_of_week)

        # Add day-wise breakdown
        week_schedule = []
        current_date = start_of_week

        while current_date <= end_of_week:
            day_info = {
                "date": current_date.isoformat(),
                "day_name": current_date.strftime("%A"),
                "is_trading_day": current_date.isoformat() in calendar["trading_days"],
                "is_weekend": current_date.isoformat() in calendar["weekends"],
                "is_holiday": current_date.isoformat() in calendar["holidays"],
            }
            week_schedule.append(day_info)
            current_date += timedelta(days=1)

        return {
            "week_start": start_of_week.isoformat(),
            "week_end": end_of_week.isoformat(),
            "trading_days_count": calendar["total_trading_days"],
            "schedule": week_schedule,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get week schedule: {str(e)}"
        )


@router.get("/next-trading-days/{count}")
async def get_next_trading_days(count: int = Path(..., ge=1, le=30)) -> dict[str, Any]:
    """Get next N trading days"""
    try:
        today = market_schedule.get_current_market_time().date()
        trading_days = []

        current_date = today
        days_found = 0
        max_search_days = count * 2 + 30  # Safety limit
        search_count = 0

        while days_found < count and search_count < max_search_days:
            if current_date.weekday() < 5 and not market_schedule.is_market_holiday(
                current_date
            ):
                trading_days.append(
                    {
                        "date": current_date.isoformat(),
                        "day_name": current_date.strftime("%A"),
                        "days_from_today": (current_date - today).days,
                    }
                )
                days_found += 1

            current_date += timedelta(days=1)
            search_count += 1

        return {
            "requested_count": count,
            "found_count": len(trading_days),
            "trading_days": trading_days,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get next trading days: {str(e)}"
        )


@router.post("/check-date/{check_date}")
async def check_specific_date(check_date: date) -> dict[str, Any]:
    """Check if a specific date is a trading day"""
    try:
        is_weekend = check_date.weekday() >= 5
        is_holiday = market_schedule.is_market_holiday(check_date)
        is_trading_day = not (is_weekend or is_holiday)

        return {
            "date": check_date.isoformat(),
            "day_name": check_date.strftime("%A"),
            "is_trading_day": is_trading_day,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "reason": (
                "Weekend" if is_weekend else "Holiday" if is_holiday else "Trading day"
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check date: {str(e)}")
