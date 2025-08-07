#!/usr/bin/env python3
"""
Create a dedicated user for live trading with ₹1000 budget.
This ensures complete isolation from any existing users.
"""

import asyncio

from loguru import logger
from sqlalchemy.orm import Session

from app.core.auth import AuthService
from app.db.session import get_db
from app.models.user import User


async def create_live_trading_user():
    """Create a dedicated user for ₹1000 live trading."""

    # Get database session
    db_gen = get_db()
    db: Session = next(db_gen)

    try:
        # User details for live trading
        username = "live_trader_1000"
        password = "LiveTrading1000!Secure"
        email = "live.trading.1000@shagunintelligence.local"
        full_name = "Live Trading User (₹1000 Budget)"

        # Check if user already exists
        existing_user = await AuthService.get_user(db, username)
        if existing_user:
            logger.info(f"User '{username}' already exists")
            return username, password

        # Create the user
        logger.info(f"Creating live trading user: {username}")

        hashed_password = AuthService.get_password_hash(password)
        db_user = User(
            username=username,
            email=email,
            full_name=full_name,
            hashed_password=hashed_password,
            is_active=True,
            is_superuser=False,
        )

        db.add(db_user)
        db.commit()
        db.refresh(db_user)

        logger.success("✅ Live trading user created successfully!")
        logger.info(f"Username: {username}")
        logger.info(f"Password: {password}")
        logger.info(f"Email: {email}")

        return username, password

    except Exception as e:
        logger.error(f"Failed to create live trading user: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    username, password = asyncio.run(create_live_trading_user())

    print("\n" + "=" * 60)
    print("🔐 LIVE TRADING USER CREDENTIALS")
    print("=" * 60)
    print(f"Username: {username}")
    print(f"Password: {password}")
    print("=" * 60)
    print("⚠️  KEEP THESE CREDENTIALS SECURE!")
    print("⚠️  This user is specifically for ₹1000 budget testing")
    print("=" * 60)
