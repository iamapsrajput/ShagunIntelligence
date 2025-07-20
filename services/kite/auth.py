"""Authentication and token management for Kite Connect"""

import asyncio
import aiofiles
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path
from loguru import logger

from kiteconnect import KiteConnect
from app.core.config import get_settings
from .exceptions import KiteAuthenticationError, KiteTokenExpiredError


class KiteAuthManager:
    """Manages authentication and token lifecycle for Kite Connect"""
    
    def __init__(self):
        self.settings = get_settings()
        self.kite = KiteConnect(api_key=self.settings.KITE_API_KEY)
        self.token_file = Path("data/kite_tokens.json")
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._user_profile: Optional[Dict[str, Any]] = None
        
        # Ensure data directory exists
        self.token_file.parent.mkdir(exist_ok=True)
        
    async def initialize(self) -> bool:
        """Initialize authentication from stored tokens or environment"""
        try:
            # Try to load from stored tokens first
            if await self._load_stored_tokens():
                if await self._validate_token():
                    logger.info("Successfully initialized with stored tokens")
                    return True
                else:
                    logger.warning("Stored tokens are invalid, clearing them")
                    await self._clear_stored_tokens()
            
            # Try to use environment token
            if self.settings.KITE_ACCESS_TOKEN:
                self._access_token = self.settings.KITE_ACCESS_TOKEN
                self.kite.set_access_token(self._access_token)
                
                if await self._validate_token():
                    await self._store_tokens()
                    logger.info("Successfully initialized with environment token")
                    return True
                else:
                    logger.error("Environment token is invalid")
                    self._access_token = None
            
            logger.warning("No valid tokens found. Manual authentication required.")
            return False
            
        except Exception as e:
            logger.error(f"Failed to initialize authentication: {str(e)}")
            return False
    
    def get_login_url(self) -> str:
        """Get the login URL for manual authentication"""
        return self.kite.login_url()
    
    async def generate_session(self, request_token: str) -> Dict[str, Any]:
        """Generate session using request token"""
        try:
            logger.info("Generating session with request token")
            
            # Generate session synchronously (KiteConnect doesn't support async)
            session_data = await asyncio.to_thread(
                self.kite.generate_session,
                request_token,
                api_secret=self.settings.KITE_API_SECRET
            )
            
            # Store the access token and expiry
            self._access_token = session_data["access_token"]
            self._token_expiry = datetime.now() + timedelta(days=1)  # Kite tokens expire daily
            
            # Set access token in kite instance
            self.kite.set_access_token(self._access_token)
            
            # Get user profile
            await self._fetch_user_profile()
            
            # Store tokens for future use
            await self._store_tokens()
            
            logger.info(f"Session generated successfully for user: {self._user_profile.get('user_name', 'Unknown')}")
            
            return {
                "access_token": self._access_token,
                "user_profile": self._user_profile,
                "expires_at": self._token_expiry.isoformat() if self._token_expiry else None
            }
            
        except Exception as e:
            logger.error(f"Failed to generate session: {str(e)}")
            raise KiteAuthenticationError(f"Session generation failed: {str(e)}")
    
    async def refresh_token_if_needed(self) -> bool:
        """Check and refresh token if it's about to expire"""
        if not self._access_token:
            logger.warning("No access token available for refresh")
            return False
        
        # Check if token is about to expire (within 1 hour)
        if self._token_expiry and datetime.now() >= (self._token_expiry - timedelta(hours=1)):
            logger.warning("Token is about to expire, manual re-authentication required")
            await self._clear_stored_tokens()
            self._access_token = None
            self._token_expiry = None
            return False
        
        # Validate current token
        if not await self._validate_token():
            logger.warning("Current token is invalid")
            await self._clear_stored_tokens()
            self._access_token = None
            self._token_expiry = None
            return False
        
        return True
    
    def get_access_token(self) -> Optional[str]:
        """Get current access token"""
        return self._access_token
    
    def get_user_profile(self) -> Optional[Dict[str, Any]]:
        """Get current user profile"""
        return self._user_profile
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return self._access_token is not None
    
    async def logout(self):
        """Logout and clear all stored tokens"""
        logger.info("Logging out and clearing tokens")
        await self._clear_stored_tokens()
        self._access_token = None
        self._token_expiry = None
        self._user_profile = None
    
    async def _validate_token(self) -> bool:
        """Validate current access token by making a test API call"""
        if not self._access_token:
            return False
        
        try:
            # Make a lightweight API call to validate token
            profile = await asyncio.to_thread(self.kite.profile)
            self._user_profile = profile
            logger.debug("Token validation successful")
            return True
        except Exception as e:
            logger.error(f"Token validation failed: {str(e)}")
            return False
    
    async def _fetch_user_profile(self):
        """Fetch and store user profile"""
        try:
            self._user_profile = await asyncio.to_thread(self.kite.profile)
        except Exception as e:
            logger.error(f"Failed to fetch user profile: {str(e)}")
            self._user_profile = None
    
    async def _load_stored_tokens(self) -> bool:
        """Load tokens from storage"""
        try:
            if not self.token_file.exists():
                return False
            
            async with aiofiles.open(self.token_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)
            
            self._access_token = data.get('access_token')
            expiry_str = data.get('expires_at')
            self._user_profile = data.get('user_profile')
            
            if expiry_str:
                self._token_expiry = datetime.fromisoformat(expiry_str)
            
            if self._access_token:
                self.kite.set_access_token(self._access_token)
                logger.debug("Loaded stored tokens")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to load stored tokens: {str(e)}")
            return False
    
    async def _store_tokens(self):
        """Store tokens to file"""
        try:
            data = {
                'access_token': self._access_token,
                'expires_at': self._token_expiry.isoformat() if self._token_expiry else None,
                'user_profile': self._user_profile,
                'stored_at': datetime.now().isoformat()
            }
            
            async with aiofiles.open(self.token_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
            
            logger.debug("Stored tokens successfully")
            
        except Exception as e:
            logger.error(f"Failed to store tokens: {str(e)}")
    
    async def _clear_stored_tokens(self):
        """Clear stored tokens"""
        try:
            if self.token_file.exists():
                self.token_file.unlink()
                logger.debug("Cleared stored tokens")
        except Exception as e:
            logger.error(f"Failed to clear stored tokens: {str(e)}")