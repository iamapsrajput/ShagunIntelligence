"""
Secure API key management with encryption and rotation support.
"""

import os
import json
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
from pathlib import Path
import asyncio
from loguru import logger
import aiofiles
from dataclasses import dataclass, asdict
from enum import Enum

from .api_config import APIProvider, get_api_config


class KeyStatus(str, Enum):
    """API key status."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING_ROTATION = "pending_rotation"


@dataclass
class APIKeyRecord:
    """Record for an API key."""
    provider: str
    key_type: str  # 'api_key', 'api_secret', 'access_token', 'bearer_token'
    encrypted_value: str
    created_at: datetime
    expires_at: Optional[datetime]
    last_rotated: Optional[datetime]
    status: KeyStatus
    usage_count: int = 0
    last_used: Optional[datetime] = None
    metadata: Dict[str, Any] = None


class APIKeyManager:
    """
    Secure API key management with encryption and rotation.
    
    Features:
    - Encryption at rest using Fernet
    - Automatic key rotation
    - Usage tracking
    - Secure key storage
    - Environment variable integration
    """
    
    def __init__(self, storage_path: str = None, master_password: str = None):
        self.storage_path = Path(storage_path or os.getenv("API_KEY_STORAGE", "./config/api_keys.enc"))
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption
        self._init_encryption(master_password)
        
        # Load existing keys
        self.keys: Dict[str, Dict[str, APIKeyRecord]] = {}
        self._load_keys()
        
        # Rotation settings
        self.rotation_days = get_api_config().api_key_rotation_days
        self.rotation_check_interval = 3600  # Check every hour
        self.rotation_task = None
        
        logger.info(f"APIKeyManager initialized with storage at {self.storage_path}")
    
    def _init_encryption(self, master_password: str = None):
        """Initialize encryption with master key."""
        if master_password:
            # Derive key from password
            password = master_password.encode()
            salt = b'shagunintelligence_api_salt'  # In production, use random salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            self.cipher = Fernet(key)
        else:
            # Use environment key or generate new one
            key = os.getenv("API_ENCRYPTION_KEY")
            if not key:
                key = Fernet.generate_key().decode()
                logger.warning(
                    f"Generated new encryption key. Save this to API_ENCRYPTION_KEY env var: {key}"
                )
            self.cipher = Fernet(key.encode() if isinstance(key, str) else key)
    
    def _load_keys(self):
        """Load encrypted keys from storage."""
        if not self.storage_path.exists():
            logger.info("No existing key storage found")
            return
        
        try:
            with open(self.storage_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher.decrypt(encrypted_data)
            data = json.loads(decrypted_data.decode())
            
            # Convert to APIKeyRecord objects
            for provider, keys in data.items():
                self.keys[provider] = {}
                for key_type, record_data in keys.items():
                    record_data['created_at'] = datetime.fromisoformat(record_data['created_at'])
                    if record_data.get('expires_at'):
                        record_data['expires_at'] = datetime.fromisoformat(record_data['expires_at'])
                    if record_data.get('last_rotated'):
                        record_data['last_rotated'] = datetime.fromisoformat(record_data['last_rotated'])
                    if record_data.get('last_used'):
                        record_data['last_used'] = datetime.fromisoformat(record_data['last_used'])
                    
                    self.keys[provider][key_type] = APIKeyRecord(**record_data)
            
            logger.info(f"Loaded {sum(len(k) for k in self.keys.values())} API keys")
            
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
    
    async def _save_keys(self):
        """Save encrypted keys to storage."""
        try:
            # Convert to serializable format
            data = {}
            for provider, keys in self.keys.items():
                data[provider] = {}
                for key_type, record in keys.items():
                    record_dict = asdict(record)
                    # Convert datetime objects to ISO format
                    for field in ['created_at', 'expires_at', 'last_rotated', 'last_used']:
                        if record_dict.get(field):
                            record_dict[field] = record_dict[field].isoformat()
                    data[provider][key_type] = record_dict
            
            # Encrypt and save
            json_data = json.dumps(data).encode()
            encrypted_data = self.cipher.encrypt(json_data)
            
            async with aiofiles.open(self.storage_path, 'wb') as f:
                await f.write(encrypted_data)
            
            logger.debug("API keys saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving API keys: {e}")
    
    async def set_key(
        self,
        provider: APIProvider,
        key_type: str,
        value: str,
        expires_at: Optional[datetime] = None
    ) -> bool:
        """Set an API key."""
        try:
            provider_str = provider.value
            
            # Encrypt the value
            encrypted_value = self.cipher.encrypt(value.encode()).decode()
            
            # Create or update record
            if provider_str not in self.keys:
                self.keys[provider_str] = {}
            
            # Check if updating existing key
            existing = self.keys[provider_str].get(key_type)
            last_rotated = datetime.now() if existing else None
            
            record = APIKeyRecord(
                provider=provider_str,
                key_type=key_type,
                encrypted_value=encrypted_value,
                created_at=existing.created_at if existing else datetime.now(),
                expires_at=expires_at,
                last_rotated=last_rotated,
                status=KeyStatus.ACTIVE,
                usage_count=existing.usage_count if existing else 0,
                metadata={}
            )
            
            self.keys[provider_str][key_type] = record
            await self._save_keys()
            
            logger.info(f"Set {key_type} for {provider_str}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting API key: {e}")
            return False
    
    def get_key(self, provider: APIProvider, key_type: str) -> Optional[str]:
        """Get a decrypted API key."""
        try:
            provider_str = provider.value
            
            # First check environment variables
            env_prefix = get_api_config().Config.env_prefix_map.get(provider_str, "")
            env_var = f"{env_prefix}{key_type.upper()}"
            env_value = os.getenv(env_var)
            
            if env_value:
                # Update usage stats
                self._update_usage(provider_str, key_type)
                return env_value
            
            # Check stored keys
            if provider_str in self.keys and key_type in self.keys[provider_str]:
                record = self.keys[provider_str][key_type]
                
                # Check if expired
                if record.expires_at and record.expires_at < datetime.now():
                    logger.warning(f"API key {key_type} for {provider_str} has expired")
                    record.status = KeyStatus.EXPIRED
                    return None
                
                # Check if revoked
                if record.status == KeyStatus.REVOKED:
                    logger.warning(f"API key {key_type} for {provider_str} is revoked")
                    return None
                
                # Decrypt and return
                decrypted_value = self.cipher.decrypt(record.encrypted_value.encode()).decode()
                
                # Update usage stats
                self._update_usage(provider_str, key_type)
                
                return decrypted_value
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting API key: {e}")
            return None
    
    def _update_usage(self, provider: str, key_type: str):
        """Update key usage statistics."""
        if provider in self.keys and key_type in self.keys[provider]:
            record = self.keys[provider][key_type]
            record.usage_count += 1
            record.last_used = datetime.now()
            
            # Check if rotation needed
            if record.last_rotated:
                days_since_rotation = (datetime.now() - record.last_rotated).days
                if days_since_rotation > self.rotation_days:
                    record.status = KeyStatus.PENDING_ROTATION
    
    async def rotate_key(
        self,
        provider: APIProvider,
        key_type: str,
        new_value: str,
        expires_at: Optional[datetime] = None
    ) -> bool:
        """Rotate an API key."""
        try:
            provider_str = provider.value
            
            # Get existing record
            if provider_str not in self.keys or key_type not in self.keys[provider_str]:
                logger.error(f"Cannot rotate non-existent key: {provider_str}/{key_type}")
                return False
            
            old_record = self.keys[provider_str][key_type]
            
            # Archive old key (keep for audit trail)
            archive_key = f"{key_type}_archived_{datetime.now().timestamp()}"
            old_record.status = KeyStatus.REVOKED
            self.keys[provider_str][archive_key] = old_record
            
            # Set new key
            success = await self.set_key(provider, key_type, new_value, expires_at)
            
            if success:
                logger.info(f"Successfully rotated {key_type} for {provider_str}")
                
                # Notify about rotation
                await self._notify_rotation(provider_str, key_type)
            
            return success
            
        except Exception as e:
            logger.error(f"Error rotating API key: {e}")
            return False
    
    async def _notify_rotation(self, provider: str, key_type: str):
        """Notify about key rotation."""
        # In production, this would send notifications
        logger.warning(
            f"API key rotation completed for {provider}/{key_type}. "
            f"Update any dependent systems."
        )
    
    def get_all_keys_status(self) -> Dict[str, Any]:
        """Get status of all API keys."""
        status = {}
        
        for provider in APIProvider:
            provider_str = provider.value
            provider_status = {
                'enabled': False,
                'keys': {}
            }
            
            # Check configuration
            config = get_api_config().get_api_config(provider)
            if config:
                provider_status['enabled'] = config.enabled
            
            # Check keys
            for key_type in ['api_key', 'api_secret', 'access_token', 'bearer_token']:
                key_info = {
                    'exists': False,
                    'source': None,
                    'status': None,
                    'expires_at': None,
                    'last_used': None,
                    'usage_count': 0
                }
                
                # Check environment
                env_prefix = get_api_config().Config.env_prefix_map.get(provider_str, "")
                env_var = f"{env_prefix}{key_type.upper()}"
                if os.getenv(env_var):
                    key_info['exists'] = True
                    key_info['source'] = 'environment'
                
                # Check stored
                elif provider_str in self.keys and key_type in self.keys[provider_str]:
                    record = self.keys[provider_str][key_type]
                    key_info['exists'] = True
                    key_info['source'] = 'storage'
                    key_info['status'] = record.status.value
                    key_info['expires_at'] = record.expires_at.isoformat() if record.expires_at else None
                    key_info['last_used'] = record.last_used.isoformat() if record.last_used else None
                    key_info['usage_count'] = record.usage_count
                
                if key_info['exists'] or key_type == 'api_key':  # Always show api_key
                    provider_status['keys'][key_type] = key_info
            
            status[provider_str] = provider_status
        
        return status
    
    def get_rotation_schedule(self) -> List[Dict[str, Any]]:
        """Get key rotation schedule."""
        schedule = []
        
        for provider, keys in self.keys.items():
            for key_type, record in keys.items():
                if record.status != KeyStatus.ACTIVE:
                    continue
                
                rotation_info = {
                    'provider': provider,
                    'key_type': key_type,
                    'last_rotated': record.last_rotated.isoformat() if record.last_rotated else None,
                    'next_rotation': None,
                    'days_until_rotation': None,
                    'status': 'ok'
                }
                
                if record.last_rotated:
                    next_rotation = record.last_rotated + timedelta(days=self.rotation_days)
                    rotation_info['next_rotation'] = next_rotation.isoformat()
                    days_until = (next_rotation - datetime.now()).days
                    rotation_info['days_until_rotation'] = days_until
                    
                    if days_until <= 0:
                        rotation_info['status'] = 'overdue'
                    elif days_until <= 7:
                        rotation_info['status'] = 'due_soon'
                
                schedule.append(rotation_info)
        
        return sorted(schedule, key=lambda x: x.get('days_until_rotation', float('inf')))
    
    async def start_rotation_monitor(self):
        """Start automatic rotation monitoring."""
        if self.rotation_task:
            return
        
        self.rotation_task = asyncio.create_task(self._rotation_monitor_loop())
        logger.info("Started API key rotation monitor")
    
    async def stop_rotation_monitor(self):
        """Stop rotation monitoring."""
        if self.rotation_task:
            self.rotation_task.cancel()
            self.rotation_task = None
            logger.info("Stopped API key rotation monitor")
    
    async def _rotation_monitor_loop(self):
        """Monitor keys for rotation."""
        while True:
            try:
                # Check all keys
                for provider, keys in self.keys.items():
                    for key_type, record in keys.items():
                        if record.status != KeyStatus.ACTIVE:
                            continue
                        
                        if record.last_rotated:
                            days_since = (datetime.now() - record.last_rotated).days
                            if days_since >= self.rotation_days:
                                record.status = KeyStatus.PENDING_ROTATION
                                logger.warning(
                                    f"Key {key_type} for {provider} is due for rotation "
                                    f"({days_since} days old)"
                                )
                
                # Save any status updates
                await self._save_keys()
                
                # Wait before next check
                await asyncio.sleep(self.rotation_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rotation monitor: {e}")
                await asyncio.sleep(60)  # Wait a minute on error
    
    def export_template(self, filepath: str = ".env.template"):
        """Export environment variable template."""
        template_lines = [
            "# Shagun Intelligence API Configuration Template",
            "# Copy this file to .env and fill in your API keys",
            "",
            "# Environment: development, staging, production",
            "ENVIRONMENT=development",
            "",
            "# Encryption key for API storage (generate with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')",
            "API_ENCRYPTION_KEY=",
            "",
        ]
        
        for provider in APIProvider:
            config = get_api_config().get_api_config(provider)
            if not config:
                continue
            
            prefix = get_api_config().Config.env_prefix_map.get(provider.value, "")
            
            template_lines.extend([
                f"# {provider.value.upper().replace('_', ' ')}",
                f"# Base URL: {config.base_url}",
                f"# Rate limit: {config.rate_limit_per_minute}/min",
                f"# Tier: {config.tier.value}",
                f"{prefix}API_KEY=",
                f"{prefix}API_SECRET=",
                f"{prefix}ACCESS_TOKEN=",
                f"{prefix}BEARER_TOKEN=",
                f"{prefix}ENABLED=true",
                ""
            ])
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(template_lines))
        
        logger.info(f"Exported API configuration template to {filepath}")


# Singleton instance
_key_manager_instance: Optional[APIKeyManager] = None

def get_api_key_manager(storage_path: str = None, master_password: str = None) -> APIKeyManager:
    """Get the API key manager instance."""
    global _key_manager_instance
    
    if _key_manager_instance is None:
        _key_manager_instance = APIKeyManager(storage_path, master_password)
    
    return _key_manager_instance