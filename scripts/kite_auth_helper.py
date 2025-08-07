#!/usr/bin/env python3
"""
Kite Connect Authentication Helper

This script helps automate the Kite Connect OAuth authentication flow
to obtain access tokens for the Shagun Intelligence trading platform.
"""

import os
import sys
import webbrowser
from urllib.parse import parse_qs, urlparse

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import get_settings


def update_env_file(key: str, value: str):
    """Update a key-value pair in the .env file."""
    env_file_path = ".env"

    # Read current content
    if os.path.exists(env_file_path):
        with open(env_file_path) as f:
            lines = f.readlines()
    else:
        lines = []

    # Update or add the key
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            updated = True
            break

    if not updated:
        lines.append(f"{key}={value}\n")

    # Write back to file
    with open(env_file_path, "w") as f:
        f.writelines(lines)

    print(f"✅ Updated {key} in .env file")


def extract_request_token_from_url(url: str) -> str:
    """Extract request token from the callback URL."""
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)

    if "request_token" in query_params:
        return query_params["request_token"][0]
    else:
        raise ValueError("Request token not found in URL")


def main():
    """Main authentication flow."""
    print("🔐 Kite Connect Authentication Helper")
    print("=" * 50)

    # Load settings
    try:
        settings = get_settings()
    except Exception as e:
        print(f"❌ Error loading settings: {e}")
        print("Please ensure your .env file has KITE_API_KEY and KITE_API_SECRET")
        return

    if not settings.KITE_API_KEY or not settings.KITE_API_SECRET:
        print("❌ Missing Kite Connect credentials")
        print("Please add KITE_API_KEY and KITE_API_SECRET to your .env file")
        return

    try:
        from kiteconnect import KiteConnect
    except ImportError:
        print("❌ kiteconnect package not installed")
        print("Install it with: poetry add kiteconnect")
        return

    # Initialize Kite Connect
    kite = KiteConnect(api_key=settings.KITE_API_KEY)

    # Step 1: Generate login URL
    login_url = kite.login_url()
    print("\n🌐 Step 1: Authentication URL Generated")
    print(f"URL: {login_url}")

    # Step 2: Open browser
    print("\n🚀 Step 2: Opening browser for authentication...")
    try:
        webbrowser.open(login_url)
        print("✅ Browser opened successfully")
    except Exception as e:
        print(f"⚠️ Could not open browser automatically: {e}")
        print(f"Please manually visit: {login_url}")

    # Step 3: Get callback URL from user
    print("\n📋 Step 3: Complete Authentication")
    print("1. Log in with your Zerodha credentials")
    print("2. Authorize the application")
    print("3. You'll be redirected to a callback URL")
    print("4. Copy the entire callback URL and paste it below")
    print()

    callback_url = input("📎 Paste the callback URL here: ").strip()

    if not callback_url:
        print("❌ No callback URL provided")
        return

    # Step 4: Extract request token
    try:
        request_token = extract_request_token_from_url(callback_url)
        print(f"✅ Request token extracted: {request_token[:10]}...")

        # Update .env file with request token
        update_env_file("KITE_REQUEST_TOKEN", request_token)

    except Exception as e:
        print(f"❌ Error extracting request token: {e}")
        print("Please ensure you copied the complete callback URL")
        return

    # Step 5: Generate access token
    print("\n🔑 Step 4: Generating Access Token...")
    try:
        data = kite.generate_session(request_token, api_secret=settings.KITE_API_SECRET)
        access_token = data["access_token"]
        user_id = data["user_id"]

        print("✅ Access token generated successfully")
        print(f"User ID: {user_id}")
        print(f"Access Token: {access_token[:10]}...")

        # Update .env file with access token
        update_env_file("KITE_ACCESS_TOKEN", access_token)

    except Exception as e:
        print(f"❌ Error generating access token: {e}")
        return

    # Step 6: Verify authentication
    print("\n🧪 Step 5: Verifying Authentication...")
    try:
        kite.set_access_token(access_token)
        profile = kite.profile()

        print("✅ Authentication successful!")
        print(f"Account: {profile['user_name']} ({profile['email']})")
        print(f"Broker: {profile['broker']}")

    except Exception as e:
        print(f"❌ Authentication verification failed: {e}")
        return

    print("\n🎉 Setup Complete!")
    print("=" * 30)
    print("Your Kite Connect authentication is now configured.")
    print("You can now run the trading platform with full API access.")
    print()
    print("Next steps:")
    print("1. Run: poetry run python scripts/validate_api_keys.py")
    print("2. Start platform: poetry run uvicorn app.main:app --reload")
    print()
    print("⚠️ Note: Access tokens expire daily at 6:00 AM IST")
    print("You'll need to regenerate them for continuous operation.")


if __name__ == "__main__":
    main()
