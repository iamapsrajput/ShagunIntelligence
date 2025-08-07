#!/bin/bash

# SSL/TLS Certificate Setup Script for Shagun Intelligence Trading Platform
# Supports both Let's Encrypt (production) and self-signed certificates (development)

set -e

# Configuration
DOMAIN="${DOMAIN:-shagunintelligence.com}"
EMAIL="${EMAIL:-admin@shagunintelligence.com}"
SSL_DIR="${SSL_DIR:-./nginx/ssl}"
CERT_TYPE="${CERT_TYPE:-letsencrypt}"  # Options: letsencrypt, selfsigned

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running as root (required for Let's Encrypt)
check_root() {
    if [[ $CERT_TYPE == "letsencrypt" && $EUID -ne 0 ]]; then
        error "Let's Encrypt setup requires root privileges. Please run with sudo."
        exit 1
    fi
}

# Create SSL directory
create_ssl_directory() {
    log "Creating SSL directory: $SSL_DIR"
    mkdir -p "$SSL_DIR"
    chmod 700 "$SSL_DIR"
}

# Install certbot if not present
install_certbot() {
    if ! command -v certbot &> /dev/null; then
        log "Installing certbot..."
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            apt-get update
            apt-get install -y certbot python3-certbot-nginx
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew &> /dev/null; then
                brew install certbot
            else
                error "Please install Homebrew first or install certbot manually"
                exit 1
            fi
        else
            error "Unsupported operating system for automatic certbot installation"
            exit 1
        fi
    fi
}

# Generate Let's Encrypt certificate
generate_letsencrypt_cert() {
    log "Generating Let's Encrypt certificate for $DOMAIN"

    # Stop nginx if running to free port 80
    if systemctl is-active --quiet nginx 2>/dev/null; then
        log "Stopping nginx temporarily..."
        systemctl stop nginx
        RESTART_NGINX=true
    fi

    # Generate certificate using standalone mode
    certbot certonly \
        --standalone \
        --non-interactive \
        --agree-tos \
        --email "$EMAIL" \
        -d "$DOMAIN" \
        -d "www.$DOMAIN"

    # Copy certificates to our SSL directory
    cp "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" "$SSL_DIR/"
    cp "/etc/letsencrypt/live/$DOMAIN/privkey.pem" "$SSL_DIR/"
    cp "/etc/letsencrypt/live/$DOMAIN/chain.pem" "$SSL_DIR/"

    # Set proper permissions
    chmod 644 "$SSL_DIR/fullchain.pem" "$SSL_DIR/chain.pem"
    chmod 600 "$SSL_DIR/privkey.pem"

    success "Let's Encrypt certificate generated successfully"
}

# Generate self-signed certificate for development
generate_selfsigned_cert() {
    log "Generating self-signed certificate for development"

    # Create OpenSSL configuration
    cat > "$SSL_DIR/openssl.conf" << EOF
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
C = IN
ST = Maharashtra
L = Mumbai
O = Shagun Intelligence
OU = Trading Platform
CN = $DOMAIN

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = $DOMAIN
DNS.2 = www.$DOMAIN
DNS.3 = localhost
IP.1 = 127.0.0.1
EOF

    # Generate private key
    openssl genrsa -out "$SSL_DIR/privkey.pem" 2048

    # Generate certificate
    openssl req -new -x509 -key "$SSL_DIR/privkey.pem" \
        -out "$SSL_DIR/fullchain.pem" \
        -days 365 \
        -config "$SSL_DIR/openssl.conf" \
        -extensions v3_req

    # Create chain file (same as fullchain for self-signed)
    cp "$SSL_DIR/fullchain.pem" "$SSL_DIR/chain.pem"

    # Set proper permissions
    chmod 644 "$SSL_DIR/fullchain.pem" "$SSL_DIR/chain.pem"
    chmod 600 "$SSL_DIR/privkey.pem"

    # Clean up
    rm "$SSL_DIR/openssl.conf"

    success "Self-signed certificate generated successfully"
    warning "This is a self-signed certificate. Browsers will show security warnings."
}

# Setup certificate renewal (for Let's Encrypt)
setup_renewal() {
    if [[ $CERT_TYPE == "letsencrypt" ]]; then
        log "Setting up automatic certificate renewal"

        # Create renewal script
        cat > "/etc/cron.d/certbot-renewal" << EOF
# Renew Let's Encrypt certificates twice daily
0 */12 * * * root certbot renew --quiet --post-hook "systemctl reload nginx"
EOF

        # Test renewal
        certbot renew --dry-run

        success "Automatic renewal configured"
    fi
}

# Verify certificate
verify_certificate() {
    log "Verifying certificate..."

    if [[ -f "$SSL_DIR/fullchain.pem" && -f "$SSL_DIR/privkey.pem" ]]; then
        # Check certificate validity
        openssl x509 -in "$SSL_DIR/fullchain.pem" -text -noout | grep -E "(Subject:|Issuer:|Not After :)"

        # Check if certificate matches private key
        cert_hash=$(openssl x509 -noout -modulus -in "$SSL_DIR/fullchain.pem" | openssl md5)
        key_hash=$(openssl rsa -noout -modulus -in "$SSL_DIR/privkey.pem" | openssl md5)

        if [[ "$cert_hash" == "$key_hash" ]]; then
            success "Certificate and private key match"
        else
            error "Certificate and private key do not match!"
            exit 1
        fi
    else
        error "Certificate files not found!"
        exit 1
    fi
}

# Create DH parameters for enhanced security
generate_dhparam() {
    if [[ ! -f "$SSL_DIR/dhparam.pem" ]]; then
        log "Generating DH parameters (this may take a while)..."
        openssl dhparam -out "$SSL_DIR/dhparam.pem" 2048
        chmod 644 "$SSL_DIR/dhparam.pem"
        success "DH parameters generated"
    else
        log "DH parameters already exist"
    fi
}

# Main execution
main() {
    log "Starting SSL/TLS setup for Shagun Intelligence Trading Platform"
    log "Domain: $DOMAIN"
    log "Certificate type: $CERT_TYPE"

    check_root
    create_ssl_directory

    case $CERT_TYPE in
        "letsencrypt")
            install_certbot
            generate_letsencrypt_cert
            setup_renewal
            ;;
        "selfsigned")
            generate_selfsigned_cert
            ;;
        *)
            error "Invalid certificate type: $CERT_TYPE. Use 'letsencrypt' or 'selfsigned'"
            exit 1
            ;;
    esac

    generate_dhparam
    verify_certificate

    # Restart nginx if it was stopped
    if [[ $RESTART_NGINX == true ]]; then
        log "Restarting nginx..."
        systemctl start nginx
    fi

    success "SSL/TLS setup completed successfully!"
    log "Certificate files are located in: $SSL_DIR"
    log "- fullchain.pem: Full certificate chain"
    log "- privkey.pem: Private key"
    log "- chain.pem: Certificate chain"
    log "- dhparam.pem: DH parameters"

    if [[ $CERT_TYPE == "letsencrypt" ]]; then
        log "Automatic renewal is configured via cron"
    fi
}

# Show usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --domain DOMAIN     Domain name (default: shagunintelligence.com)"
    echo "  -e, --email EMAIL       Email for Let's Encrypt (default: admin@shagunintelligence.com)"
    echo "  -t, --type TYPE         Certificate type: letsencrypt or selfsigned (default: letsencrypt)"
    echo "  -s, --ssl-dir DIR       SSL directory (default: ./nginx/ssl)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --type selfsigned                    # Generate self-signed certificate"
    echo "  $0 --domain example.com --type letsencrypt  # Generate Let's Encrypt certificate"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--domain)
            DOMAIN="$2"
            shift 2
            ;;
        -e|--email)
            EMAIL="$2"
            shift 2
            ;;
        -t|--type)
            CERT_TYPE="$2"
            shift 2
            ;;
        -s|--ssl-dir)
            SSL_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Run main function
main
