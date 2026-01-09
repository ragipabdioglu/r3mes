FROM nginx:alpine

# Install certbot for Let's Encrypt and envsubst for template processing
RUN apk add --no-cache certbot certbot-nginx curl gettext

# Copy nginx configuration template
COPY nginx.conf /etc/nginx/templates/nginx.conf.template

# Create directories for certbot
RUN mkdir -p /var/www/certbot /etc/letsencrypt

# Create entrypoint script for SSL initialization and template processing
RUN echo '#!/bin/sh' > /entrypoint.sh && \
    echo 'set -e' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Process nginx config template with environment variables' >> /entrypoint.sh && \
    echo 'if [ -n "$$DOMAIN" ]; then' >> /entrypoint.sh && \
    echo '  envsubst "$$DOMAIN" < /etc/nginx/templates/nginx.conf.template > /etc/nginx/nginx.conf' >> /entrypoint.sh && \
    echo 'else' >> /entrypoint.sh && \
    echo '  echo "Warning: DOMAIN not set, using default r3mes.network"' >> /entrypoint.sh && \
    echo '  export DOMAIN=r3mes.network' >> /entrypoint.sh && \
    echo '  envsubst "$$DOMAIN" < /etc/nginx/templates/nginx.conf.template > /etc/nginx/nginx.conf' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Start nginx in background' >> /entrypoint.sh && \
    echo 'nginx -g "daemon on;"' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Wait for nginx to start' >> /entrypoint.sh && \
    echo 'sleep 2' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Run certbot renewal in background' >> /entrypoint.sh && \
    echo 'while :; do' >> /entrypoint.sh && \
    echo '  certbot renew --quiet --no-self-upgrade' >> /entrypoint.sh && \
    echo '  nginx -s reload || true' >> /entrypoint.sh && \
    echo '  sleep 12h' >> /entrypoint.sh && \
    echo 'done &' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Keep nginx in foreground' >> /entrypoint.sh && \
    echo 'nginx -g "daemon off;"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Expose ports
EXPOSE 80 443

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Use entrypoint script
ENTRYPOINT ["/entrypoint.sh"]

