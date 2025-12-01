FROM nginx:alpine

# Copy entire frontend directory to Nginx html directory
# This includes both public/ (HTML) and src/ (CSS, JS)
# Copy public files to root
COPY apps/frontend/public/ /usr/share/nginx/html/
# Copy src files to src directory
COPY apps/frontend/src/ /usr/share/nginx/html/src/

# Expose port
EXPOSE 80

# Nginx runs automatically
