# Use official Nginx image
FROM nginx:alpine

# Remove default Nginx configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom Nginx configuration
COPY UI/nginx.conf /etc/nginx/conf.d/

# Copy static files (HTML, CSS, JS)
COPY UI/index.html /usr/share/nginx/html/
COPY UI/style.css /usr/share/nginx/html/
COPY UI/app.js /usr/share/nginx/html/

# Expose port 80
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
