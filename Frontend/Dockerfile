# Stage 1: Build the React application
FROM node:18-alpine as build

WORKDIR /app

# Copy package.json and package-lock.json
COPY Frontend/package.json Frontend/package-lock.json ./

# Install dependencies
RUN npm install

# Copy the rest of the frontend application files
COPY Frontend/ ./

# Build the application
RUN npm run build

# Stage 2: Serve the application with Nginx
FROM nginx:1.21.3-alpine

# Copy the custom Nginx configuration
COPY Frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Copy the built files from the build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
