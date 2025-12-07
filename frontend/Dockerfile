# Use Node.js for building/serving
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY frontend/package.json frontend/package-lock.json* ./

# Install dependencies
RUN npm install

# Copy frontend code
COPY frontend/ .

# Expose Vite default port
EXPOSE 5173

# Run Vite dev server with host 0.0.0.0 to allow external access
CMD ["npm", "run", "dev", "--", "--host"]
