# Stage 1: Build the React app
FROM node:14 AS build

WORKDIR /home/atharav/react/docker_app/admin

# Copy package.json and package-lock.json
COPY package.json package-lock.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Adjust permissions for the entire working directory
RUN chmod -R 777 /home/atharav/react/docker_app/admin

# Build the React application
RUN npm run build

# Stage 2: Serve the app with a simple Node.js server
FROM node:14

WORKDIR /home/atharav/react/docker_app/admin

# Copy build files from the previous stage
COPY --from=build /home/atharav/react/docker_app/admin/build ./build

# Install serve package to serve static files
RUN npm install -g serve

# Expose the port serve is running on
EXPOSE 3000

# Command to run the static file server
CMD ["serve", "-s", "build"]
