# Build stage
FROM golang:1.24-alpine AS builder

WORKDIR /app

# Copy go.mod and download dependencies
COPY go.mod ./
# RUN go mod download # Not strictly necessary if no external deps, but good practice

# Copy source code
COPY . .

# Build the application
RUN go build -o main .

# Run stage
FROM alpine:latest

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/main .

# Copy static assets
COPY --from=builder /app/static ./static

# Expose port
EXPOSE 8080

# Run the application
CMD ["./main"]
