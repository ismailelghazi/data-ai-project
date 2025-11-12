#!/bin/sh

# Usage: ./wait-for-db.sh <host> [port]
HOST="$1"
PORT="${2:-5432}"

echo "Waiting for PostgreSQL at $HOST:$PORT..."

# Wait until the database is ready
until pg_isready -h "$HOST" -p "$PORT"; do
  echo "Postgres is unavailable - sleeping"
  sleep 2
done

echo "Postgres is up - continuing..."
