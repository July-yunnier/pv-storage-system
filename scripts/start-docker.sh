#!/usr/bin/env bash
# Run from any directory. --offline uses only preloaded images and never builds/pulls.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode="${1:---build}"
case "$mode" in
  --build|--offline) ;;
  *) echo "Usage: bash scripts/start-docker.sh [--build|--offline]" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/start-docker.sh [--build|--offline]" >&2
  exit 2
fi
command -v docker >/dev/null || { echo 'Docker with Compose v2 is required.' >&2; exit 1; }
docker info >/dev/null
docker compose version >/dev/null
docker compose config --quiet

if [[ "$mode" == '--offline' ]]; then
  docker compose up -d --no-build --pull never --wait --wait-timeout 180
else
  docker compose up -d --build --wait --wait-timeout 180
fi
docker compose ps
echo "UI: http://localhost:${FRONTEND_PORT:-8080}"
echo "API docs: http://localhost:${BACKEND_PORT:-8000}/docs"
echo 'Stop: docker compose down (keeps stored data). Logs: docker compose logs -f'
