#!/bin/bash
# engram/deploy/alibaba_cloud.sh
#
# Deploys Engram backend to Alibaba Cloud ECS.
# Run from your local machine with the ECS SSH key configured.
#
# Prerequisites:
#   - Alibaba Cloud ECS instance (Ubuntu 22.04, minimum 4GB RAM)
#   - SSH key configured in ~/.ssh/alibaba_ecs
#   - Docker installed on ECS
#   - DASHSCOPE_API_KEY exported in your shell
#
# Usage:
#   export ECS_HOST=your-ecs-ip
#   export DASHSCOPE_API_KEY=sk-xxx
#   bash deploy/alibaba_cloud.sh

set -euo pipefail

: "${ECS_HOST:?ECS_HOST not set. Export your Alibaba Cloud ECS IP.}"
: "${DASHSCOPE_API_KEY:?DASHSCOPE_API_KEY not set.}"

ECS_USER="${ECS_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/alibaba_ecs}"
REMOTE_DIR="/opt/engram"

echo "═══════════════════════════════════════════════════"
echo "  Engram → Alibaba Cloud ECS Deployment"
echo "  Host: $ECS_HOST"
echo "  User: $ECS_USER"
echo "═══════════════════════════════════════════════════"

SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no $ECS_USER@$ECS_HOST"

# 1. Ensure Docker is installed on ECS
echo "▸ Checking Docker on ECS..."
$SSH "docker --version || (curl -fsSL https://get.docker.com | sh && systemctl enable docker && systemctl start docker)"

# 2. Create deployment directory
echo "▸ Setting up directories..."
$SSH "mkdir -p $REMOTE_DIR/data/{db,chroma,skills}"

# 3. Copy backend code
echo "▸ Syncing backend code..."
rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.env' \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    ./backend/ "$ECS_USER@$ECS_HOST:$REMOTE_DIR/backend/"

# 4. Write .env on ECS (never committed to git)
echo "▸ Writing environment config..."
$SSH "cat > $REMOTE_DIR/backend/.env << 'EOF'
DASHSCOPE_API_KEY=$DASHSCOPE_API_KEY
ACTIVE_MODEL=qwen3-32b
CONSOLIDATOR_MODEL=qwen3-32b
DB_PATH=/data/db/engram.db
CHROMA_PATH=/data/chroma
SKILLS_PATH=/data/skills
CONSOLIDATION_CRON=*/30 * * * *
DECAY_DEPRECATE_THRESHOLD=0.06
CONTEXT_BUDGET_TOKENS=4000
ALLOWED_ORIGINS=*
PORT=8000
EOF"

# 5. Build Docker image on ECS
echo "▸ Building Docker image..."
$SSH "cd $REMOTE_DIR/backend && docker build -t engram-backend:latest ."

# 6. Stop old container if running
echo "▸ Stopping old container..."
$SSH "docker stop engram-backend 2>/dev/null || true && docker rm engram-backend 2>/dev/null || true"

# 7. Start new container
echo "▸ Starting Engram backend..."
$SSH "docker run -d \
    --name engram-backend \
    --restart unless-stopped \
    -p 8000:8000 \
    -v $REMOTE_DIR/data:/data \
    --env-file $REMOTE_DIR/backend/.env \
    engram-backend:latest"

# 8. Wait for health check
echo "▸ Waiting for health check..."
sleep 10
$SSH "curl -sf http://localhost:8000/health && echo 'HEALTH: OK' || echo 'HEALTH: FAILED'"

# 9. Print deployment proof info
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Deployment complete!"
echo "  Backend: http://$ECS_HOST:8000"
echo "  Health:  http://$ECS_HOST:8000/health"
echo "  API docs: http://$ECS_HOST:8000/docs"
echo ""
echo "  For the hackathon proof recording:"
echo "  curl http://$ECS_HOST:8000/health"
echo "  curl http://$ECS_HOST:8000/metrics/summary"
echo "═══════════════════════════════════════════════════"
