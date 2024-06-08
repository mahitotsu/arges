#!/bin/bash
bun run \
    --bun \
    --cwd ../arges-webapp \
    build && \
bun run cdk deploy \
    --method=direct \
    --no-staging \
    --verbose && \
date