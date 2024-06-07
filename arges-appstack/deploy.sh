#!/bin/bash
bun run \
    --cwd ../arges-webapp \
    build && \
bun run cdk deploy \
    --method=direct \
    --no-staging \
    --verbose && \
date