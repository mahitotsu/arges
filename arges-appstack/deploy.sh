#!/bin/bash
bun run \
    --cwd ../arges-webapp \
    --bun \
    build && \
bunx --cwd ../arges-webapp \
    esbuild \
    --bundle ../arges-webapp/.output/server/index.mjs \
    --minify \
    --outdir=../arges-webapp/.output/dist \
    --platform=node && \
bun run cdk deploy \
    --method=direct \
    --no-staging \
    --verbose && \
date