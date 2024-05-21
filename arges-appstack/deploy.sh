#!/bin/bash
cdk deploy \
    --buid "pnpm --prefix ../arges-webapp build" \
    --method direct \
    --no-staging \
    --all
date