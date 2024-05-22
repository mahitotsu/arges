#!/bin/bash
pnpm --prefix ../arges-webapp build && \
cdk deploy \
    --method direct \
    --no-staging \
    --all
date