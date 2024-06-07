#!/usr/bin/env node
import { App } from 'aws-cdk-lib';
import 'source-map-support/register';
import { AppStack } from '../lib/AppStack';

const app = new App();
new AppStack(app, 'AppStack', {
  env: {
    region: 'ap-northeast-1', account: process.env.CDK_DEFAULT_ACCOUNT
  }
});