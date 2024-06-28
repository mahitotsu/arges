import { CfnOutput, RemovalPolicy, Stack, StackProps } from "aws-cdk-lib";
import { Certificate } from "aws-cdk-lib/aws-certificatemanager";
import { AllowedMethods, CachePolicy, CfnDistribution, CfnOriginAccessControl, Distribution, KeyGroup, OriginRequestPolicy, PublicKey } from "aws-cdk-lib/aws-cloudfront";
import { FunctionUrlOrigin, S3Origin } from "aws-cdk-lib/aws-cloudfront-origins";
import { Mfa, OAuthScope, UserPool } from "aws-cdk-lib/aws-cognito";
import { Effect, PolicyStatement, ServicePrincipal } from "aws-cdk-lib/aws-iam";
import { Alias, Architecture, Code, Function, FunctionUrlAuthType, InvokeMode, Runtime } from "aws-cdk-lib/aws-lambda";
import { ARecord, PublicHostedZone, RecordTarget } from "aws-cdk-lib/aws-route53";
import { CloudFrontTarget, UserPoolDomainTarget } from "aws-cdk-lib/aws-route53-targets";
import { BlockPublicAccess, Bucket } from "aws-cdk-lib/aws-s3";
import { BucketDeployment, Source } from "aws-cdk-lib/aws-s3-deployment";
import { Construct } from 'constructs';
import { KeyPairProvider } from "./keypair/KeyPairProvider";

export class AppStack extends Stack {

    constructor(scope: Construct, id: string, props: StackProps) {
        super(scope, id, props);

        // ==========
        // Domain names and certificates
        // ==========
        const domainName = 'mahitotsu.com';
        const webappRecordName = 'www';
        const authRecordName = 'auth';
        const hostedZone = PublicHostedZone.fromHostedZoneAttributes(this, 'HostedZone', {
            zoneName: 'mahitotsu.com', hostedZoneId: 'Z00285912B2ULPDZAM9V9',
        });
        const certificate = Certificate.fromCertificateArn(this, 'Certificate',
            `arn:aws:acm:us-east-1:${this.account}:certificate/053dc7b0-3805-42bd-8d17-28db8cc027bc`);
        const keyPair = new KeyPairProvider(this, 'KeyPair');
        const oauth2Prefix = '/oauth2'

        // ==========
        // ID provider
        // ==========
        const redirectPath = `${oauth2Prefix}/idpresponse`;
        const userPool = new UserPool(this, 'UserPool', {
            selfSignUpEnabled: false,
            signInAliases: { email: true, username: false, phone: false, preferredUsername: false },
            autoVerify: { email: true, phone: false },
            mfa: Mfa.OFF,
            removalPolicy: RemovalPolicy.DESTROY,
        });
        const authDomain = userPool.addDomain('AuthDomain', {
            customDomain: { domainName: `${authRecordName}.${domainName}`, certificate, }
        });
        const authRecord = new ARecord(this, 'AuthRecord', {
            recordName: authRecordName, zone: hostedZone,
            target: RecordTarget.fromAlias(new UserPoolDomainTarget(authDomain)),
        });
        const callbackUrl = `https://${webappRecordName}.${domainName}${redirectPath}`;
        const authClient = userPool.addClient('AuthClient', {
            generateSecret: true,
            authFlows: { userPassword: true },
            oAuth: {
                flows: { authorizationCodeGrant: true, },
                scopes: [OAuthScope.OPENID, OAuthScope.EMAIL],
                callbackUrls: [callbackUrl],
            }
        });

        // ==========
        // Nuxt3 server
        // ==========
        const serverFunction = new Function(this, 'ServerFunction', {
            environment: {
                NUXT_SIGN_IN_URL: authDomain.signInUrl(authClient, {
                    redirectUri: callbackUrl,
                })
            },
            code: Code.fromAsset(`${__dirname}/../../arges-webapp/.output/dist`),
            handler: 'index.handler',
            runtime: Runtime.NODEJS_20_X,
            architecture: Architecture.ARM_64,
            memorySize: 256,
            reservedConcurrentExecutions: 10,
        });
        const currentServer = new Alias(serverFunction, 'CurrentServer', {
            aliasName: 'current',
            version: serverFunction.currentVersion,
        });
        const serverUrl = currentServer.addFunctionUrl({
            invokeMode: InvokeMode.BUFFERED,
            authType: FunctionUrlAuthType.AWS_IAM,
        });

        // ==========
        // Nuxt3 static contents
        // ==========
        const publicAssetsBucket = new Bucket(this, 'PublicAssetsBucket', {
            removalPolicy: RemovalPolicy.DESTROY,
            autoDeleteObjects: true,
            publicReadAccess: false,
            blockPublicAccess: BlockPublicAccess.BLOCK_ALL,
        });
        new BucketDeployment(publicAssetsBucket, 'PublicAssetsDeployment', {
            destinationBucket: publicAssetsBucket,
            destinationKeyPrefix: 'public',
            sources: [Source.asset(`${__dirname}/../../arges-webapp/.output/public`)],
            memoryLimit: 512,
        });

        // ==========
        // Contents Distribution Network
        // ==========
        const lambdaOac = new CfnOriginAccessControl(this, 'LambdaOac', {
            originAccessControlConfig: {
                name: 'LambdaOac',
                originAccessControlOriginType: 'lambda',
                signingBehavior: 'always',
                signingProtocol: 'sigv4',
            }
        });
        const s3Oac = new CfnOriginAccessControl(this, 'S3Oac', {
            originAccessControlConfig: {
                name: 'S3Oac',
                originAccessControlOriginType: 's3',
                signingBehavior: 'always',
                signingProtocol: 'sigv4',
            }
        });

        const publicKey = new PublicKey(this, 'PublicKey', { encodedKey: keyPair.publicKey })
        const keyGroup = new KeyGroup(this, 'KeyGroup', { items: [publicKey] });
        const distribution = new Distribution(this, 'WebappDistribution', {
            domainNames: [`${webappRecordName}.${domainName}`],
            certificate,
            defaultBehavior: {
                origin: new FunctionUrlOrigin(serverUrl),
                cachePolicy: CachePolicy.CACHING_DISABLED,
                allowedMethods: AllowedMethods.ALLOW_ALL,
                originRequestPolicy: OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                compress: true,
                trustedKeyGroups: [keyGroup],
            },
            errorResponses: [{
                httpStatus: 403,
                responseHttpStatus: 200,
                responsePagePath: `${oauth2Prefix}/redirectToSignInUrl`,
            }]
        });
        distribution.addBehavior(`${oauth2Prefix}/*`, new FunctionUrlOrigin(serverUrl), {
            cachePolicy: CachePolicy.CACHING_DISABLED,
            allowedMethods: AllowedMethods.ALLOW_GET_HEAD_OPTIONS,
            originRequestPolicy: OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
        });
        distribution.addBehavior('*.*', new S3Origin(publicAssetsBucket, { originPath: 'public' }), {
            cachePolicy: CachePolicy.CACHING_DISABLED,
            allowedMethods: AllowedMethods.ALLOW_GET_HEAD_OPTIONS,
            originRequestPolicy: OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
            compress: true,
            trustedKeyGroups: [keyGroup],
        });
        const webappRecord = new ARecord(this, 'WebappRecord', {
            recordName: webappRecordName, zone: hostedZone,
            target: RecordTarget.fromAlias(new CloudFrontTarget(distribution)),
        });

        const cfnDistribution = distribution.node.defaultChild as CfnDistribution;
        cfnDistribution.addPropertyOverride('DistributionConfig.Origins.0.OriginAccessControlId', lambdaOac.attrId);
        cfnDistribution.addPropertyOverride('DistributionConfig.Origins.1.OriginAccessControlId', lambdaOac.attrId);
        cfnDistribution.addPropertyOverride('DistributionConfig.Origins.2.OriginAccessControlId', s3Oac.attrId);
        cfnDistribution.addPropertyOverride('DistributionConfig.Origins.2.S3OriginConfig.OriginAccessIdentity', '');

        currentServer.addPermission('OacPermission', {
            principal: new ServicePrincipal('cloudfront.amazonaws.com'),
            action: 'lambda:InvokeFunctionUrl',
            sourceArn: `arn:aws:cloudfront::${this.account}:distribution/${distribution.distributionId}`,
        });
        publicAssetsBucket.addToResourcePolicy(new PolicyStatement({
            effect: Effect.ALLOW,
            principals: [new ServicePrincipal('cloudfront.amazonaws.com')],
            actions: ["s3:GetObject"],
            resources: [publicAssetsBucket.arnForObjects('*')],
            conditions: {
                StringEquals: {
                    'aws:SourceArn': `arn:aws:cloudfront::${this.account}:distribution/${distribution.distributionId}`,
                }
            }
        }));

        // ==========
        // Outputs of this stack
        // ==========
        new CfnOutput(this, 'WebappUrl', { value: `https://${webappRecord.domainName}` });
    }
}