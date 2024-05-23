import { RemovalPolicy, ScopedAws } from "aws-cdk-lib";
import { AllowedMethods, CachePolicy, CfnDistribution, CfnOriginAccessControl, Distribution, KeyGroup, OriginRequestPolicy, PublicKey, ResponseHeadersPolicy, ViewerProtocolPolicy } from "aws-cdk-lib/aws-cloudfront";
import { FunctionUrlOrigin } from "aws-cdk-lib/aws-cloudfront-origins";
import { ServicePrincipal } from "aws-cdk-lib/aws-iam";
import { Alias, FunctionUrlAuthType, InvokeMode, ParamsAndSecretsLayerVersion, ParamsAndSecretsVersions, Runtime } from "aws-cdk-lib/aws-lambda";
import { NodejsFunction } from "aws-cdk-lib/aws-lambda-nodejs";
import { LogGroup, RetentionDays } from "aws-cdk-lib/aws-logs";
import { Secret } from "aws-cdk-lib/aws-secretsmanager";
import { Construct } from "constructs";
import { KeyPairProvider } from "./KeyPairProvider";
import { WebServer } from "./WebServer";

export class EdgeLocation extends Construct {

    constructor(scope: Construct, id: string, props: {
        keyPairProvider: KeyPairProvider;
        webServer: WebServer;
    }) {
        super(scope, id);
        const { accountId } = new ScopedAws(this);

        const secret = new Secret(this, 'Secret', {
            secretObjectValue: {
                privateKey: props.keyPairProvider.privateKeyAsJsonString
            },
        });

        const publicKeyForDistribution = new PublicKey(this, props.keyPairProvider.keyPairName, {
            encodedKey: props.keyPairProvider.publicKey,
        });
        const keyGroup = new KeyGroup(this, 'KeyGroup', {
            items: [publicKeyForDistribution]
        });

        const signIn = (() => {
            const signIn = new Construct(this, 'SignIn');
            const handler = new NodejsFunction(signIn, 'Handler', {
                runtime: Runtime.NODEJS_LATEST,
                entry: `${__dirname}/EdgeLocation/signInHandler.ts`,
                handler: 'handler',
                paramsAndSecrets: ParamsAndSecretsLayerVersion.fromVersion(ParamsAndSecretsVersions.V1_0_103),
                environment: {
                    'SECRET_NAME': secret.secretName,
                    'KEY_PAIR_ID': publicKeyForDistribution.publicKeyId,
                },
                logGroup: new LogGroup(signIn, 'HandlerLog', {
                    removalPolicy: RemovalPolicy.DESTROY,
                    retention: RetentionDays.ONE_DAY,
                }),
            });
            const current = new Alias(signIn, 'Current', {
                aliasName: 'current',
                version: handler.currentVersion,
            });
            secret.grantRead(current);
            const endpoint = current.addFunctionUrl({
                authType: FunctionUrlAuthType.AWS_IAM,
                invokeMode: InvokeMode.BUFFERED,
            });
            return { handler: current, endpoint, };
        })();

        const oacLambda = new CfnOriginAccessControl(this, 'OACLambda', {
            originAccessControlConfig: {
                name: 'OACLambda',
                originAccessControlOriginType: 'lambda',
                signingBehavior: 'always',
                signingProtocol: 'sigv4',
            },
        });

        const distribution = new Distribution(this, 'Distribution', {
            defaultBehavior: {
                origin: new FunctionUrlOrigin(props.webServer.endpoint),
                viewerProtocolPolicy: ViewerProtocolPolicy.HTTPS_ONLY,
                originRequestPolicy: OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                responseHeadersPolicy: ResponseHeadersPolicy.SECURITY_HEADERS,
                cachePolicy: CachePolicy.CACHING_DISABLED,
                allowedMethods: AllowedMethods.ALLOW_ALL,
                trustedKeyGroups: [keyGroup],
            },
            additionalBehaviors: {
                '/sign-in': {
                    origin: new FunctionUrlOrigin(signIn.endpoint),
                    viewerProtocolPolicy: ViewerProtocolPolicy.HTTPS_ONLY,
                    originRequestPolicy: OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                    responseHeadersPolicy: ResponseHeadersPolicy.SECURITY_HEADERS,
                    cachePolicy: CachePolicy.CACHING_DISABLED,
                    allowedMethods: AllowedMethods.ALLOW_GET_HEAD,
                }
            }
        });
        const cfnDistribution = distribution.node.defaultChild as CfnDistribution;
        const oacLambdaPermission = {
            principal: new ServicePrincipal('cloudfront.amazonaws.com'),
            action: 'lambda:InvokeFunctionUrl',
            sourceArn: `arn:aws:cloudfront::${accountId}:distribution/${distribution.distributionId}`,
        };

        cfnDistribution.addPropertyOverride('DistributionConfig.Origins.0.OriginAccessControlId', oacLambda.attrId);
        cfnDistribution.addPropertyOverride('DistributionConfig.Origins.1.OriginAccessControlId', oacLambda.attrId);
        props.webServer.handler.addPermission('AllowCloudFrontServicePrincipal', oacLambdaPermission);
        signIn.handler.addPermission('AllowCloudFrontServicePrincipal', oacLambdaPermission);

        this._domainName = distribution.domainName;
    }

    private readonly _domainName: string;

    get domainName() { return this._domainName; }
}