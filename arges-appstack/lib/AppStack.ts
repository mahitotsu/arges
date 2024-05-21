import { CfnOutput, CustomResource, Duration, Names, RemovalPolicy, ScopedAws, Stack, StackProps } from "aws-cdk-lib";
import { AllowedMethods, CachePolicy, CfnDistribution, CfnOriginAccessControl, Distribution, KeyGroup, OriginRequestPolicy, PublicKey, ResponseHeadersPolicy, ViewerProtocolPolicy } from "aws-cdk-lib/aws-cloudfront";
import { FunctionUrlOrigin } from "aws-cdk-lib/aws-cloudfront-origins";
import { Effect, PolicyStatement, ServicePrincipal } from "aws-cdk-lib/aws-iam";
import { FunctionUrlAuthType, InvokeMode, Runtime } from "aws-cdk-lib/aws-lambda";
import { NodejsFunction, OutputFormat } from "aws-cdk-lib/aws-lambda-nodejs";
import { LogGroup, RetentionDays } from "aws-cdk-lib/aws-logs";
import { StringParameter } from "aws-cdk-lib/aws-ssm";
import { Provider } from "aws-cdk-lib/custom-resources";
import { Construct } from "constructs";

export class AppStack extends Stack {

    constructor(scope: Construct, id: string, props?: StackProps) {
        super(scope, id, props);
        const { accountId, region, stackName, } = new ScopedAws(this);

        // -----
        // generate a keypair and store the private key in parameter store
        // -----
        const { publicKey, privateKey } = (() => {
            const parent = new Construct(this, 'KeyPairGenerator');
            const handler = new NodejsFunction(parent, 'Handler', {
                runtime: Runtime.NODEJS_LATEST,
                entry: `${__dirname}/KeyPairGenerator.ts`,
                handler: 'handler',
                timeout: Duration.seconds(30),
                logGroup: new LogGroup(parent, 'HandlerLog', {
                    removalPolicy: RemovalPolicy.DESTROY,
                    retention: RetentionDays.ONE_DAY,
                }),
                initialPolicy: [
                    new PolicyStatement({
                        effect: Effect.ALLOW,
                        actions: ['ssm:PutParameter', 'ssm:DeleteParameter'],
                        resources: [`arn:aws:ssm:${region}:${accountId}:parameter/${stackName}/*`],
                    }),
                ],
            });
            const provider = new Provider(parent, 'Provider', {
                onEventHandler: handler,
                logGroup: new LogGroup(parent, 'ProviderLog', {
                    removalPolicy: RemovalPolicy.DESTROY,
                    retention: RetentionDays.ONE_DAY,
                }),
            });
            const resource = new CustomResource(parent, 'CustomResource', {
                serviceToken: provider.serviceToken,
                removalPolicy: RemovalPolicy.DESTROY,
            });
            const publicKey = StringParameter.fromStringParameterAttributes(parent, 'PublicKey', {
                parameterName: resource.getAttString('PublicKey'),
                simpleName: false,
                forceDynamicReference: true,
            });
            const privateKey = StringParameter.fromSecureStringParameterAttributes(parent, 'PrivateKey', {
                parameterName: resource.getAttString('PrivateKey'),
                simpleName: false,
            });
            return { publicKey, privateKey, }
        })();

        // -----
        //
        // -----
        const { webServer, entryPoint } = (() => {
            const parent = new Construct(this, 'WebServer');
            const webServer = new NodejsFunction(parent, 'Handler', {
                runtime: Runtime.NODEJS_LATEST,
                entry: `${__dirname}/../../arges-webapp/.output/server/index.mjs`,
                handler: 'handler',
                timeout: Duration.seconds(30),
                bundling: {
                    format: OutputFormat.ESM,
                },
                logGroup: new LogGroup(parent, 'HandlerLog', {
                    removalPolicy: RemovalPolicy.DESTROY,
                    retention: RetentionDays.ONE_DAY,
                }),
            });
            const entryPoint = webServer.addFunctionUrl({
                invokeMode: InvokeMode.BUFFERED,
                authType: FunctionUrlAuthType.AWS_IAM,
            })
            return { webServer, entryPoint };
        })();

        // -----
        // create a distribution and associate origins
        // -----
        (() => {
            const parent = new Construct(this, 'EdgeLocation');
            const publicKeyForDistribution = new PublicKey(parent, 'PublicKey', {
                encodedKey: publicKey.stringValue,
            });
            const keyGroup = new KeyGroup(parent, 'KeyGroup', {
                items: [publicKeyForDistribution]
            });
            const oacForDefaultOrigin = new CfnOriginAccessControl(parent, 'OACForDefaultOrigin', {
                originAccessControlConfig: {
                    name: Names.uniqueResourceName(parent, {}),
                    originAccessControlOriginType: 'lambda',
                    signingBehavior: 'always',
                    signingProtocol: 'sigv4',
                },
            });

            const distribution = new Distribution(parent, 'Distribution', {
                defaultBehavior: {
                    origin: new FunctionUrlOrigin(entryPoint),
                    viewerProtocolPolicy: ViewerProtocolPolicy.HTTPS_ONLY,
                    originRequestPolicy: OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                    responseHeadersPolicy: ResponseHeadersPolicy.SECURITY_HEADERS,
                    cachePolicy: CachePolicy.CACHING_DISABLED,
                    allowedMethods: AllowedMethods.ALLOW_ALL,
                    trustedKeyGroups: [keyGroup],
                }
            });
            const cfnDistribution = distribution.node.defaultChild as CfnDistribution;
            cfnDistribution.addPropertyOverride('DistributionConfig.Origins.0.OriginAccessControlId', oacForDefaultOrigin.attrId);

            webServer.addPermission('AllowCloudFrontServicePrincipal', {
                principal: new ServicePrincipal('cloudfront.amazonaws.com'),
                action: 'lambda:InvokeFunctionUrl',
                sourceArn: `arn:aws:cloudfront::${accountId}:distribution/${distribution.distributionId}`,
            });

            new CfnOutput(parent, 'EntryPointUrl', { value: `https://${distribution.domainName}/` });
        })();
    }
}