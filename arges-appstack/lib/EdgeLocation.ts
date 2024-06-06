import { Fn, ScopedAws } from "aws-cdk-lib";
import { AllowedMethods, CachePolicy, CfnDistribution, CfnOriginAccessControl, Distribution, KeyGroup, OriginRequestPolicy, ResponseHeadersPolicy, ViewerProtocolPolicy } from "aws-cdk-lib/aws-cloudfront";
import { FunctionUrlOrigin, S3Origin } from "aws-cdk-lib/aws-cloudfront-origins";
import { Effect, PolicyStatement, ServicePrincipal } from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";
import { AuthProvider } from "./AuthProvider";
import { SecretsAndParams } from "./SecretsAndParams";
import { WebContents } from "./WebContents";
import { WebServer } from "./WebServer";

export class EdgeLocation extends Construct {

    constructor(scope: Construct, id: string, props: {
        authProvider: AuthProvider;
        secretsAndParams: SecretsAndParams;
        webServer: WebServer;
        webContents: WebContents;
    }) {
        super(scope, id);
        const { accountId, region } = new ScopedAws(this);

        const keyGroup = new KeyGroup(this, 'KeyGroup', {
            items: [props.secretsAndParams.publicKey]
        });

        const oacLambda = new CfnOriginAccessControl(this, 'OACLambda', {
            originAccessControlConfig: {
                name: 'OACLambda',
                originAccessControlOriginType: 'lambda',
                signingBehavior: 'always',
                signingProtocol: 'sigv4',
            },
        });

        const oacS3 = new CfnOriginAccessControl(this, 'OACS3', {
            originAccessControlConfig: {
                name: 'OACS3',
                originAccessControlOriginType: 's3',
                signingBehavior: 'always',
                signingProtocol: 'sigv4',
            },
        });

        const callbackPath = '/sign-in';
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
                [callbackPath]: {
                    origin: new FunctionUrlOrigin(props.webServer.endpoint),
                    viewerProtocolPolicy: ViewerProtocolPolicy.HTTPS_ONLY,
                    originRequestPolicy: OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                    responseHeadersPolicy: ResponseHeadersPolicy.SECURITY_HEADERS,
                    cachePolicy: CachePolicy.CACHING_DISABLED,
                    allowedMethods: AllowedMethods.ALLOW_GET_HEAD,
                },
                '*.*': {
                    origin: new S3Origin(props.webContents.contents, { originPath: 'public', }),
                    viewerProtocolPolicy: ViewerProtocolPolicy.HTTPS_ONLY,
                    originRequestPolicy: OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                    responseHeadersPolicy: ResponseHeadersPolicy.SECURITY_HEADERS,
                    cachePolicy: CachePolicy.CACHING_DISABLED,
                    allowedMethods: AllowedMethods.ALLOW_GET_HEAD,
                    trustedKeyGroups: [keyGroup],
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

        cfnDistribution.addPropertyOverride('DistributionConfig.Origins.2.S3OriginConfig.OriginAccessIdentity', '');
        cfnDistribution.addPropertyOverride('DistributionConfig.Origins.2.OriginAccessControlId', oacS3.attrId);
        props.webContents.contents.addToResourcePolicy(new PolicyStatement({
            effect: Effect.ALLOW,
            actions: ['s3:GetObject'],
            principals: [new ServicePrincipal('cloudfront.amazonaws.com')],
            resources: [props.webContents.contents.arnForObjects('*')],
        }));

        props.authProvider.addDomain(Fn.select(0, Fn.split('.', distribution.domainName)));
        props.authProvider.addClient(`https://${distribution.domainName}${callbackPath}`);
        props.secretsAndParams.createSecret(props.authProvider);

        this._signInUrl = props.authProvider.signInUrl;
    }

    private readonly _signInUrl: string | undefined;

    get signInUrl() { return this._signInUrl; }
}