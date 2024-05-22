import { ScopedAws } from "aws-cdk-lib";
import { AllowedMethods, CachePolicy, CfnDistribution, CfnOriginAccessControl, Distribution, KeyGroup, OriginRequestPolicy, PublicKey, ResponseHeadersPolicy, ViewerProtocolPolicy } from "aws-cdk-lib/aws-cloudfront";
import { FunctionUrlOrigin } from "aws-cdk-lib/aws-cloudfront-origins";
import { ServicePrincipal } from "aws-cdk-lib/aws-iam";
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
            }
        });
        const cfnDistribution = distribution.node.defaultChild as CfnDistribution;
        const oacLambdaPermission = {
            principal: new ServicePrincipal('cloudfront.amazonaws.com'),
            action: 'lambda:InvokeFunctionUrl',
            sourceArn: `arn:aws:cloudfront::${accountId}:distribution/${distribution.distributionId}`,
        };

        cfnDistribution.addPropertyOverride('DistributionConfig.Origins.0.OriginAccessControlId', oacLambda.attrId);
        props.webServer.handler.addPermission('AllowCloudFrontServicePrincipal', oacLambdaPermission);
    }
}