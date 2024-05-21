import { CustomResource, Duration, RemovalPolicy, ScopedAws, Stack, StackProps } from "aws-cdk-lib";
import { Effect, PolicyStatement } from "aws-cdk-lib/aws-iam";
import { Runtime } from "aws-cdk-lib/aws-lambda";
import { NodejsFunction } from "aws-cdk-lib/aws-lambda-nodejs";
import { LogGroup, RetentionDays } from "aws-cdk-lib/aws-logs";
import { StringParameter } from "aws-cdk-lib/aws-ssm";
import { Provider } from "aws-cdk-lib/custom-resources";
import { Construct } from "constructs";

export class AppStack extends Stack {

    constructor(scope: Construct, id: string, props?: StackProps) {
        super(scope, id, props);
        const { accountId, region, stackName, } = new ScopedAws(this);

        // -----
        // generate and store a keypair
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
            });
            handler.addToRolePolicy(new PolicyStatement({
                effect: Effect.ALLOW,
                actions: ['ssm:PutParameter', 'ssm:DeleteParameter'],
                resources: [`arn:aws:ssm:${region}:${accountId}:parameter/${stackName}/*`],
            }));
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
    }
}