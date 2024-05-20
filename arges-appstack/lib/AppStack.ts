import { CustomResource, Fn, RemovalPolicy, SecretValue, Stack, StackProps } from "aws-cdk-lib";
import { Runtime } from "aws-cdk-lib/aws-lambda";
import { NodejsFunction } from "aws-cdk-lib/aws-lambda-nodejs";
import { LogGroup, RetentionDays } from "aws-cdk-lib/aws-logs";
import { Secret } from "aws-cdk-lib/aws-secretsmanager";
import { Provider } from "aws-cdk-lib/custom-resources";
import { Construct } from "constructs";

export class AppStack extends Stack {

    constructor(scope: Construct, id: string, props?: StackProps) {
        super(scope, id, props);

        // -----
        // prepare keypair
        // -----
        const { publicKey, privateKey } = (() => {
            const parent = new Construct(this, 'KeyPairGenerator');
            const handler = new NodejsFunction(parent, 'Handler', {
                runtime: Runtime.NODEJS_LATEST,
                entry: `${__dirname}/KeyPairGenerator.ts`,
                handler: 'handler',
                logGroup: new LogGroup(parent, 'HandlerLog', {
                    removalPolicy: RemovalPolicy.DESTROY,
                    retention: RetentionDays.ONE_DAY,
                })
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
            });
            return {
                publicKey: Fn.join('\\n', Fn.split('\n', resource.getAttString('PublicKey'))),
                privateKey: Fn.join('\\n', Fn.split('\n', resource.getAttString('PrivateKey'))),
            };
        })();

        const keyPairSecret = new Secret(this, 'KeyPair', {
            secretObjectValue: {
                publicKey: SecretValue.unsafePlainText(publicKey),
                privateKey: SecretValue.unsafePlainText(privateKey),
            }
        });

        // -----
        //
        // -----
    }
}