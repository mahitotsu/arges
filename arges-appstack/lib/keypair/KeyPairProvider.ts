import { CustomResource, RemovalPolicy, ScopedAws, SecretValue } from "aws-cdk-lib";
import { Effect, PolicyStatement } from "aws-cdk-lib/aws-iam";
import { Runtime } from "aws-cdk-lib/aws-lambda";
import { NodejsFunction } from "aws-cdk-lib/aws-lambda-nodejs";
import { Provider } from "aws-cdk-lib/custom-resources";
import { Construct } from "constructs";

export class KeyPairProvider extends Construct {

    constructor(scope: Construct, id: string) {
        super(scope, id);
        const { accountId, region } = new ScopedAws(this);

        const handler = new NodejsFunction(this, 'Handler', {
            entry: `${__dirname}/handler.ts`,
            handler: 'handler',
            runtime: Runtime.NODEJS_20_X,
        });
        handler.addToRolePolicy(new PolicyStatement({
            effect: Effect.ALLOW,
            actions: ['ssm:PutParameter', 'ssm:DeleteParameter'],
            resources: [`arn:aws:ssm:${region}:${accountId}:parameter/*`],
        }))

        const provider = new Provider(this, 'Provider', {
            onEventHandler: handler,
        });
        const resource = new CustomResource(this, 'Resource', {
            serviceToken: provider.serviceToken,
            removalPolicy: RemovalPolicy.DESTROY,
        });

        this._publicKey = resource.getAttString('publicKey');
        this._privateKey = SecretValue.ssmSecure(resource.getAttString('privateKey'));
    }

    private readonly _publicKey: string;
    private readonly _privateKey: SecretValue;

    get publicKey(): string { return this._publicKey; }
    get privateKey(): SecretValue { return this._privateKey; }
}