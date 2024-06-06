import { Duration, RemovalPolicy, ScopedAws } from "aws-cdk-lib";
import { Effect, PolicyStatement } from "aws-cdk-lib/aws-iam";
import { Alias, FunctionUrlAuthType, IFunction, IFunctionUrl, InvokeMode, ParamsAndSecretsLayerVersion, ParamsAndSecretsVersions, Runtime } from "aws-cdk-lib/aws-lambda";
import { NodejsFunction } from "aws-cdk-lib/aws-lambda-nodejs";
import { LogGroup, RetentionDays } from "aws-cdk-lib/aws-logs";
import { Construct } from "constructs";
import { AuthProvider } from "./AuthProvider";
import { SecretsAndParams } from "./SecretsAndParams";

export class WebServer extends Construct {

    constructor(scope: Construct, id: string, props: {
        secretsAndParams: SecretsAndParams,
        authProvider: AuthProvider,
    }) {
        super(scope, id);
        const { region, accountId } = new ScopedAws(this);

        const secretName = props.secretsAndParams.secretName;
        const webServer = new NodejsFunction(this, 'Handler', {
            runtime: Runtime.NODEJS_20_X,
            entry: `${__dirname}/../../arges-webapp/.output/server/index.mjs`,
            handler: 'handler',
            timeout: Duration.seconds(10),
            paramsAndSecrets: ParamsAndSecretsLayerVersion.fromVersion(ParamsAndSecretsVersions.V1_0_103),
            environment: {
                'SECRET_NAME': secretName,
                'KEY_PAIR_ID': props.secretsAndParams.publicKey.publicKeyId,
                'USER_POOL_ID': props.authProvider.userPool.userPoolId,
            },
            logGroup: new LogGroup(this, 'HandlerLog', {
                removalPolicy: RemovalPolicy.DESTROY,
                retention: RetentionDays.ONE_DAY,
            }),
        });

        const current = new Alias(this, 'Current', {
            aliasName: 'current',
            version: webServer.currentVersion,
        });
        current.addToRolePolicy(new PolicyStatement({
            effect: Effect.ALLOW,
            actions: ['secretsmanager:GetSecretValue'],
            resources: [`arn:aws:secretsmanager:${region}:${accountId}:secret:${secretName}-*`],
        }));

        const endpoint = current.addFunctionUrl({
            invokeMode: InvokeMode.BUFFERED,
            authType: FunctionUrlAuthType.AWS_IAM,
        });

        this._handler = current;
        this._endpoint = endpoint;
    }

    private readonly _handler: IFunction;
    private readonly _endpoint: IFunctionUrl;

    get handler() { return this._handler; }
    get endpoint() { return this._endpoint; }
}