import { Duration, RemovalPolicy } from "aws-cdk-lib";
import { Alias, FunctionUrlAuthType, IFunction, IFunctionUrl, InvokeMode, Runtime } from "aws-cdk-lib/aws-lambda";
import { NodejsFunction, OutputFormat } from "aws-cdk-lib/aws-lambda-nodejs";
import { LogGroup, RetentionDays } from "aws-cdk-lib/aws-logs";
import { Construct } from "constructs";

export class WebServer extends Construct {

    constructor(scope: Construct, id: string, props: {}) {
        super(scope, id);

        const webServer = new NodejsFunction(this, 'Handler', {
            runtime: Runtime.NODEJS_LATEST,
            entry: `${__dirname}/../../arges-webapp/.output/server/index.mjs`,
            handler: 'handler',
            timeout: Duration.seconds(10),
            bundling: {
                format: OutputFormat.ESM,
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