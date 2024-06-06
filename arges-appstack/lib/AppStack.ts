import { CfnOutput, Stack, StackProps } from "aws-cdk-lib";
import { Construct } from "constructs";
import { AuthProvider } from "./AuthProvider";
import { EdgeLocation } from "./EdgeLocation";
import { SecretsAndParams } from "./SecretsAndParams";
import { WebContents } from "./WebContents";
import { WebServer } from "./WebServer";

export class AppStack extends Stack {

    constructor(scope: Construct, id: string, props?: StackProps) {
        super(scope, id, props);

        const authProvider = new AuthProvider(this, 'AuthProvider');
        const secretsAndParams = new SecretsAndParams(this, 'SecretsAndParams', { authProvider, });

        const webContents = new WebContents(this, 'WebContents', {});
        const webServer = new WebServer(this, 'WebServer', { authProvider, secretsAndParams, });

        const edgeLocation = new EdgeLocation(this, 'EdgeLocation', {
            authProvider, secretsAndParams, webServer, webContents,
        });

        new CfnOutput(this, 'WebappLocation', { value: edgeLocation.signInUrl! });
    }
}