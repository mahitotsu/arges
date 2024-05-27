import { CfnOutput, Stack, StackProps } from "aws-cdk-lib";
import { Construct } from "constructs";
import { AuthProvider } from "./AuthProvider";
import { EdgeLocation } from "./EdgeLocation";
import { KeyPairProvider } from "./KeyPairProvider";
import { WebContents } from "./WebContents";
import { WebServer } from "./WebServer";

export class AppStack extends Stack {

    constructor(scope: Construct, id: string, props?: StackProps) {
        super(scope, id, props);

        const keyPairProvider = new KeyPairProvider(this, 'KeyPairProvider');
        const webServer = new WebServer(this, 'WebServer', {});
        const webContents = new WebContents(this, 'WebContents', {});

        const authProvider = new AuthProvider(this, 'AuthProvider');
        const edgeLocation = new EdgeLocation(this, 'EdgeLocation', {
            keyPairProvider, authProvider, webServer, webContents,
        });

        new CfnOutput(this, 'WebappLocation', { value: edgeLocation.signInUrl! });
    }
}