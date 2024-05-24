import { RemovalPolicy } from "aws-cdk-lib";
import { Mfa, OAuthScope, UserPool, UserPoolClient, UserPoolDomain } from "aws-cdk-lib/aws-cognito";
import { Construct } from "constructs";

export class AuthProvider extends Construct {

    constructor(scope: Construct, id: string) {
        super(scope, id);

        const userPool = new UserPool(this, 'UserPool', {
            selfSignUpEnabled: false,
            signInAliases: { email: true, username: false, },
            autoVerify: { email: true, },
            mfa: Mfa.OFF,
            removalPolicy: RemovalPolicy.DESTROY,
        });

        this._userPool = userPool;
    }

    private readonly _userPool;
    private _domain: UserPoolDomain | undefined;
    private _client: UserPoolClient | undefined;
    private _signInUrl: string | undefined;

    get domain() { return this._domain; }
    get client() { return this._client; }
    get signInUrl() {
        return this._domain && this._client && this._signInUrl
            ? this._domain.signInUrl(this._client, { redirectUri: this._signInUrl })
            : undefined;
    }

    addDomain(domainPrefix: string) {
        this._domain = this._userPool.addDomain('Domain', {
            cognitoDomain: {
                domainPrefix,
            },
        });
    }

    addClient(callbackUrl: string) {
        this._client = this._userPool.addClient('Client', {
            generateSecret: true,
            oAuth: {
                flows: {
                    authorizationCodeGrant: true,
                    implicitCodeGrant: false,
                },
                scopes: [OAuthScope.OPENID, OAuthScope.EMAIL],
                callbackUrls: [callbackUrl],
            },
            authFlows: {
                userPassword: true,
            },
        });
        this._signInUrl = callbackUrl;
    }
}    
