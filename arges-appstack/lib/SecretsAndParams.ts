import { Names, SecretValue } from "aws-cdk-lib";
import { IPublicKey, PublicKey } from "aws-cdk-lib/aws-cloudfront";
import { Secret } from "aws-cdk-lib/aws-secretsmanager";
import { Construct } from "constructs";
import { AuthProvider } from "./AuthProvider";
import { KeyPairProvider } from "./KeyPairProvider";

export class SecretsAndParams extends Construct {

    constructor(scope: Construct, id: string, props: {
    }) {
        super(scope, id);

        const keyPairProvider = new KeyPairProvider(this, 'KeyPairProvider');

        const publicKey = new PublicKey(this, keyPairProvider.keyPairName, {
            encodedKey: keyPairProvider.publicKey,
        });

        const secretName = `${Names.uniqueResourceName(new Construct(this, 'SecretName'), {})}`;

        this._publicKey = publicKey;
        this._privateKey = keyPairProvider.privateKeyAsJsonString;
        this._secretName = secretName;
    }

    private readonly _publicKey: IPublicKey;
    private readonly _privateKey: SecretValue;
    private readonly _secretName: string;

    get publicKey() { return this._publicKey; }
    get secretName() { return this._secretName; }

    createSecret(authProvider: AuthProvider) {
        const secret = new Secret(this, 'Secret', {
            secretName: this._secretName,
            secretObjectValue: {
                privateKey: this._privateKey,
                clientId: SecretValue.unsafePlainText(authProvider.client!.userPoolClientId),
                clientSecret: authProvider.client!.userPoolClientSecret,
                redirectUrl: SecretValue.unsafePlainText(authProvider.redirectUrl!),
            },
        });
    }
}