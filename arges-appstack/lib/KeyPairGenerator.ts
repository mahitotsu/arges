import { CdkCustomResourceHandler } from "aws-lambda";
import { generateKeyPairSync } from "crypto";

export const handler: CdkCustomResourceHandler = async (event, context) => {

    switch (event.RequestType) {
        case 'Create':
            const { publicKey, privateKey } = generateKeyPairSync('rsa', {
                modulusLength: 2048,
                publicKeyEncoding: {
                    type: 'pkcs1',
                    format: 'pem'
                },
                privateKeyEncoding: {
                    type: 'pkcs1',
                    format: 'pem',
                }
            });
            return {
                Data: {
                    PublicKey: publicKey,
                    PrivateKey: privateKey
                }
            };
        case 'Update':
            return {};
        case 'Delete':
            return {};
    }
}