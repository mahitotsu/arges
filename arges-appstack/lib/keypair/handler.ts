import { CdkCustomResourceHandler } from "aws-lambda";
import { generateKeyPairSync } from 'crypto';

export const handler: CdkCustomResourceHandler = async (event) => {

    switch (event.RequestType) {
        case 'Create':
            const { privateKey, publicKey } = generateKeyPairSync('rsa', {
                modulusLength: 2048,
                publicKeyEncoding: {
                    type: 'pkcs1',
                    format: 'pem'
                },
                privateKeyEncoding: {
                    type: 'pkcs1',
                    format: 'pem'
                }
            });
            return {
                Data: {
                    publicKey,
                    privateKey,
                }
            }
        default:
            return {};
    }
}