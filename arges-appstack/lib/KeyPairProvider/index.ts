import { CdkCustomResourceHandler } from "aws-lambda";
import { generateKeyPairSync } from "crypto";

export const handler: CdkCustomResourceHandler = async (event, context) => {

    const stackName = event.StackId.split('/')[1];
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
                Data: { publicKey, privateKey, }
            };

        case 'Update':
            return {};

        case 'Delete':
            return {};
    }
}
