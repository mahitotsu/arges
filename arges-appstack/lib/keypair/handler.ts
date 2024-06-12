import { DeleteParameterCommand, PutParameterCommand, SSMClient } from "@aws-sdk/client-ssm";
import { CdkCustomResourceHandler } from "aws-lambda";
import { generateKeyPairSync } from 'crypto';

const client = new SSMClient();

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
            await client.send(new PutParameterCommand({
                Name: event.LogicalResourceId,
                Type: 'SecureString',
                Value: privateKey,
                Overwrite: true
            }));
            return {
                Data: {
                    publicKey,
                    privateKey: event.LogicalResourceId,
                }
            }
        case 'Delete':
            await client.send(new DeleteParameterCommand({
                Name: event.PhysicalResourceId
            }));
        default:
            return {};
    }
}