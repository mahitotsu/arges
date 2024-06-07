import { RemovalPolicy, Stack, StackProps } from "aws-cdk-lib";
import { LogGroup, RetentionDays } from "aws-cdk-lib/aws-logs";
import { BlockPublicAccess, Bucket } from "aws-cdk-lib/aws-s3";
import { BucketDeployment, Source } from "aws-cdk-lib/aws-s3-deployment";
import { Construct } from "constructs";

export class AppStack extends Stack {

    constructor(scope: Construct, id: string, props: StackProps) {
        super(scope, id, props);

        const publicAssetsBucket = new Bucket(this, 'PublicAssetsBucket', {
            removalPolicy: RemovalPolicy.DESTROY,
            autoDeleteObjects: true,
            publicReadAccess: false,
            blockPublicAccess: BlockPublicAccess.BLOCK_ALL,
        });
        new BucketDeployment(publicAssetsBucket, 'PublicAssetsDeployment', {
            destinationBucket: publicAssetsBucket,
            destinationKeyPrefix: 'public',
            sources: [Source.asset(`${__dirname}/../../arges-webapp/.output/public`)],
            memoryLimit: 512,
            logGroup: new LogGroup(publicAssetsBucket, 'PublicAssetsBucketDeploymentLogGroup', {
                removalPolicy: RemovalPolicy.DESTROY,
                retention: RetentionDays.ONE_DAY,
            })
        });
    }
}