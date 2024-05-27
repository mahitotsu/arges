import { RemovalPolicy } from "aws-cdk-lib";
import { LogGroup, RetentionDays } from "aws-cdk-lib/aws-logs";
import { BlockPublicAccess, Bucket } from "aws-cdk-lib/aws-s3";
import { BucketDeployment, Source } from "aws-cdk-lib/aws-s3-deployment";
import { Construct } from "constructs";

export class WebContents extends Construct {

    constructor(scope: Construct, id: string, props:{}) {
        super(scope, id);

        const contents = new Bucket(this, 'Contents', {
            publicReadAccess: false,
            blockPublicAccess: BlockPublicAccess.BLOCK_ALL,
            removalPolicy: RemovalPolicy.DESTROY,
            autoDeleteObjects: true,
        });
        new BucketDeployment(this, 'Deployment', {
            destinationBucket: contents,
            destinationKeyPrefix: 'public',
            memoryLimit: 512,
            sources: [Source.asset(`${__dirname}/../../arges-webapp/.output/public`)],
            logGroup: new LogGroup(this, 'DeploymentLog', {
                removalPolicy: RemovalPolicy.DESTROY,
                retention: RetentionDays.ONE_DAY,
            }),
        });

        this._contents = contents;
    }

    private readonly _contents;

    get contents() { return this._contents; }
}