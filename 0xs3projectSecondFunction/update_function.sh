#!/bin/bash

# cleanup old shit
rm my-deployment-package.zip

# add all packages to my new zip
cd package
zip -r ../my-deployment-package.zip .

# add my function to the zip too
cd ..
zip -g my-deployment-package.zip lambda_function.py

# push this zip with my function and the packages up to lambda as the latest code
aws lambda update-function-code \
--function-name buyUSDTwithSUSD \
--zip-file fileb://my-deployment-package.zip

# (run me with sh update_function.sh)