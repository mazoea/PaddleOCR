@echo off
REM Build and Deploy Script for AWS Lambda Container (Windows Batch)
REM This script builds the Docker image and pushes it to AWS ECR

setlocal enabledelayedexpansion

REM Configuration
if "%AWS_REGION%"=="" set AWS_REGION=eu-central-1
if "%AWS_ACCOUNT_ID%"=="" set AWS_ACCOUNT_ID=610654100262
if "%ECR_REPOSITORY%"=="" set ECR_REPOSITORY=te-server
if "%IMAGE_TAG%"=="" set IMAGE_TAG=tm_paddle_v3_test
set LAMBDAFNC="te-advent-tm-v2"

echo ==========================================
echo AWS Lambda Container Build ^& Deploy
echo ==========================================
echo Region: %AWS_REGION%
echo Account ID: %AWS_ACCOUNT_ID%
echo Repository: %ECR_REPOSITORY%
echo Image Tag: %IMAGE_TAG%
echo ==========================================
echo.

REM Check if AWS_ACCOUNT_ID is set
if "%AWS_ACCOUNT_ID%"=="your-account-id" (
    echo Error: Please set AWS_ACCOUNT_ID environment variable!
    echo Example: set AWS_ACCOUNT_ID=123456789012
    pause
    exit /b 1
)

cp -r ../___latin_PP-OCRv5_mobile_rec_infer ./latin_PP-OCRv5_mobile_rec_infer

REM Step 1: Build the Docker image
echo.
echo Step 1: Building Docker image...
docker build -f Dockerfile_ocr -t %ECR_REPOSITORY%:%IMAGE_TAG% .

if %ERRORLEVEL% neq 0 (
    echo Error: Docker build failed!
    pause
    exit /b 1
)


REM Step 2: Create ECR repository if it doesn't exist
echo.
echo Step 2: Creating ECR repository ^(if it doesn't exist^)...
aws ecr describe-repositories --repository-names %ECR_REPOSITORY% --region %AWS_REGION% >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Creating new ECR repository...
    aws ecr create-repository --repository-name %ECR_REPOSITORY% --region %AWS_REGION%
) else (
    echo ECR repository already exists.
)

REM Step 3: Authenticate Docker to ECR
echo.
echo Step 3: Authenticating Docker to ECR...
for /f "tokens=*" %%i in ('aws ecr get-login-password --region %AWS_REGION%') do set LOGIN_PASSWORD=%%i
echo %LOGIN_PASSWORD% | docker login --username AWS --password-stdin %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com

if %ERRORLEVEL% neq 0 (
    echo Error: Docker login failed!
    pause
    exit /b 1
)

REM Step 4: Tag the image
echo.
echo Step 4: Tagging Docker image...
docker tag %ECR_REPOSITORY%:%IMAGE_TAG% %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPOSITORY%:%IMAGE_TAG%

REM Step 5: Push the image to ECR
echo.
echo Step 5: Pushing Docker image to ECR...
docker push %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPOSITORY%:%IMAGE_TAG%

if %ERRORLEVEL% neq 0 (
    echo Error: Docker push failed!
    exit /b 1
)

REM Step 6: Get the image URI
set IMAGE_URI=%AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPOSITORY%:%IMAGE_TAG%

echo.
echo ==========================================
echo Build and Push Complete!
echo ==========================================
echo Image URI: %IMAGE_URI%
echo.
echo Next steps:
echo 1. Create or update Lambda function:
echo    aws lambda create-function ^
echo      --function-name text-detection ^
echo      --package-type Image ^
echo      --code ImageUri=%IMAGE_URI% ^
echo      --role arn:aws:iam::%AWS_ACCOUNT_ID%:role/lambda-execution-role ^
echo      --timeout 300 ^
echo      --memory-size 2048 ^
echo      --region %AWS_REGION%
echo.
echo 2. Or update existing function:
echo    aws lambda update-function-code ^
echo      --function-name text-detection ^
echo      --image-uri %IMAGE_URI% ^
echo      --region %AWS_REGION%
echo ==========================================

aws lambda update-function-code --function-name %LAMBDAFNC% --image-uri %IMAGE_URI% --region %AWS_REGION%

call aws lambda wait function-updated --function-name %LAMBDAFNC% --region %AWS_REGION% --no-cli-pager

rm -rf PP-OCRv5_mobile_rec_infer
pause

endlocal
