# Usage: bash download_openfold_params.sh /path/to/download/directory
if ! command -v aws &> /dev/null; then
    echo $'Error: AWS CLI is not installed. Please check that your OpenFold environment is properly installed or see https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html for AWS CLI installation instructions' >&2
    exit 1
fi

set -e

# Parse arguments
DOWNLOAD_DIR=""
for arg in "$@"; do
    case $arg in
        --download_dir=*)
            DOWNLOAD_DIR="${arg#*=}"
            shift
            ;;
        *)
            echo "Error: Unknown argument: $arg" >&2
            echo "Usage: bash download_openfold_params.sh [--download_dir=/path/to/download/directory]" >&2
            exit 1
            ;;
    esac
done

# Use provided directory or default
if [[ -z "${DOWNLOAD_DIR}" ]]; then
    if [[ -z "${OPENFOLD_CACHE}" ]]; then
        DOWNLOAD_DIR="${HOME}/openfold3/"
    else
        DOWNLOAD_DIR="${OPENFOLD_CACHE}/"
    fi
    echo "No download directory provided. Using default: ${DOWNLOAD_DIR}"
fi

OPENFOLD_BUCKET="s3://openfold"
CHECKPOINT_PATH="openfold3_params/of3_ft3_v1.pt"
BASENAME=$(basename "${SOURCE_URL}")

mkdir -p "${DOWNLOAD_DIR}"
aws s3 cp "${OPENFOLD_BUCKET}/${CHECKPOINT_PATH}" ${DOWNLOAD_DIR} --no-sign-request