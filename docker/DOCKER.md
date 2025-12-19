## Generating and updating production.lock file

While a conda env can be created from `environments/production-linux-64.yml`, this causes the environment to be resolved from scratch everytime. 
For reproducible builds, one needs to generate a .lock file that exactly re-creates the environment.

When you modify `environments/production-linux-64.yml`, you need to regenerate the lock file to pin exact versions. This ensures reproducible builds, prevents conda from resolving the environment again. `environment/production.lock` is then used for 'stable' builds.

```bash
# Build the lock file generator image
docker build -f docker/Dockerfile.update-reqs -t openfold3-update-reqs .

# Generate the lock file (linux-64 only for now)
docker run -v $(pwd)/environments:/output --rm openfold3-update-reqs 

# Commit the updated lock file
git add environments/production-linux-64.lock
git commit -m "Update production-linux-64.lock"
```

## Development images

These images are the biggest but come with all the build tooling, needed to compile things at runtime (Deepspeed)

```bash
docker build \
    -f docker/Dockerfile \
    --target devel \
    -t openfold-docker:devel-yaml .
```

Or more explicitly

```bash
docker build \
    -f docker/Dockerfile \
    --build-arg BUILD_MODE=yaml \
    --build-arg CUDA_BASE_IMAGE_TAG=12.1.1-cudnn8-devel-ubuntu22.04 \
    --target devel \
    -t openfold-docker:devel-yaml .
```

## Test images

Build the test image, with additional test-only dependencies

```bash
docker build \
    -f docker/Dockerfile \
    --target test \
    -t openfold-docker:test .
```

Run the unit tests

```bash
docker run \
    --rm \
    -v $(pwd -P):/opt/openfold3 \
    -t openfold-docker:test \
    pytest openfold3/tests -vvv
```

## Production images

Build a 'stable' image with all the dependancies exactly pinned (production.lock)

```bash
docker build \
    -f docker/Dockerfile \
    --build-arg BUILD_MODE=lock \
    --build-arg CUDA_BASE_IMAGE_TAG=12.1.1-cudnn8-devel-ubuntu22.04 \
    --target devel \
    -t openfold-docker:devel-locked .
```

For Blackwell image build, see [Build_instructions_blackwell.md](Build_instructions_blackwell.md)


