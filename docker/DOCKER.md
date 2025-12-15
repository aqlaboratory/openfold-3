## Updating the Lock File

When you modify `environments/production.yml`, you need to regenerate the lock file to pin exact versions. This ensures reproducible builds.

```bash
# Build the lock file generator image
docker build -f docker/Dockerfile.update-reqs -t openfold3-update-reqs .

# Generate the lock file (linux-64 only for now)
docker run --rm openfold3-update-reqs > environments/production.lock

# Commit the updated lock file
git add environments/production.lock
git commit -m "Update production.lock"
```

## Production images

TODO

For Blackwell image build, see [Build_instructions_blackwell.md](Build_instructions_blackwell.md)

## Development images

These images are the biggest but come with all the build tooling, needed to compile things at runtime (Deepspeed)

```bash
docker build \
    -f docker/Dockerfile \
    --target devel \
    -t openfold-docker:devel .
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
