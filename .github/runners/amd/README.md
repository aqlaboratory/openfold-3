# AMD self-hosted runner

The `test-pixi-amd` jobs in
[`ci-integration-test-pixi-amd-reusable.yml`](../../workflows/ci-integration-test-pixi-amd-reusable.yml)
run on a persistent AMD GPU box (`runs-on: [self-hosted, amd-gpu]`). This
directory holds what is needed to set that box up again.

| File | Purpose |
|---|---|
| `gha-runner@.service` | systemd user unit that starts a runner at boot, inside a `screen` session |
| `buildkitd.toml` | storage cap for the shared `openfold3-ci` buildx builder; applied by the workflow |

The runner lives in `~/actions-runner-public` under the runner user's home and
is registered to `aqlaboratory/openfold-3` as `omsf-amd-aupcloud`.

## Requirements on the box

- Docker with the buildx plugin, and `screen`.
- The runner user is in the `docker`, `render` and `video` groups (the last two
  give access to `/dev/kfd` and `/dev/dri`).
- No sudo is needed for anything below.

## Setting up a runner

1. Get a registration token from the repository's
   Settings → Actions → Runners → New self-hosted runner, and download the
   runner tarball linked there.

2. Unpack and register it. The only custom label the workflow needs is
   `amd-gpu`:

   ```bash
   mkdir ~/actions-runner-public && cd ~/actions-runner-public
   tar xzf ~/actions-runner-linux-x64-<version>.tar.gz
   ./config.sh --url https://github.com/aqlaboratory/openfold-3 \
               --token <TOKEN> \
               --name omsf-amd-aupcloud \
               --labels amd-gpu \
               --work _work --unattended
   ```

3. Install the unit and start the runner. The instance name after `@` is the
   runner's directory name:

   ```bash
   mkdir -p ~/.config/systemd/user
   cp gha-runner@.service ~/.config/systemd/user/
   loginctl enable-linger "$USER"      # start user services at boot, without a login
   systemctl --user daemon-reload
   systemctl --user enable --now gha-runner@actions-runner-public
   ```

4. Check that the runner shows as **Idle** on the repository's runner settings
   page.

The buildx builder needs no setup: the workflow creates `openfold3-ci` from
`buildkitd.toml` if it does not exist.

## Day-to-day

```bash
systemctl --user status 'gha-runner@*'               # runner state
screen -r actions-runner-public                      # attach to the console; detach with Ctrl-a d
systemctl --user restart gha-runner@actions-runner-public
systemctl --user stop gha-runner@actions-runner-public      # stays enabled: starts again at boot
systemctl --user disable gha-runner@actions-runner-public   # do not start at boot
```

Runner logs are in `<runner dir>/_diag/`.

## Known problems

**`A session for this runner already exists` after a restart.** Expected after
an unclean stop or reboot. The runner retries every 30 seconds and reconnects
by itself once GitHub drops the old session. It only needs attention if it
keeps repeating, which means a second process is using the same registration.

**Runner exits immediately after its directory was renamed or moved.** `bin`
and `externals` are symlinks with absolute paths. Repoint them, then restart
the service:

```bash
cd ~/actions-runner-public
ln -sfn "$PWD"/bin.<version> bin
ln -sfn "$PWD"/externals.<version> externals
```

**Checkout fails with `EACCES`.** The test container runs as root and can leave
root-owned files in the workspace. The workflow chowns the workspace before
checkout, so this should clear on the next run. By hand:

```bash
docker run --rm -v ~/actions-runner-public/_work:/w busybox chown -R "$(id -u):$(id -g)" /w
```

**`No space left on device`.** The workflow removes its test image after each
run and caps the builder cache, but check what is using the disk with
`docker system df`. To clear this project's images and the builder cache:

```bash
docker images --format '{{.ID}} {{.Repository}}' \
  | awk '$2 ~ /aqlaboratory\/openfold/ {print $1}' | sort -u | xargs -r docker rmi -f
docker buildx prune --builder openfold3-ci -af
```

The box is shared with other users. Do not run `docker system prune -a` or
`docker image prune -a`: they delete other users' images too.

**Changing the builder's cache limits.** `buildkitd.toml` is only read when the
builder is created. After editing it, run `docker buildx rm openfold3-ci` on
the box; the next job recreates the builder with the new limits.

**The runner disappeared from GitHub.** GitHub removes self-hosted runners that
have been offline for 14 days. Remove the runner's `.runner` and `.credentials*`
files and register it again from step 1.
