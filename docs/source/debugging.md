# Debugging OpenFold

## Docker

### Prerequisites

- VS Code with the debugpy extension
- OpenFold docker image with `debugpy` installed


```bash
docker run --rm \
    --ipc=host \
    -it openfold-docker:devel \
    python -c "import debugpy; print(debugpy)"
...
<module 'debugpy' from '/opt/conda/envs/openfold3/lib/python3.12/site-packages/debugpy/__init__.py'>
```

### Step 1: Instrument your code

Put this snippet at the very top of your launch script 

```python
import debugpy
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger to attach...")
debugpy.wait_for_client()
```

For example, put this at the very top of `openfold3/run_openfold.py`


### Step 2: Create a launch configuration

Create `.vscode/launch.json` to tell VS Code to connect to `localhost:5678` on the docker container.

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Remote Attach",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder:openfold-3}",
                    "remoteRoot": "/opt/openfold3"
                }
            ]
        }
    ]
}
```

> **Note**: Use `${workspaceFolder}` instead if openfold-3 is your only workspace folder.

### Step 3: Launch your container

Launch the docker container with the debugger port exposed:

```bash
docker run --rm \
    --gpus all \
    --ipc=host \
    -p 5678:5678 \
    -it openfold-docker:devel \
    run_openfold predict \
        --query_json /data/query.json \
        --runner_yaml ./examples/example_runner_yamls/low_mem.yml \
        --output_dir /tmp/output
```

> **Important**: The `-p 5678:5678` flag is required to expose the debugger port.

The script will print "Waiting for debugger to attach..." and pause until you connect. 

### Step 4: Attach the debugger

In VS Code, open the Run and Debug panel (`Ctrl+Shift+D`) and click the green play button to attach. 
