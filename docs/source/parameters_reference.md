# OpenFold3 Parameters

The list of parameters available with OpenFold3 are as follows

```{list-table}
:header-rows: 1
:widths: 30 30 20 20

* - Checkpoint Name
  - File Name
  - Version compatibility
  - Number of training steps
* - **openfold3-p2-155k (default)**
  - **`of3-p2-155k.pt`**
  - **`>=0.4.0`**
  - **155,000**
* - openfold3-p1 _(deprecated)_
  - `of3_ft3_v1.pt`
  - `<0.4.0`
  - 78,000
```

## Which parameters is OpenFold3 using?

If a specific parmater file path is provided, OpenFold3 will use that file. Otherwise, OpenFold3 will first check default directories for previously downloaded parameters. 

The following schematic describes the resolution of paths through searching default directories.
```{mermaid}

flowchart TD
    START([Start]) --> A{"<i>inference-ckpt-path<i> provided?"}
    A -- YES --> Z(["Use provided path: <i>inference-ckpt-path<i>"])
    A -- NO --> B{"<i>inference-ckpt-name<i> provided?"}
    B -- YES --> C["Use <i>inference-ckpt-name<i>"]
    B -- NO --> D[Use default model: <i>openfold3-p2-155k<i>]
    C --> E[Parameter path resolution]
    D --> E
    E --> F{"<i>$OPENFOLD_CACHE<i> set or <i>cache-path<i> specified"?}
    F -- YES --> G{<i>ckpt_root<i> file exists in <i>$OPENFOLD_CACHE?<i>}
    F -- NO --> H([Download parameters to <i>~/.openfold3<i>])
    G -- YES --> I([Download parameters to $OPENFOLD_CACHE/ckpt_root path])
    G -- NO --> J(["Download parameters to <i>$OPENFOLD_CACHE<i>"])

    classDef decision  fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef terminal  fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef download  fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef process   fill:#f3e8ff,stroke:#a855f7,color:#3b0764

    class A,B,F,G decision
    class Z,I terminal
    class H,J download
    class C,D,E process

```

