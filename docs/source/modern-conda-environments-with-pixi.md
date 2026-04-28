# Modern OpenFold conda environments with pixi

Pixi is a modern tool that allows for much faster environment building over conda, Pixes also comes with .lock environment files out of the box. 

Key terms when working with pixi
- an "environment", it's a group of "features", an example environment is `openfold3-cuda13-pypi`
- a "feature" is a group of packages/dependencies (either from conda or pypi)
- each environment is then composed from one or more features 

Here is an overview diagram showing the various environments and features currently available. 

```{figure} ../imgs/pixi-environments-and-features.png
:class: only-light
:alt: Pixi for OpenFold – environments and features
```